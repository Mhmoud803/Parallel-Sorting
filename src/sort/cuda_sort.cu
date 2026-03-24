// cuda_sort.cu — CUDA bitonic sort
//
// Bitonic sort requires the logical array length to be a power of two.
// For arbitrary input sizes we pad on the host with INT_MAX up to the next
// power of two, sort the padded buffer on the device, then copy only the
// original prefix back to the caller. Since INT_MAX compares after all valid
// inputs in ascending order, the padded sentinels naturally drift to the end.

#include "sort/cuda_sort.hpp"

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

inline void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) {
        return;
    }

    std::ostringstream oss;
    oss << file << ":" << line << " CUDA call failed: " << expr
        << " -> " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

std::size_t next_power_of_two(std::size_t n) {
    if (n <= 1) {
        return 1;
    }

    --n;
    for (std::size_t shift = 1; shift < (sizeof(std::size_t) * 8); shift <<= 1) {
        n |= (n >> shift);
    }
    return n + 1;
}

__device__ inline std::size_t pair_left_index(std::size_t pair_idx, std::size_t stride) {
    return ((pair_idx / stride) * (stride << 1)) + (pair_idx % stride);
}

__global__ void shared_bitonic_tile_kernel(int* data, std::size_t n) {
    extern __shared__ int tile[];

    const std::size_t threads_per_block = static_cast<std::size_t>(blockDim.x);
    const std::size_t tile_size = threads_per_block << 1;
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * tile_size;
    const std::size_t tid = static_cast<std::size_t>(threadIdx.x);

    const std::size_t idx0 = base + tid;
    const std::size_t idx1 = idx0 + threads_per_block;

    tile[tid] =
        (idx0 < n) ? data[idx0] : INT_MAX;
    tile[tid + threads_per_block] =
        (idx1 < n) ? data[idx1] : INT_MAX;

    __syncthreads();

    // This kernel executes the complete bitonic schedule for a tile using the
    // same index rules as the global algorithm. The tile direction therefore
    // alternates by block, which makes it a valid starting point for the later
    // cross-block merge stages in global memory.
    for (std::size_t k = 2; k <= tile_size; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {
            const std::size_t left = pair_left_index(tid, j);
            const std::size_t right = left + j;
            const bool ascending = (((base + left) & k) == 0);

            const int lhs = tile[left];
            const int rhs = tile[right];
            if ((ascending && lhs > rhs) || (!ascending && lhs < rhs)) {
                tile[left] = rhs;
                tile[right] = lhs;
            }

            __syncthreads();
        }
    }

    if (idx0 < n) {
        data[idx0] = tile[tid];
    }
    if (idx1 < n) {
        data[idx1] = tile[tid + threads_per_block];
    }
}

__global__ void bitonic_step_kernel(int* data, std::size_t j, std::size_t k, std::size_t n) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
        static_cast<std::size_t>(threadIdx.x);

    if (idx >= n) {
        return;
    }

    const std::size_t partner = idx ^ j;
    if (partner <= idx || partner >= n) {
        return;
    }

    const bool ascending = ((idx & k) == 0);
    const int lhs = data[idx];
    const int rhs = data[partner];

    if ((ascending && lhs > rhs) || (!ascending && lhs < rhs)) {
        data[idx] = rhs;
        data[partner] = lhs;
    }
}

}  // namespace

void cuda_bitonic_sort(std::vector<int>& arr, int block_size) {
    if (block_size <= 0) {
        throw std::invalid_argument("cuda_bitonic_sort: block_size must be > 0");
    }

    if (arr.size() <= 1) {
        return;
    }

    if (block_size > 1024) {
        throw std::invalid_argument("cuda_bitonic_sort: block_size must be <= 1024");
    }

    if ((block_size & (block_size - 1)) != 0) {
        throw std::invalid_argument("cuda_bitonic_sort: block_size must be a power of two");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw std::runtime_error("cuda_bitonic_sort: no CUDA-capable device available");
    }

    const std::size_t original_size = arr.size();
    const std::size_t padded_size = next_power_of_two(original_size);

    std::vector<int> host_buffer(padded_size, INT_MAX);
    std::copy(arr.begin(), arr.end(), host_buffer.begin());

    int* d_arr = nullptr;

    try {
        CUDA_CHECK(cudaMalloc(&d_arr, padded_size * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_arr,
                              host_buffer.data(),
                              padded_size * sizeof(int),
                              cudaMemcpyHostToDevice));

        const dim3 threads(static_cast<unsigned int>(block_size));
        const dim3 blocks(static_cast<unsigned int>((padded_size + block_size - 1) / block_size));
        const std::size_t local_span = static_cast<std::size_t>(block_size) << 1;
        const dim3 tile_blocks(static_cast<unsigned int>((padded_size + local_span - 1) / local_span));
        const std::size_t shared_bytes = local_span * sizeof(int);

        // Early stages are executed inside shared memory for each tile of
        // 2 * block_size elements. Once the bitonic sequence length exceeds the
        // tile size, compare-swap partners can span multiple blocks and we
        // transition to the global-memory kernel.
        shared_bitonic_tile_kernel<<<tile_blocks, threads, shared_bytes>>>(d_arr, padded_size);
        CUDA_CHECK(cudaPeekAtLastError());

        for (std::size_t k = local_span << 1; k <= padded_size; k <<= 1) {
            for (std::size_t j = k >> 1; j > 0; j >>= 1) {
                bitonic_step_kernel<<<blocks, threads>>>(d_arr, j, k, padded_size);
                CUDA_CHECK(cudaPeekAtLastError());
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_buffer.data(),
                              d_arr,
                              padded_size * sizeof(int),
                              cudaMemcpyDeviceToHost));
    } catch (...) {
        if (d_arr != nullptr) {
            cudaFree(d_arr);
        }
        throw;
    }

    CUDA_CHECK(cudaFree(d_arr));
    std::copy(host_buffer.begin(),
              host_buffer.begin() + static_cast<std::ptrdiff_t>(original_size),
              arr.begin());
}
