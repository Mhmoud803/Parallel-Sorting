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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
using namespace std;

namespace {

inline void throw_cuda_error(cudaError_t err, const char* expr, const char* file, int line) {
    if (err == cudaSuccess) {
        return;
    }

    ostringstream oss;
    oss << file << ":" << line << " CUDA call failed: " << expr
        << " -> " << cudaGetErrorString(err);
    throw runtime_error(oss.str());
}

#define CUDA_CHECK(expr) throw_cuda_error((expr), #expr, __FILE__, __LINE__)

long long next_power_of_two(long long n) {
    if (n <= 1) {
        return 1;
    }

    --n;
    for (long long shift = 1;
         shift < static_cast<long long>(sizeof(long long) * 8);
         shift <<= 1) {
        n |= (n >> shift);
    }
    return n + 1;
}

__device__ inline long long pair_left_index(long long pair_idx, long long stride) {
    return ((pair_idx / stride) * (stride << 1)) + (pair_idx % stride);
}

__global__ void shared_bitonic_tile_kernel(int* data, long long n) {
    extern __shared__ int tile[];

    const long long threads_per_block = static_cast<long long>(blockDim.x);
    const long long tile_size = threads_per_block << 1;
    const long long base = static_cast<long long>(blockIdx.x) * tile_size;
    const long long tid = static_cast<long long>(threadIdx.x);

    const long long idx0 = base + tid;
    const long long idx1 = idx0 + threads_per_block;

    tile[tid] =
        (idx0 < n) ? data[idx0] : INT_MAX;
    tile[tid + threads_per_block] =
        (idx1 < n) ? data[idx1] : INT_MAX;

    __syncthreads();

    // This kernel executes the complete bitonic schedule for a tile using the
    // same index rules as the global algorithm. The tile direction therefore
    // alternates by block, which makes it a valid starting point for the later
    // cross-block merge stages in global memory.
    for (long long k = 2; k <= tile_size; k <<= 1) {
        for (long long j = k >> 1; j > 0; j >>= 1) {
            const long long left = pair_left_index(tid, j);
            const long long right = left + j;
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

__global__ void bitonic_step_kernel(int* data, long long j, long long k, long long n) {
    const long long idx =
        static_cast<long long>(blockIdx.x) * static_cast<long long>(blockDim.x) +
        static_cast<long long>(threadIdx.x);

    if (idx >= n) {
        return;
    }

    const long long partner = idx ^ j;
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

void cuda_bitonic_sort(vector<int>& arr, int block_size) {
    if (block_size <= 0) {
        throw invalid_argument("cuda_bitonic_sort: block_size must be > 0");
    }

    if (arr.size() <= 1) {
        return;
    }

    if (block_size > 1024) {
        throw invalid_argument("cuda_bitonic_sort: block_size must be <= 1024");
    }

    if ((block_size & (block_size - 1)) != 0) {
        throw invalid_argument("cuda_bitonic_sort: block_size must be a power of two");
    }

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw runtime_error("cuda_bitonic_sort: no CUDA-capable device available");
    }

    const long long original_size = static_cast<long long>(arr.size());
    const long long padded_size = next_power_of_two(original_size);

    vector<int> host_buffer(padded_size, INT_MAX);
    copy(arr.begin(), arr.end(), host_buffer.begin());

    int* d_arr = nullptr;
    cudaEvent_t h2d_start = nullptr;
    cudaEvent_t h2d_stop = nullptr;
    cudaEvent_t compute_start = nullptr;
    cudaEvent_t compute_stop = nullptr;
    cudaEvent_t d2h_start = nullptr;
    cudaEvent_t d2h_stop = nullptr;

    auto destroy_event = [](cudaEvent_t& ev) {
        if (ev != nullptr) {
            cudaEventDestroy(ev);
            ev = nullptr;
        }
    };

    try {
        CUDA_CHECK(cudaEventCreate(&h2d_start));
        CUDA_CHECK(cudaEventCreate(&h2d_stop));
        CUDA_CHECK(cudaEventCreate(&compute_start));
        CUDA_CHECK(cudaEventCreate(&compute_stop));
        CUDA_CHECK(cudaEventCreate(&d2h_start));
        CUDA_CHECK(cudaEventCreate(&d2h_stop));

        CUDA_CHECK(cudaMalloc(&d_arr, padded_size * sizeof(int)));

        CUDA_CHECK(cudaEventRecord(h2d_start));
        CUDA_CHECK(cudaMemcpy(d_arr,
                              host_buffer.data(),
                              padded_size * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h2d_stop));
        CUDA_CHECK(cudaEventSynchronize(h2d_stop));

        const dim3 threads(static_cast<unsigned int>(block_size));
        const dim3 blocks(static_cast<unsigned int>((padded_size + block_size - 1) / block_size));
        const long long local_span = static_cast<long long>(block_size) << 1;
        const dim3 tile_blocks(static_cast<unsigned int>((padded_size + local_span - 1) / local_span));
        const long long shared_bytes = local_span * sizeof(int);

        CUDA_CHECK(cudaEventRecord(compute_start));

        // Early stages are executed inside shared memory for each tile of
        // 2 * block_size elements. Once the bitonic sequence length exceeds the
        // tile size, compare-swap partners can span multiple blocks and we
        // transition to the global-memory kernel.
        shared_bitonic_tile_kernel<<<tile_blocks, threads, shared_bytes>>>(d_arr, padded_size);
        CUDA_CHECK(cudaPeekAtLastError());

        for (long long k = local_span << 1; k <= padded_size; k <<= 1) {
            for (long long j = k >> 1; j > 0; j >>= 1) {
                bitonic_step_kernel<<<blocks, threads>>>(d_arr, j, k, padded_size);
                CUDA_CHECK(cudaPeekAtLastError());
            }
        }

        CUDA_CHECK(cudaEventRecord(compute_stop));
        CUDA_CHECK(cudaEventSynchronize(compute_stop));

        CUDA_CHECK(cudaEventRecord(d2h_start));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host_buffer.data(),
                              d_arr,
                              padded_size * sizeof(int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(d2h_stop));
        CUDA_CHECK(cudaEventSynchronize(d2h_stop));

        float h2d_ms = 0.0f;
        float compute_ms = 0.0f;
        float d2h_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2d_start, h2d_stop));
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms, compute_start, compute_stop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

        const double transfer_ms = static_cast<double>(h2d_ms) + static_cast<double>(d2h_ms);
        const double compute_ms_d = static_cast<double>(compute_ms);
        const double measured_total_ms = compute_ms_d + transfer_ms;
        const double compute_percentage = (measured_total_ms > 0.0)
            ? (compute_ms_d / measured_total_ms) * 100.0
            : 0.0;

        cout << fixed << setprecision(1)
             << "[CUDA Metrics] Compute Percentage: " << compute_percentage << "%"
             << " (Compute=" << compute_ms_d << " ms, Transfer=" << transfer_ms << " ms)\n";

        destroy_event(h2d_start);
        destroy_event(h2d_stop);
        destroy_event(compute_start);
        destroy_event(compute_stop);
        destroy_event(d2h_start);
        destroy_event(d2h_stop);
    } catch (...) {
        if (d_arr != nullptr) {
            cudaFree(d_arr);
        }

        destroy_event(h2d_start);
        destroy_event(h2d_stop);
        destroy_event(compute_start);
        destroy_event(compute_stop);
        destroy_event(d2h_start);
        destroy_event(d2h_stop);

        throw;
    }

    CUDA_CHECK(cudaFree(d_arr));
    copy(host_buffer.begin(),
         host_buffer.begin() + static_cast<ptrdiff_t>(original_size),
         arr.begin());
}
