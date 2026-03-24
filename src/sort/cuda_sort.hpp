#pragma once

#include <vector>

/**
 * Bitonic sort on the GPU using CUDA.
 *
 * Bitonic sort requires a power-of-two length, so non-power-of-two arrays are
 * padded with INT_MAX on the host before transfer, sorted on the device, then
 * truncated back to the original size after the device-to-host copy. Early
 * bitonic stages that fit inside one block are executed from shared memory;
 * later cross-block merge stages fall back to a global-memory kernel.
 *
 * @param arr        Array to sort in-place (host memory).
 * @param block_size CUDA threads per block.
 */
void cuda_bitonic_sort(std::vector<int>& arr, int block_size);
