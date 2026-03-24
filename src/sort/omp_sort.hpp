#pragma once

#include <vector>

/**
 * Parallel merge sort using OpenMP tasks.
 *
 * Large ranges are recursively split into two halves and processed with
 * OpenMP tasks. Small ranges switch to std::sort to avoid task-creation
 * overhead. A shared scratch buffer is used during merges, with each merge
 * operating on a disjoint subrange to avoid races.
 *
 * @param arr         Array to sort in-place.
 * @param num_threads Number of OpenMP threads to use.
 * @param cutoff      Serial fallback threshold used to limit task overhead.
 */
void omp_merge_sort(std::vector<int>& arr, int num_threads, int cutoff);
