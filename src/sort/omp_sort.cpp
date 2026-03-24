#include "sort/omp_sort.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <omp.h>

namespace {

void merge_ranges(std::vector<int>& arr,
                  std::vector<int>& scratch,
                  std::size_t lo,
                  std::size_t mid,
                  std::size_t hi) {
    std::merge(arr.begin() + static_cast<std::ptrdiff_t>(lo),
               arr.begin() + static_cast<std::ptrdiff_t>(mid),
               arr.begin() + static_cast<std::ptrdiff_t>(mid),
               arr.begin() + static_cast<std::ptrdiff_t>(hi),
               scratch.begin() + static_cast<std::ptrdiff_t>(lo));

    std::copy(scratch.begin() + static_cast<std::ptrdiff_t>(lo),
              scratch.begin() + static_cast<std::ptrdiff_t>(hi),
              arr.begin() + static_cast<std::ptrdiff_t>(lo));
}

void merge_sort_task(std::vector<int>& arr,
                     std::vector<int>& scratch,
                     std::size_t lo,
                     std::size_t hi,
                     std::size_t serial_cutoff) {
    const std::size_t len = hi - lo;
    if (len <= 1) {
        return;
    }

    if (len <= serial_cutoff) {
        std::sort(arr.begin() + static_cast<std::ptrdiff_t>(lo),
                  arr.begin() + static_cast<std::ptrdiff_t>(hi));
        return;
    }

    const std::size_t mid = lo + (len / 2);
    const bool spawn_tasks = len >= (serial_cutoff << 1);

    #pragma omp task shared(arr, scratch) if(spawn_tasks)
    merge_sort_task(arr, scratch, lo, mid, serial_cutoff);

    #pragma omp task shared(arr, scratch) if(spawn_tasks)
    merge_sort_task(arr, scratch, mid, hi, serial_cutoff);

    #pragma omp taskwait
    merge_ranges(arr, scratch, lo, mid, hi);
}

}  // namespace

void omp_merge_sort(std::vector<int>& arr, int num_threads, int cutoff) {
    if (num_threads <= 0) {
        throw std::invalid_argument("omp_merge_sort: num_threads must be > 0");
    }

    if (cutoff <= 0) {
        throw std::invalid_argument("omp_merge_sort: cutoff must be > 0");
    }

    if (arr.size() <= 1) {
        return;
    }

    std::vector<int> scratch(arr.size());
    const std::size_t serial_cutoff = static_cast<std::size_t>(cutoff);

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        merge_sort_task(arr, scratch, 0, arr.size(), serial_cutoff);
    }
}
