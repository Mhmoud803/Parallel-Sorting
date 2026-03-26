#include "sort/omp_sort.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <omp.h>
using namespace std;

namespace {

void merge_ranges(vector<int>& arr,
                  vector<int>& scratch,
                  long long lo,
                  long long mid,
                  long long hi) {
    long long i = lo;
    long long j = mid;
    long long k = lo;

    while (i < mid && j < hi) {
        if (arr[i] <= arr[j]) {
            scratch[k++] = arr[i++];
        } else {
            scratch[k++] = arr[j++];
        }
    }

    while (i < mid) {
        scratch[k++] = arr[i++];
    }

    while (j < hi) {
        scratch[k++] = arr[j++];
    }

    for (long long idx = lo; idx < hi; ++idx) {
        arr[static_cast<ptrdiff_t>(idx)] = scratch[static_cast<ptrdiff_t>(idx)];
    }
}

void merge_sort_task(vector<int>& arr,
                     vector<int>& scratch,
                     long long lo,
                     long long hi,
                     long long serial_cutoff) {
    const long long len = hi - lo;
    if (len <= 1) {
        return;
    }

    if (len <= serial_cutoff) {
        sort(arr.begin() + static_cast<ptrdiff_t>(lo),
             arr.begin() + static_cast<ptrdiff_t>(hi));
        return;
    }

    const long long mid = lo + (len / 2);
    const bool spawn_tasks = len >= (serial_cutoff << 1);

    #pragma omp task shared(arr, scratch) if(spawn_tasks)
    merge_sort_task(arr, scratch, lo, mid, serial_cutoff);

    #pragma omp task shared(arr, scratch) if(spawn_tasks)
    merge_sort_task(arr, scratch, mid, hi, serial_cutoff);

    #pragma omp taskwait
    merge_ranges(arr, scratch, lo, mid, hi);
}

}  // namespace

void omp_merge_sort(vector<int>& arr, int num_threads, int cutoff) {
    if (num_threads <= 0) {
        throw invalid_argument("omp_merge_sort: num_threads must be > 0");
    }

    if (cutoff <= 0) {
        throw invalid_argument("omp_merge_sort: cutoff must be > 0");
    }

    if (arr.size() <= 1) {
        return;
    }

    vector<int> scratch(arr.size());
    const long long serial_cutoff = static_cast<long long>(cutoff);

    omp_set_num_threads(num_threads);

    #pragma omp parallel
    {
        #pragma omp single nowait
        merge_sort_task(arr,
                        scratch,
                        0LL,
                        static_cast<long long>(arr.size()),
                        serial_cutoff);
    }
}
