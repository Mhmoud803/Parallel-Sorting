#include "sort/serial_sort.hpp"

#include <iostream>
using namespace std;

namespace {

void serial_merge_sort_recursive(vector<int>& arr,
                                 vector<int>& scratch,
                                 long long left,
                                 long long right) {
    // Sort range [left, right) using divide-and-conquer.
    if (right - left <= 1) {
        return;
    }

    const long long mid = left + (right - left) / 2;
    serial_merge_sort_recursive(arr, scratch, left, mid);
    serial_merge_sort_recursive(arr, scratch, mid, right);

    long long i = left;
    long long j = mid;
    long long k = left;

    while (i < mid && j < right) {
        if (arr[static_cast<ptrdiff_t>(i)] <= arr[static_cast<ptrdiff_t>(j)]) {
            scratch[static_cast<ptrdiff_t>(k++)] = arr[static_cast<ptrdiff_t>(i++)];
        } else {
            scratch[static_cast<ptrdiff_t>(k++)] = arr[static_cast<ptrdiff_t>(j++)];
        }
    }

    while (i < mid) {
        scratch[static_cast<ptrdiff_t>(k++)] = arr[static_cast<ptrdiff_t>(i++)];
    }

    while (j < right) {
        scratch[static_cast<ptrdiff_t>(k++)] = arr[static_cast<ptrdiff_t>(j++)];
    }

    for (long long idx = left; idx < right; ++idx) {
        arr[static_cast<ptrdiff_t>(idx)] = scratch[static_cast<ptrdiff_t>(idx)];
    }
}

} // namespace

void serial_sort(vector<int>& arr) {
    if (arr.size() <= 1) {
        return;
    }

    vector<int> scratch(arr.size());
    serial_merge_sort_recursive(arr, scratch, 0, arr.size());
}

bool verify_sorted(const vector<int>& reference,
                   const vector<int>& result) {
    if (reference.size() != result.size()) {
        cerr << "[VERIFY] Size mismatch: reference=" << reference.size()
                  << "  result=" << result.size() << "\n";
        return false;
    }

    for (long long i = 0; i < static_cast<long long>(reference.size()); ++i) {
        if (reference[static_cast<ptrdiff_t>(i)] != result[static_cast<ptrdiff_t>(i)]) {
            cerr << "[VERIFY] First mismatch at index " << i
                 << ": expected " << reference[static_cast<ptrdiff_t>(i)]
                 << ", got "      << result[static_cast<ptrdiff_t>(i)]    << "\n";
            return false;
        }
    }

    return true;
}
