#include "sort/serial_sort.hpp"

#include <iostream>
using namespace std;

namespace {

void serial_merge_sort_recursive(vector<int>& arr,
                                 vector<int>& scratch,
                                 size_t left,
                                 size_t right) {
    // Sort range [left, right) using divide-and-conquer.
    if (right - left <= 1) {
        return;
    }

    const size_t mid = left + (right - left) / 2;
    serial_merge_sort_recursive(arr, scratch, left, mid);
    serial_merge_sort_recursive(arr, scratch, mid, right);

    size_t i = left;
    size_t j = mid;
    size_t k = left;

    while (i < mid && j < right) {
        if (arr[i] <= arr[j]) {
            scratch[k++] = arr[i++];
        } else {
            scratch[k++] = arr[j++];
        }
    }

    while (i < mid) {
        scratch[k++] = arr[i++];
    }

    while (j < right) {
        scratch[k++] = arr[j++];
    }

    for (size_t idx = left; idx < right; ++idx) {
        arr[idx] = scratch[idx];
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

    for (size_t i = 0; i < reference.size(); ++i) {
        if (reference[i] != result[i]) {
            cerr << "[VERIFY] First mismatch at index " << i
                      << ": expected " << reference[i]
                      << ", got "      << result[i]    << "\n";
            return false;
        }
    }

    return true;
}
