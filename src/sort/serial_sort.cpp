#include "sort/serial_sort.hpp"

#include <algorithm>
#include <iostream>

namespace {

void serial_merge_sort_recursive(std::vector<int>& arr,
                                 std::vector<int>& scratch,
                                 std::size_t left,
                                 std::size_t right) {
    // Sort range [left, right) using divide-and-conquer.
    if (right - left <= 1) {
        return;
    }

    const std::size_t mid = left + (right - left) / 2;
    serial_merge_sort_recursive(arr, scratch, left, mid);
    serial_merge_sort_recursive(arr, scratch, mid, right);

    std::merge(arr.begin() + static_cast<std::ptrdiff_t>(left),
               arr.begin() + static_cast<std::ptrdiff_t>(mid),
               arr.begin() + static_cast<std::ptrdiff_t>(mid),
               arr.begin() + static_cast<std::ptrdiff_t>(right),
               scratch.begin() + static_cast<std::ptrdiff_t>(left));

    std::copy(scratch.begin() + static_cast<std::ptrdiff_t>(left),
              scratch.begin() + static_cast<std::ptrdiff_t>(right),
              arr.begin() + static_cast<std::ptrdiff_t>(left));
}

} // namespace

void serial_sort(std::vector<int>& arr) {
    if (arr.size() <= 1) {
        return;
    }

    std::vector<int> scratch(arr.size());
    serial_merge_sort_recursive(arr, scratch, 0, arr.size());
}

bool verify_sorted(const std::vector<int>& reference,
                   const std::vector<int>& result) {
    if (reference.size() != result.size()) {
        std::cerr << "[VERIFY] Size mismatch: reference=" << reference.size()
                  << "  result=" << result.size() << "\n";
        return false;
    }

    for (std::size_t i = 0; i < reference.size(); ++i) {
        if (reference[i] != result[i]) {
            std::cerr << "[VERIFY] First mismatch at index " << i
                      << ": expected " << reference[i]
                      << ", got "      << result[i]    << "\n";
            return false;
        }
    }

    return true;
}
