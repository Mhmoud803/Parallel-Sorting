#include "sort/serial_sort.hpp"

#include <algorithm>
#include <iostream>

void serial_sort(std::vector<int>& arr) {
    // std::sort uses introsort: O(n log n) worst-case, cache-friendly.
    std::sort(arr.begin(), arr.end());
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
