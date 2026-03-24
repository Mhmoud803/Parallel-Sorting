#pragma once

#include <vector>

/**
 * Sort `arr` in-place using std::sort (introsort, O(n log n)).
 * Used as both the serial benchmark target and the reference for verification.
 */
void serial_sort(std::vector<int>& arr);

/**
 * Verify that `result` is element-wise identical to `reference`.
 *
 * Both vectors must have been produced from the same input; `reference`
 * should be the output of serial_sort and `result` the output under test.
 *
 * Prints the index and values of the first mismatch to stderr.
 *
 * @return true  if all elements match, false otherwise.
 */
bool verify_sorted(const std::vector<int>& reference,
                   const std::vector<int>& result);
