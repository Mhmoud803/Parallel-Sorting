#pragma once

#include <vector>
#include "cli/cli.hpp"  // Distribution enum

/**
 * Generate a synthetic array of `size` integers.
 *
 * @param size  Number of elements to generate.
 * @param dist  Which statistical distribution to use.
 * @param seed  RNG seed for full reproducibility.
 * @return      Freshly allocated vector of length `size`.
 */
std::vector<int> generate_array(long long size, Distribution dist, unsigned int seed);
