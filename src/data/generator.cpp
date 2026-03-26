#include "data/generator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using namespace std;

// ── Private generators ────────────────────────────────────────────────────────

// Uniform integers drawn from the full int range (halved to avoid overflow
// when algorithms compute midpoints or deltas).
static vector<int> gen_uniform(long long size, mt19937& rng) {
    constexpr int lo = numeric_limits<int>::min() / 2;
    constexpr int hi = numeric_limits<int>::max() / 2;
    uniform_int_distribution<int> dist(lo, hi);

    vector<int> arr(static_cast<size_t>(size));
    for (auto& x : arr) x = dist(rng);
    return arr;
}

// Gaussian (normal) distribution with mean=0, stddev=1e6, clamped to int range.
// Produces a bell-shaped workload with many clustered keys.
static vector<int> gen_gaussian(long long size, mt19937& rng) {
    normal_distribution<double> dist(0.0, 1'000'000.0);
    const double lo = static_cast<double>(numeric_limits<int>::min());
    const double hi = static_cast<double>(numeric_limits<int>::max());

    vector<int> arr(static_cast<size_t>(size));
    for (auto& x : arr) {
        const double v = clamp(dist(rng), lo, hi);
        x = static_cast<int>(v);
    }
    return arr;
}

// Nearly sorted: consecutive integers with ~1% random pairwise swaps.
// Tests algorithms that exploit existing order (adaptive sorts).
static vector<int> gen_nearly_sorted(long long size, mt19937& rng) {
    vector<int> arr(static_cast<size_t>(size));
    iota(arr.begin(), arr.end(), 0);  // 0, 1, 2, ..., size-1

    const long long num_swaps = max(1LL, size / 100);
    uniform_int_distribution<long long> idx(0, size - 1);
    for (long long s = 0; s < num_swaps; ++s) {
        swap(arr[static_cast<size_t>(idx(rng))],
             arr[static_cast<size_t>(idx(rng))]);
    }
    return arr;
}

// Reversed: size-1, size-2, ..., 1, 0.
// Worst case for naive algorithms that assume partial order.
static vector<int> gen_reversed(long long size) {
    vector<int> arr(static_cast<size_t>(size));
    iota(arr.begin(), arr.end(), 0);
    reverse(arr.begin(), arr.end());
    return arr;
}

// ── Public API ────────────────────────────────────────────────────────────────

vector<int> generate_array(long long size,
                           Distribution dist,
                           unsigned int seed) {
    if (size <= 0) throw invalid_argument("generate_array: size must be > 0");

    mt19937 rng(seed);

    switch (dist) {
        case Distribution::Uniform:      return gen_uniform(size, rng);
        case Distribution::Gaussian:     return gen_gaussian(size, rng);
        case Distribution::NearlySorted: return gen_nearly_sorted(size, rng);
        case Distribution::Reversed:     return gen_reversed(size);
    }

    // Unreachable; silence compiler warnings.
    throw invalid_argument("generate_array: unknown distribution");
}
