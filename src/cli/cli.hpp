#pragma once

#include <string>
using namespace std;

// ── Distribution types ────────────────────────────────────────────────────────
enum class Distribution {
    Uniform,       // i.i.d. uniform integers
    Gaussian,      // normal distribution, mapped to int
    NearlySorted,  // sorted with ~1% random swaps
    Reversed,      // reverse-sorted (worst case for many algorithms)
};

// ── Sort implementations ──────────────────────────────────────────────────────
enum class Implementation {
    Serial,  // serial baseline
    OMP,     // OpenMP parallel merge sort
    CUDA,    // CUDA bitonic sort
};

// ── Resolved program configuration ────────────────────────────────────────────
struct Config {
    long long      size       = 0;                    // --size
    Distribution   dist       = Distribution::Uniform; // --distribution
    unsigned int   seed       = 42;                   // --seed
    Implementation impl       = Implementation::Serial; // --impl
    int            threads    = 1;                    // --threads
    int            cutoff     = 10000;                // --cutoff
    int            block_size = 256;                  // --block-size
    int            repeats    = 1;                    // --repeats
    string         output     = "";                   // --output (CSV path)
};

// Parse command-line arguments into a Config.
// Throws invalid_argument with a descriptive message on bad input.
Config parse_args(int argc, char* argv[]);

// Pretty-print the resolved configuration to stdout.
void print_config(const Config& cfg);

// Print a usage/help string to stdout.
void print_usage(const char* prog);

// String conversions used when writing CSV rows.
string dist_to_string(Distribution d);
string impl_to_string(Implementation i);
