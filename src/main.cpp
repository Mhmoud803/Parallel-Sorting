/**
 * main.cpp — Project 2: OpenMP MergeSort vs CUDA Bitonic Sort
 *
 * Pipeline per run:
 *   1. Parse CLI arguments into a Config struct.
 *   2. Generate the input array according to size/distribution/seed.
 *   3. Compute the serial reference result (used for correctness verification).
 *   4. Run the requested implementation `cfg.repeats` times, timing each.
 *   5. Verify the final output against the reference.
 *   6. Optionally append a CSV row to --output.
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "cli/cli.hpp"
#include "data/generator.hpp"
#include "sort/cuda_sort.hpp"
#include "sort/omp_sort.hpp"
#include "sort/serial_sort.hpp"

namespace fs = std::filesystem;

// ── Timing helper ─────────────────────────────────────────────────────────────
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

static double elapsed_ms(TimePoint t0, TimePoint t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ── CSV output ────────────────────────────────────────────────────────────────
static void write_csv_row(const Config& cfg, double avg_ms) {
    if (cfg.output.empty()) return;

    // Write header only when the file is new / empty.
    const bool write_header =
        !fs::exists(cfg.output) || fs::file_size(cfg.output) == 0;

    std::ofstream out(cfg.output, std::ios::app);
    if (!out) {
        std::cerr << "[WARN] Cannot open output file: " << cfg.output << "\n";
        return;
    }

    if (write_header) {
        out << "impl,size,distribution,seed,threads,block_size,repeats,avg_ms\n";
    }

    out << impl_to_string(cfg.impl) << ","
        << cfg.size                 << ","
        << dist_to_string(cfg.dist) << ","
        << cfg.seed                 << ","
        << cfg.threads              << ","
        << cfg.block_size           << ","
        << cfg.repeats              << ","
        << std::fixed << std::setprecision(4) << avg_ms << "\n";

    std::cout << "[OUTPUT] Row appended to " << cfg.output << "\n";
}

// ── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    // ── 1. Parse CLI ──────────────────────────────────────────────────────────
    Config cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n\n";
        print_usage(argv[0]);
        return 1;
    }
    print_config(cfg);

    // ── 2. Generate input data ────────────────────────────────────────────────
    std::cout << "[INFO]  Generating " << cfg.size << " elements ("
              << dist_to_string(cfg.dist) << ", seed=" << cfg.seed << ") ...\n";

    const std::vector<int> original =
        generate_array(cfg.size, cfg.dist, cfg.seed);

    // ── 3. Serial reference (run once; not counted in benchmark timing) ───────
    std::cout << "[INFO]  Computing serial reference ...\n";
    std::vector<int> reference = original;
    serial_sort(reference);

    // ── 4. Benchmark loop ─────────────────────────────────────────────────────
    std::cout << "[INFO]  Running '" << impl_to_string(cfg.impl) << "' for "
              << cfg.repeats << " repeat(s) ...\n";

    double total_ms = 0.0;
    std::vector<int> arr;   // re-used each repeat

    for (int rep = 0; rep < cfg.repeats; ++rep) {
        arr = original;     // fresh copy — measures sort time only

        const TimePoint t0 = Clock::now();

        switch (cfg.impl) {
            case Implementation::Serial:
                serial_sort(arr);
                break;

            case Implementation::OMP:
                omp_merge_sort(arr, cfg.threads, cfg.cutoff);
                break;

            case Implementation::CUDA:
                cuda_bitonic_sort(arr, cfg.block_size);
                break;
        }

        const TimePoint t1 = Clock::now();
        const double ms = elapsed_ms(t0, t1);
        total_ms += ms;

        std::cout << "  [rep " << std::setw(3) << (rep + 1)
                  << "/" << cfg.repeats << "]  "
                  << std::fixed << std::setprecision(3) << ms << " ms\n";
    }

    const double avg_ms = total_ms / cfg.repeats;
    std::cout << "[RESULT] average = " << std::fixed << std::setprecision(3)
              << avg_ms << " ms  (total=" << total_ms << " ms)\n";

    // ── 5. Correctness verification ───────────────────────────────────────────
    if (!verify_sorted(reference, arr)) {
        std::cerr << "[ERROR] Verification FAILED — output does not match "
                     "the serial reference!\n";
        return 2;
    }
    std::cout << "[VERIFY] Correctness check PASSED.\n";

    // ── 6. Persist results ────────────────────────────────────────────────────
    write_csv_row(cfg, avg_ms);

    return 0;
}
