#include "cli/cli.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// ── Internal helpers ──────────────────────────────────────────────────────────

// Advance the index and return the next argument string.
// Throws if there is no next argument.
static std::string require_value(int& i, int argc, char* argv[]) {
    const std::string flag = argv[i];
    ++i;
    if (i >= argc) {
        throw std::invalid_argument("Flag '" + flag + "' requires a value");
    }
    return argv[i];
}

// ── Public API ────────────────────────────────────────────────────────────────

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    bool has_size = false;

    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];

        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);

        } else if (key == "--size") {
            const long long v = std::stoll(require_value(i, argc, argv));
            if (v <= 0) throw std::invalid_argument("--size must be > 0");
            cfg.size = v;
            has_size = true;

        } else if (key == "--distribution") {
            const std::string v = require_value(i, argc, argv);
            if      (v == "uniform")       cfg.dist = Distribution::Uniform;
            else if (v == "gaussian")      cfg.dist = Distribution::Gaussian;
            else if (v == "nearly_sorted") cfg.dist = Distribution::NearlySorted;
            else if (v == "reversed")      cfg.dist = Distribution::Reversed;
            else throw std::invalid_argument(
                "Unknown distribution '" + v +
                "'. Valid: uniform, gaussian, nearly_sorted, reversed");

        } else if (key == "--seed") {
            cfg.seed = static_cast<unsigned int>(
                std::stoul(require_value(i, argc, argv)));

        } else if (key == "--impl") {
            const std::string v = require_value(i, argc, argv);
            if      (v == "serial") cfg.impl = Implementation::Serial;
            else if (v == "omp")    cfg.impl = Implementation::OMP;
            else if (v == "cuda")   cfg.impl = Implementation::CUDA;
            else throw std::invalid_argument(
                "Unknown impl '" + v + "'. Valid: serial, omp, cuda");

        } else if (key == "--threads") {
            const int v = std::stoi(require_value(i, argc, argv));
            if (v <= 0) throw std::invalid_argument("--threads must be > 0");
            cfg.threads = v;

        } else if (key == "--cutoff") {
            const int v = std::stoi(require_value(i, argc, argv));
            if (v <= 0) throw std::invalid_argument("--cutoff must be > 0");
            cfg.cutoff = v;

        } else if (key == "--block-size") {
            const int v = std::stoi(require_value(i, argc, argv));
            if (v <= 0) throw std::invalid_argument("--block-size must be > 0");
            cfg.block_size = v;

        } else if (key == "--repeats") {
            const int v = std::stoi(require_value(i, argc, argv));
            if (v <= 0) throw std::invalid_argument("--repeats must be > 0");
            cfg.repeats = v;

        } else if (key == "--output") {
            cfg.output = require_value(i, argc, argv);

        } else {
            throw std::invalid_argument("Unknown flag '" + key + "'");
        }
    }

    if (!has_size) {
        throw std::invalid_argument("--size is required");
    }

    return cfg;
}

void print_usage(const char* prog) {
    std::cout <<
        "Usage: " << prog << " [OPTIONS]\n"
        "\n"
        "Required:\n"
        "  --size        INT    Number of elements to sort\n"
        "  --distribution STR   Input distribution:\n"
        "                         uniform | gaussian | nearly_sorted | reversed\n"
        "\n"
        "Optional:\n"
        "  --seed        INT    RNG seed                      (default: 42)\n"
        "  --impl        STR    Implementation: serial|omp|cuda (default: serial)\n"
        "  --threads     INT    OpenMP thread count            (default: 1)\n"
        "  --cutoff      INT    OpenMP serial cutoff           (default: 10000)\n"
        "  --block-size  INT    CUDA threads per block         (default: 256)\n"
        "  --repeats     INT    Timing repetitions to average  (default: 1)\n"
        "  --output      PATH   Append CSV result row to file\n"
        "  -h, --help           Show this message\n"
        "\n"
        "Examples:\n"
        "  " << prog << " --size 10000000 --distribution uniform --impl serial\n"
        "  " << prog << " --size 10000000 --distribution gaussian --impl omp "
                        "--threads 8 --cutoff 10000 --repeats 5 --output results/omp.csv\n"
        "  " << prog << " --size 16777216 --distribution reversed --impl cuda "
                        "--block-size 256 --output results/cuda.csv\n";
}

void print_config(const Config& cfg) {
    std::cout
        << "┌─ Configuration ─────────────────────────────┐\n"
        << "│  size         : " << std::setw(24) << cfg.size         << " │\n"
        << "│  distribution : " << std::setw(24) << dist_to_string(cfg.dist) << " │\n"
        << "│  seed         : " << std::setw(24) << cfg.seed         << " │\n"
        << "│  impl         : " << std::setw(24) << impl_to_string(cfg.impl) << " │\n"
        << "│  threads      : " << std::setw(24) << cfg.threads      << " │\n"
        << "│  cutoff       : " << std::setw(24) << cfg.cutoff       << " │\n"
        << "│  block_size   : " << std::setw(24) << cfg.block_size   << " │\n"
        << "│  repeats      : " << std::setw(24) << cfg.repeats      << " │\n"
        << "│  output       : " << std::setw(24) << (cfg.output.empty() ? "(none)" : cfg.output) << " │\n"
        << "└─────────────────────────────────────────────┘\n";
}

std::string dist_to_string(Distribution d) {
    switch (d) {
        case Distribution::Uniform:      return "uniform";
        case Distribution::Gaussian:     return "gaussian";
        case Distribution::NearlySorted: return "nearly_sorted";
        case Distribution::Reversed:     return "reversed";
    }
    return "unknown";
}

std::string impl_to_string(Implementation i) {
    switch (i) {
        case Implementation::Serial: return "serial";
        case Implementation::OMP:    return "omp";
        case Implementation::CUDA:   return "cuda";
    }
    return "unknown";
}
