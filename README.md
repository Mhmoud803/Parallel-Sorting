# Project 2 — OpenMP MergeSort vs CUDA Bitonic Sort

Benchmarks a parallel **merge sort** (OpenMP) against a **bitonic sort** (CUDA), with a `std::sort`-based serial baseline for correctness verification and timing reference.

---

## Directory Layout

```text
project2-sort/
├── CMakeLists.txt
├── README.md
├── run_experiments.sh        # Reproducible experiment driver (supports nsys profiling)
├── results/                  # CSV output lives here (git-ignored)
└── src/
    ├── main.cpp              # Entry point: ties CLI → data → sort → verify → output
    ├── cli/
    │   ├── cli.hpp           # Config struct + parse_args declaration
    │   └── cli.cpp           # Argument parsing implementation
    ├── data/
    │   ├── generator.hpp     # generate_array declaration
    │   └── generator.cpp     # Uniform / Gaussian / NearlySorted / Reversed generators
    └── sort/
        ├── serial_sort.hpp   # serial_sort + verify_sorted declarations
        ├── serial_sort.cpp   # std::sort wrapper + element-wise verifier
        ├── omp_sort.hpp      # omp_merge_sort declaration  (stub)
        ├── omp_sort.cpp      # OpenMP merge sort           (stub — Phase 2)
        ├── cuda_sort.hpp     # cuda_bitonic_sort declaration (stub)
        └── cuda_sort.cu      # CUDA bitonic sort            (stub — Phase 2)
```

---

## Prerequisites

| Tool | Minimum version |
|------|-----------------|
| CMake | 3.18 |
| GCC / Clang | C++17 support |
| CUDA Toolkit | 11.0 |
| OpenMP | 4.5 |

---

## Build

### Quick Start (Recommended)

For most users, one command is enough after cloning:

```bash
cd project2-sort
bash run_experiments.sh
```

What this script does automatically:
- finds or builds `sort_bench`
- runs all benchmark phases (Serial, OpenMP, CUDA)
- merges CSV outputs into `results/results.csv`
- creates a local Python virtual environment in `.venv` (if missing)
- installs plotting dependencies (`pandas`, `matplotlib`, `seaborn`) in that venv
- generates plots in `results/plots/`

If your GPU architecture is not detected correctly by default (for example, on an RTX 30-series card), specify the architecture explicitly:

```bash
CUDA_ARCHITECTURES=86 bash run_experiments.sh
```

Common values: `75` (Turing), `80` (Ampere/A100), `86` (RTX 30xx), `89` (RTX 40xx).

---

### Manual Build (Alternative)

```bash
# 1. Clone / enter the project
cd project2-sort

# 2. Configure (defaulting to Ampere / RTX 30-series architecture)
cmake -S . -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES=86   # 75=Turing, 80=Ampere, 86=RTX30, 89=RTX40

# 3. Compile
cmake --build build -j$(nproc)

# The binary is at: build/sort_bench
```

---

## Run

### Synopsis

```text
./build/sort_bench \
    --size         <N>        \
    --distribution <dist>     \
    --seed         <INT>      \
    --impl         <impl>     \
    [--threads     <T>]       \
    [--block-size  <B>]       \
    [--repeats     <R>]       \
    [--output      <file.csv>]
```

### Flag Reference

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--size` | positive int | **required** | Number of elements |
| `--distribution` | `uniform` `gaussian` `nearly_sorted` `reversed` | `uniform` | Input distribution |
| `--seed` | int | `42` | RNG seed for reproducibility |
| `--impl` | `serial` `omp` `cuda` | `serial` | Sort implementation |
| `--threads` | int | `1` | OMP thread count |
| `--block-size` | int | `256` | CUDA threads per block |
| `--repeats` | int | `1` | Timing repetitions (averaged) |
| `--output` | path | *(none)* | Append CSV row to this file |

### Examples

```bash
# Serial baseline, 10 M elements, Gaussian distribution
./build/sort_bench --size 10000000 --distribution gaussian --impl serial --repeats 5

# OpenMP merge sort, 4 threads, uniform, write CSV
./build/sort_bench --size 10000000 --distribution uniform --impl omp \
    --threads 4 --repeats 5 --output results/omp.csv

# CUDA bitonic sort
./build/sort_bench --size 16777216 --distribution reversed --impl cuda \
    --block-size 256 --repeats 5 --output results/cuda.csv
```

### Reproduce All Experiments

```bash
bash run_experiments.sh
```

Results are written to `results/` as CSV files suitable for plotting.

---

## Output Format

Each `--output` file is a CSV with one row per invocation:

```text
impl,size,distribution,seed,threads,block_size,repeats,avg_ms
serial,10000000,gaussian,42,1,256,5,342.17
```

---


