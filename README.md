# Project 2 — OpenMP MergeSort vs CUDA Bitonic Sort

A High-Performance Computing (HPC) benchmarking project comparing a CPU-parallel **OpenMP MergeSort** against a GPU-accelerated **CUDA Bitonic Sort**. The parallel implementations are evaluated against a highly optimized, custom **Serial MergeSort** baseline for correctness verification and precise speedup calculations.

---

## 🚀 Key Architectural Optimizations

This project implements several advanced HPC techniques to maximize hardware utilization:

* **Optimized Memory Management (Serial & OpenMP):** Completely eliminates dynamic memory allocation inside the recursion tree. A single `scratch` array is allocated once in the wrapper function and passed by reference.
* **Manual Index-Based Merge:** Replaces abstraction-heavy `std::merge` with a highly efficient, custom `i, j, k` merge loop.
* **OpenMP Task Parallelism:** Utilizes `#pragma omp task` for recursive divide-and-conquer branching, coupled with an **Adaptive Serial Cutoff** to prevent task-creation overhead on small sub-arrays.
* **CUDA Shared Memory Tiling:** The Bitonic Sort network accelerates early, dense compare-and-swap stages by loading `2 * block_size` tiles into extremely fast `__shared__` memory before falling back to global memory for larger strides.
* **CUDA Arbitrary Size Padding:** Supports non-power-of-two array sizes (up to 67.1M+ elements) by implicitly padding the problem space with `INT_MAX` sentinels, which naturally migrate to the end of the sorted array without disrupting valid data.
* **Kernel Profiling:** Implements native `cudaEvent_t` timers to explicitly decouple and report pure GPU kernel compute percentage versus Host-to-Device / Device-to-Host memory transfer penalties.

---

## 📁 Directory Layout

```text
project2-sort/
├── CMakeLists.txt
├── README.md
├── run_experiments.sh        # Reproducible experiment driver (Compiles, Runs, Plots)
├── results/                  # CSV outputs and generated Matplotlib plots live here
└── src/
    ├── main.cpp              # Entry point: CLI args → Generate Data → Sort → Verify
    ├── cli/
    │   ├── cli.hpp           
    │   └── cli.cpp           # CLI Argument parsing implementation
    ├── data/
    │   ├── generator.hpp     
    │   └── generator.cpp     # Uniform / Gaussian / NearlySorted / Reversed generators
    └── sort/
        ├── serial_sort.hpp   
        ├── serial_sort.cpp   # Custom recursive MergeSort + element-wise verifier
        ├── omp_sort.hpp      
        ├── omp_sort.cpp      # Task-based OpenMP MergeSort with adaptive cutoff
        ├── cuda_sort.hpp     
        └── cuda_sort.cu      # Hybrid Shared/Global memory CUDA Bitonic Sort
```

---

## 🛠️ Prerequisites

| Tool | Minimum version |
|------|-----------------|
| CMake | 3.18 |
| GCC / Clang | C++17 support |
| CUDA Toolkit | 11.0 |
| OpenMP | 4.5 |

*(Developed and tested on Ubuntu 24.04 LTS, AMD Ryzen 7 5800H, and NVIDIA RTX 3050).*

---

## ⚙️ Build & Run

### Quick Start (Automated Benchmark)

For a fully automated pipeline that builds the project, runs the entire experiment matrix (with 5 repeats per condition), and generates performance plots:

```bash
bash run_experiments.sh
```

**What this script does automatically:**
1. Configures and builds `sort_bench` (defaults to CUDA Architecture `86` for RTX 30-series).
2. Executes the Serial, OpenMP, and CUDA benchmarks across varying sizes and distributions.
3. Consolidates outputs into `results/results.csv`.
4. Creates a Python virtual environment (`.venv`), installs `matplotlib`/`pandas`/`seaborn`.
5. Generates the final Speedup and Execution Time plots in `results/plots/`.

*(Note: To override the default CUDA architecture, run: `CUDA_ARCHITECTURES=75 bash run_experiments.sh`)*

---

### Manual Execution (CLI Usage)

If you wish to build manually and run specific benchmark configurations:

```bash
# 1. Configure and Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j$(nproc)

# 2. Run a specific test (e.g., CUDA Bitonic Sort, 67.1M elements, Reversed data)
./build/sort_bench \
    --size 67108864 \
    --distribution reversed \
    --impl cuda \
    --block-size 512 \
    --repeats 5 \
    --output results/cuda.csv
```

### CLI Flag Reference

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--size` | positive int | **required** | Number of elements to sort |
| `--distribution` | `uniform` `gaussian` `nearly_sorted` `reversed` | `uniform` | Input data distribution |
| `--seed` | int | `42` | RNG seed for reproducibility |
| `--impl` | `serial` `omp` `cuda` | `serial` | Target sorting algorithm |
| `--threads` | int | `1` | OpenMP thread count |
| `--block-size` | int | `256` | CUDA threads per block (Optimized: 512) |
| `--repeats` | int | `1` | Number of execution repetitions for averaging |
| `--output` | path | *(none)* | Filepath to append CSV metrics |

---

## 📊 Output Example

**Standard Output (Terminal):**
```text
[INFO]  Generating 67108864 elements (uniform, seed=42) ...
[INFO]  Running 'cuda' for 5 repeat(s) ...
[CUDA Metrics] Compute Percentage: 84.1% (Compute=885.4 ms, Transfer=167.3 ms)
[RESULT] average = 1052.7 ms
[VERIFY] Correctness check PASSED.
```