#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Reproducible benchmark driver
#
# Usage:
#   bash run_experiments.sh
#   CUDA_ARCHITECTURES=86 bash run_experiments.sh
#
# Adjust the variables in the CONFIG section before running.
# Results are appended to CSV files inside the RESULTS_DIR.
# =============================================================================

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────
BINARY="./build/sort_bench"
BUILD_DIR="./build"
RESULTS_DIR="./results"
PLOT_VENV_DIR="./.venv"
CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES:-}"
SYSTEM_INFO_FILE="$RESULTS_DIR/system_info.txt"
RESULTS_CSV="$RESULTS_DIR/results.csv"
REPEATS=5
SEED=42

SIZES=(1000000 4000000 16000000 64000000)
DISTRIBUTIONS=(uniform gaussian nearly_sorted reversed)
OMP_THREAD_COUNTS=(1 2 4 8 16)
CUDA_BLOCK_SIZES=(128 256 512)
# ─────────────────────────────────────────────────────────────────────────────

append_system_info() {
    {
        echo "============================================================"
        echo "System Information"
        echo "Timestamp (UTC): $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "Hostname       : $(hostname)"
        echo "Kernel         : $(uname -srmo)"
        echo ""

        echo "[OS]"
        if [[ -r /etc/os-release ]]; then
            grep -E '^(PRETTY_NAME|NAME|VERSION)=' /etc/os-release
        else
            echo "OS metadata unavailable"
        fi
        echo ""

        echo "[CPU]"
        if command -v lscpu >/dev/null 2>&1; then
            lscpu
        else
            echo "lscpu unavailable"
        fi
        echo ""

        echo "[Compiler]"
        if command -v g++ >/dev/null 2>&1; then
            g++ --version | head -n 1
        elif command -v c++ >/dev/null 2>&1; then
            c++ --version | head -n 1
        else
            echo "Host C++ compiler unavailable"
        fi

        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version | tail -n 1
        else
            echo "nvcc unavailable"
        fi
        echo ""

        echo "[GPU]"
        if command -v nvidia-smi >/dev/null 2>&1; then
            if ! nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; then
                echo "nvidia-smi is installed but GPU details are unavailable"
            fi
        else
            echo "nvidia-smi unavailable"
        fi
        echo ""
    } >> "$SYSTEM_INFO_FILE"
}

merge_results_csv() {
    printf "Implementation,Size,Distribution,Threads,BlockSize,AvgTime(ms)\n" > "$RESULTS_CSV"

    for csv in "$SERIAL_CSV" "$OMP_CSV" "$CUDA_CSV"; do
        [[ -f "$csv" ]] || continue

        tail -n +2 "$csv" | while IFS=, read -r impl size distribution seed threads block_size repeats avg_ms; do
            case "$impl" in
                serial) implementation="Serial" ;;
                omp) implementation="OpenMP" ;;
                cuda) implementation="CUDA" ;;
                *) implementation="$impl" ;;
            esac

            printf "%s,%s,%s,%s,%s,%s\n" \
                "$implementation" \
                "$size" \
                "$distribution" \
                "$threads" \
                "$block_size" \
                "$avg_ms" >> "$RESULTS_CSV"
        done
    done
}

resolve_binary_path() {
    local candidates=(
        "$BINARY"
        "$BUILD_DIR/Release/sort_bench"
        "$BUILD_DIR/Debug/sort_bench"
        "$BUILD_DIR/RelWithDebInfo/sort_bench"
        "$BUILD_DIR/MinSizeRel/sort_bench"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -x "$candidate" ]]; then
            BINARY="$candidate"
            return 0
        fi
    done

    if [[ -d "$BUILD_DIR" ]]; then
        local discovered=""
        discovered="$(find "$BUILD_DIR" -maxdepth 3 -type f -name sort_bench -perm -111 2>/dev/null | head -n 1 || true)"
        if [[ -n "$discovered" ]]; then
            BINARY="$discovered"
            return 0
        fi
    fi

    return 1
}

ensure_binary() {
    if resolve_binary_path; then
        echo "[INFO] Using binary: $BINARY"
        return 0
    fi

    echo "[INFO] Binary not found, attempting configure+build..."

    if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
        if [[ -n "$CUDA_ARCHITECTURES" ]]; then
            echo "[INFO] Configuring CMake with CUDA architectures: $CUDA_ARCHITECTURES"
            cmake -S . -B "$BUILD_DIR" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES"
        else
            cmake -S . -B "$BUILD_DIR"
        fi
    fi

    cmake --build "$BUILD_DIR" -j"$(nproc)"

    if ! resolve_binary_path; then
        echo "[ERROR] Binary not found after build. Expected target: sort_bench"
        echo "        Checked under: $BUILD_DIR"
        exit 1
    fi

    echo "[INFO] Using binary: $BINARY"
}

setup_plot_python_env() {
    local host_python=""

    if command -v python3 >/dev/null 2>&1; then
        host_python="python3"
    elif command -v python >/dev/null 2>&1; then
        host_python="python"
    else
        echo "[WARN] Python interpreter not found. Skipping plot generation."
        return 1
    fi

    if [[ ! -x "$PLOT_VENV_DIR/bin/python" ]]; then
        echo "[INFO] Creating Python virtual environment at: $PLOT_VENV_DIR"
        "$host_python" -m venv "$PLOT_VENV_DIR"
    fi

    PLOT_PYTHON="$PLOT_VENV_DIR/bin/python"
    if ! "$PLOT_PYTHON" -c "import pandas, matplotlib, seaborn" >/dev/null 2>&1; then
        echo "[INFO] Installing plotting dependencies (pandas, matplotlib, seaborn)..."
        "$PLOT_PYTHON" -m pip install --upgrade pip
        "$PLOT_PYTHON" -m pip install pandas matplotlib seaborn
    fi

    return 0
}

ensure_binary

mkdir -p "$RESULTS_DIR"

SERIAL_CSV="$RESULTS_DIR/serial.csv"
OMP_CSV="$RESULTS_DIR/omp.csv"
CUDA_CSV="$RESULTS_DIR/cuda.csv"

echo "============================================================"
echo " Project 2 — Experiment Suite"
echo " Date : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo " Host : $(hostname)"
echo "============================================================"

append_system_info

# ── 1. Serial baseline ────────────────────────────────────────────────────────
echo ""
echo "[PHASE 1] Serial baseline"
for dist in "${DISTRIBUTIONS[@]}"; do
    for size in "${SIZES[@]}"; do
        echo "  serial  dist=$dist  size=$size"
        "$BINARY" \
            --size "$size" \
            --distribution "$dist" \
            --seed "$SEED" \
            --impl serial \
            --repeats "$REPEATS" \
            --output "$SERIAL_CSV"
    done
done

# ── 2. OpenMP MergeSort ───────────────────────────────────────────────────────
echo ""
echo "[PHASE 2] OpenMP MergeSort"
for dist in "${DISTRIBUTIONS[@]}"; do
    for size in "${SIZES[@]}"; do
        for threads in "${OMP_THREAD_COUNTS[@]}"; do
            echo "  omp  dist=$dist  size=$size  threads=$threads"
            "$BINARY" \
                --size "$size" \
                --distribution "$dist" \
                --seed "$SEED" \
                --impl omp \
                --threads "$threads" \
                --repeats "$REPEATS" \
                --output "$OMP_CSV"
        done
    done
done

# ── 3. CUDA Bitonic Sort ──────────────────────────────────────────────────────
echo ""
echo "[PHASE 3] CUDA Bitonic Sort"
for dist in "${DISTRIBUTIONS[@]}"; do
    for size in "${SIZES[@]}"; do
        for block_size in "${CUDA_BLOCK_SIZES[@]}"; do
            echo "  cuda  dist=$dist  size=$size  block_size=$block_size"
            "$BINARY" \
                --size "$size" \
                --distribution "$dist" \
                --seed "$SEED" \
                --impl cuda \
                --block-size "$block_size" \
                --repeats "$REPEATS" \
                --output "$CUDA_CSV"
        done
    done
done

merge_results_csv

echo ""
echo "[POST] Generating plots"
if [[ -f "./plot_results.py" ]]; then
    PLOT_PYTHON=""
    if setup_plot_python_env; then
        if ! "$PLOT_PYTHON" ./plot_results.py; then
            echo "[WARN] Plot generation failed. You can retry manually: $PLOT_PYTHON plot_results.py"
        fi
    fi
else
    echo "[WARN] plot_results.py not found. Skipping plot generation."
fi

echo ""
echo "============================================================"
echo " Done. Results written to: $RESULTS_DIR/"
echo " Combined results CSV : $RESULTS_CSV"
echo " System information   : $SYSTEM_INFO_FILE"
echo "============================================================"
