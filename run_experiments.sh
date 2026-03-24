#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Reproducible benchmark driver
#
# Usage:
#   bash run_experiments.sh
#
# Adjust the variables in the CONFIG section before running.
# Results are appended to CSV files inside the RESULTS_DIR.
# =============================================================================

set -euo pipefail

# ── CONFIG ────────────────────────────────────────────────────────────────────
BINARY="./build/sort_bench"
RESULTS_DIR="./results"
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
        echo "Project 2 System Information"
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

# Ensure the binary exists
if [[ ! -x "$BINARY" ]]; then
    echo "[ERROR] Binary not found: $BINARY"
    echo "        Run: cmake --build build -j\$(nproc)"
    exit 1
fi

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
echo "============================================================"
echo " Done. Results written to: $RESULTS_DIR/"
echo " Combined results CSV : $RESULTS_CSV"
echo " System information   : $SYSTEM_INFO_FILE"
echo "============================================================"
