#!/usr/bin/env python3
"""Generate publication-ready plots for Project 2 benchmark results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


CANONICAL_COLUMNS = {
    "Implementation": ["Implementation", "impl"],
    "Size": ["Size", "size"],
    "Distribution": ["Distribution", "distribution"],
    "Threads": ["Threads", "threads"],
    "BlockSize": ["BlockSize", "block_size"],
    "AvgTime(ms)": ["AvgTime(ms)", "avg_ms"],
}

IMPLEMENTATION_NAMES = {
    "serial": "Serial",
    "omp": "OpenMP",
    "cuda": "CUDA",
    "Serial": "Serial",
    "OpenMP": "OpenMP",
    "CUDA": "CUDA",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Project 2 performance plots.")
    parser.add_argument(
        "--input",
        default="results/results.csv",
        help="Path to the combined results CSV (default: results/results.csv).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory to save plot PNG files (default: results/plots).",
    )
    parser.add_argument(
        "--distribution",
        default=None,
        help="Distribution to plot. Defaults to 'uniform' if present, otherwise the first available.",
    )
    parser.add_argument(
        "--large-size",
        type=int,
        default=None,
        help="Fixed large array size for Plot B and Plot C. Defaults to the largest available compatible size.",
    )
    return parser.parse_args()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map: dict[str, str] = {}
    missing: list[str] = []

    for canonical, aliases in CANONICAL_COLUMNS.items():
        match = next((alias for alias in aliases if alias in df.columns), None)
        if match is None:
            missing.append(canonical)
        else:
            rename_map[match] = canonical

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = df.rename(columns=rename_map)[list(CANONICAL_COLUMNS.keys())].copy()
    normalized["Implementation"] = (
        normalized["Implementation"]
        .astype(str)
        .str.strip()
        .map(lambda value: IMPLEMENTATION_NAMES.get(value, value))
    )
    normalized["Distribution"] = normalized["Distribution"].astype(str).str.strip()

    for column in ["Size", "Threads", "BlockSize", "AvgTime(ms)"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized = normalized.dropna(subset=["Implementation", "Size", "Distribution", "AvgTime(ms)"])
    normalized["Size"] = normalized["Size"].astype(int)
    normalized["Threads"] = normalized["Threads"].fillna(1).astype(int)
    normalized["BlockSize"] = normalized["BlockSize"].fillna(0).astype(int)
    normalized["AvgTime(ms)"] = normalized["AvgTime(ms)"].astype(float)
    return normalized


def select_distribution(df: pd.DataFrame, requested: str | None) -> str:
    available = sorted(df["Distribution"].unique())
    if not available:
        raise ValueError("No distributions found in the results data.")

    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested distribution '{requested}' is unavailable. Choices: {', '.join(available)}"
            )
        return requested

    return "uniform" if "uniform" in available else available[0]


def select_large_size(df: pd.DataFrame, distribution: str, implementations: list[str], requested: int | None) -> int:
    subset = df[df["Distribution"] == distribution]
    compatible_sizes = []

    for size in sorted(subset["Size"].unique()):
        size_slice = subset[subset["Size"] == size]
        if all((size_slice["Implementation"] == impl).any() for impl in implementations):
            compatible_sizes.append(size)

    if not compatible_sizes:
        raise ValueError(
            f"No sizes found for distribution '{distribution}' that include: {', '.join(implementations)}"
        )

    if requested is not None:
        if requested not in compatible_sizes:
            raise ValueError(
                f"Requested size {requested} not available for {', '.join(implementations)} under '{distribution}'. "
                f"Choices: {', '.join(str(size) for size in compatible_sizes)}"
            )
        return requested

    return max(compatible_sizes)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def create_plot_a(df: pd.DataFrame, distribution: str, output_dir: Path) -> None:
    subset = df[df["Distribution"] == distribution].copy()
    if subset.empty:
        raise ValueError(f"No data available for distribution '{distribution}'")

    best_config = (
        subset.groupby(["Implementation", "Size"], as_index=False)["AvgTime(ms)"]
        .min()
        .sort_values(["Implementation", "Size"])
    )

    order = ["Serial", "OpenMP", "CUDA"]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=best_config,
        x="Size",
        y="AvgTime(ms)",
        hue="Implementation",
        hue_order=[name for name in order if name in best_config["Implementation"].unique()],
        style="Implementation",
        markers=True,
        dashes=False,
        linewidth=2.5,
        ax=ax,
    )
    ax.set_title(f"Plot A: Execution Time vs. Array Size ({distribution}, best tuned configuration)")
    ax.set_xlabel("Array Size")
    ax.set_ylabel("Average Execution Time (ms)")
    ax.ticklabel_format(style="plain", axis="x")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Implementation")
    save_figure(fig, output_dir / "plot_a_execution_time_vs_array_size.png")


def create_plot_b(df: pd.DataFrame, distribution: str, large_size: int, output_dir: Path) -> None:
    subset = df[(df["Distribution"] == distribution) & (df["Size"] == large_size)].copy()
    serial_rows = subset[subset["Implementation"] == "Serial"]
    omp_rows = subset[subset["Implementation"] == "OpenMP"]

    if serial_rows.empty or omp_rows.empty:
        raise ValueError(
            f"Plot B requires Serial and OpenMP rows for size={large_size}, distribution='{distribution}'."
        )

    serial_time = serial_rows["AvgTime(ms)"].mean()
    omp_speedup = (
        omp_rows.groupby("Threads", as_index=False)["AvgTime(ms)"]
        .mean()
        .sort_values("Threads")
    )
    omp_speedup["Speedup"] = serial_time / omp_speedup["AvgTime(ms)"]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=omp_speedup,
        x="Threads",
        y="Speedup",
        marker="o",
        linewidth=2.5,
        color="#1f77b4",
        ax=ax,
    )
    ax.set_title(f"Plot B: OpenMP Speedup vs. Number of Threads ({distribution}, size={large_size})")
    ax.set_xlabel("OpenMP Threads")
    ax.set_ylabel("Speedup (Serial Time / OpenMP Time)")
    ax.set_xticks(sorted(omp_speedup["Threads"].unique()))
    ax.grid(True, linestyle="--", alpha=0.4)

    for _, row in omp_speedup.iterrows():
        ax.annotate(
            f"{row['Speedup']:.2f}x",
            (row["Threads"], row["Speedup"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
        )

    save_figure(fig, output_dir / "plot_b_openmp_speedup_vs_threads.png")


def create_plot_c(df: pd.DataFrame, distribution: str, large_size: int, output_dir: Path) -> None:
    cuda_rows = df[
        (df["Distribution"] == distribution)
        & (df["Size"] == large_size)
        & (df["Implementation"] == "CUDA")
    ].copy()

    if cuda_rows.empty:
        raise ValueError(f"Plot C requires CUDA rows for size={large_size}, distribution='{distribution}'.")

    cuda_rows = (
        cuda_rows.groupby("BlockSize", as_index=False)["AvgTime(ms)"]
        .mean()
        .sort_values("BlockSize")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=cuda_rows, x="BlockSize", y="AvgTime(ms)", palette="Blues_d", ax=ax)
    ax.set_title(f"Plot C: CUDA Execution Time vs. Block Size ({distribution}, size={large_size})")
    ax.set_xlabel("CUDA Block Size")
    ax.set_ylabel("Average Execution Time (ms)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    for patch, value in zip(ax.patches, cuda_rows["AvgTime(ms)"]):
        ax.annotate(
            f"{value:.2f}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=10,
        )

    save_figure(fig, output_dir / "plot_c_cuda_execution_time_vs_block_size.png")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {input_path}")

    configure_style()

    df = pd.read_csv(input_path)
    df = normalize_columns(df)

    distribution = select_distribution(df, args.distribution)
    shared_large_size = select_large_size(df, distribution, ["Serial", "OpenMP", "CUDA"], args.large_size)

    create_plot_a(df, distribution, output_dir)
    create_plot_b(df, distribution, shared_large_size, output_dir)
    create_plot_c(df, distribution, shared_large_size, output_dir)

    print(f"Plots saved to: {output_dir}")
    print(f"Distribution used: {distribution}")
    print(f"Plot B size: {shared_large_size}")
    print(f"Plot C size: {shared_large_size}")


if __name__ == "__main__":
    main()
