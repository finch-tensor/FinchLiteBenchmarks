"""Benchmark SDDMM with Finch matrix masks and seeded random dense factors.

Save CSV results and plots to graph/sddmm_results. Use --plot-only to replot
saved results or --max-size to change the default 2048-by-2048 mask crop.
"""

from __future__ import annotations

import argparse
import csv
import gc
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
import sparse
from matplotlib.colors import ListedColormap, TwoSlopeNorm

import finch as ft
from finch.autoschedule import LogicExecutor, LogicNormalizer
from finch.autoschedule.capture import LogicCapture
from finch.autoschedule.compiler import LogicCompiler
from finch.autoschedule.factorizer import DefaultLogicFactorizer
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.formatter import DefaultLogicFormatter, GalleyFormatter
from finch.autoschedule.loop_orderer import BFSLoopOrderer, DefaultLoopOrderer
from finch.autoschedule.tensor_stats import UniformStatsFactory
from finch.compile_jl.compiler import FinchJLCompiler
from finch.finch_logic import LogicSimplify, Query
from finch.tests.stats_cases import load_matrix_market


DATASETS = ("ct20stif", "roadNet-PA", "soc-sign-epinions")
K_VALUES = tuple(2**j for j in range(1, 13))
REPEAT = 5
IMPLEMENTATIONS = (
    "finch_fused",
    "finch_unfused",
    "finch_autoscheduled",
    "numpy",
    "scipy",
    "pydata_sparse",
)
SYNTHETIC_CSV_FIELDS = (
    "k", "nnz_c", "density_c", "implementation", "seconds", "finch_recommends"
)
CSV_FIELDS = ("dataset", "rows", "cols", *SYNTHETIC_CSV_FIELDS)


def load_mask(dataset, max_size=2048):
    matrix = load_matrix_market(f"{dataset}.mtx")
    matrix = sps.csr_matrix(matrix[:max_size, :max_size])
    matrix.indices = matrix.indices.astype(np.int64)
    matrix.indptr = matrix.indptr.astype(np.int64)
    return matrix


def _make_inputs(k: int, c_scipy: sps.csr_matrix, seed: int = 42) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    sps.csr_matrix,
    sparse.COO,
    object,
    object,
    object,
]:
    rng = np.random.default_rng(seed + k)
    rows, cols = c_scipy.shape
    a = rng.random((rows, k))
    b = rng.random((k, cols))
    return (
        a,
        b,
        c_scipy.toarray(),
        c_scipy,
        sparse.COO.from_scipy_sparse(c_scipy),
        ft.asarray(a),
        ft.asarray(b),
        ft.asarray(c_scipy),
    )


def _fused_scheduler():
    return LogicNormalizer(
        LogicExecutor(
            DefaultLogicFactorizer(
                LogicSimplify(
                    DefaultLoopOrderer(
                        DefaultLogicFormatter(LogicCompiler(FinchJLCompiler()))
                    )
                )
            ),
            stats_factory=UniformStatsFactory(),
            cache=True,
        )
    )


def _autoscheduled_scheduler() -> tuple[LogicNormalizer, LogicCapture]:
    capture = LogicCapture(GalleyFormatter(LogicCompiler(FinchJLCompiler())))
    scheduler = LogicNormalizer(
        LogicExecutor(
            GalleyLogicFactorizer(LogicSimplify(BFSLoopOrderer(capture))),
            stats_factory=UniformStatsFactory(),
            cache=True,
        )
    )
    return scheduler, capture


def finch_fused(a, b, c, ctx):
    return ft.compute(ft.defer(c) * (ft.defer(a) @ ft.defer(b)), ctx=ctx)


def finch_unfused(a, b, c):
    return ft.multiply(c, ft.matmul(a, b))


def finch_autoscheduled(a, b, c, ctx):
    return ft.compute(ft.defer(c) * (ft.defer(a) @ ft.defer(b)), ctx=ctx)


def completed_runs(output_path):
    completed = Counter()
    if not output_path.exists() or output_path.stat().st_size == 0:
        return completed
    with output_path.open(newline="") as existing:
        reader = csv.DictReader(existing)
        if reader.fieldnames == list(CSV_FIELDS):
            for row in reader:
                completed[
                    (
                        row["dataset"], int(row["rows"]), int(row["cols"]),
                        int(row["nnz_c"]), int(row["k"]), row["implementation"],
                    )
                ] += 1
            return completed
        if reader.fieldnames != list(SYNTHETIC_CSV_FIELDS):
            raise ValueError(f"Unrecognized SDDMM CSV columns in {output_path}")

    # Synthetic measurements cannot be reused for the real matrix cases.
    archive = output_path.with_name(f"{output_path.stem}_synthetic{output_path.suffix}")
    index = 1
    while archive.exists():
        archive = output_path.with_name(
            f"{output_path.stem}_synthetic_{index}{output_path.suffix}"
        )
        index += 1
    output_path.rename(archive)
    print(f"Archived synthetic measurements to {archive}")
    return completed


def run_benchmarks(output_path, max_size=2048):
    completed = completed_runs(output_path)

    with output_path.open("a", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        if output.tell() == 0:
            writer.writeheader()
        for dataset in DATASETS:
            mask = load_mask(dataset, max_size)
            rows, cols = mask.shape
            nnz = mask.nnz
            case = (dataset, rows, cols, nnz)
            for k in K_VALUES:
                if all(
                    completed[(*case, k, name)] >= REPEAT
                    for name in IMPLEMENTATIONS
                ):
                    continue
                (
                    a_numpy,
                    b_numpy,
                    c_numpy,
                    c_scipy,
                    c_pydata,
                    a_finch,
                    b_finch,
                    c_finch,
                ) = _make_inputs(k, mask)
                fused_ctx = _fused_scheduler()
                unfused_ctx = _fused_scheduler()
                autoscheduled_ctx, capture = _autoscheduled_scheduler()
                ft.set_default_scheduler(unfused_ctx)
                implementations = (
                    ("finch_fused", lambda: finch_fused(a_finch, b_finch, c_finch, fused_ctx), None),
                    ("finch_unfused", lambda: finch_unfused(a_finch, b_finch, c_finch), None),
                    ("finch_autoscheduled", lambda: finch_autoscheduled(a_finch, b_finch, c_finch, autoscheduled_ctx), capture),
                    ("numpy", lambda: c_numpy * (a_numpy @ b_numpy), None),
                    ("scipy", lambda: c_scipy.multiply(a_numpy @ b_numpy), None),
                    ("pydata_sparse", lambda: c_pydata * (a_numpy @ b_numpy), None),
                )
                for name, function, plan_capture in implementations:
                    remaining = REPEAT - completed[(*case, k, name)]
                    if remaining <= 0:
                        continue
                    function()
                    for _ in range(remaining):
                        gc.collect()
                        start = time.perf_counter()
                        function()
                        seconds = time.perf_counter() - start
                        recommendation = ""
                        if plan_capture is not None:
                            query_count = sum(
                                isinstance(body, Query)
                                for body in plan_capture.last_prgm.bodies
                            )
                            recommendation = (
                                "fused" if query_count == 1 else "unfused"
                            )
                        writer.writerow(
                            {
                                "dataset": dataset,
                                "rows": rows,
                                "cols": cols,
                                "k": k,
                                "nnz_c": nnz,
                                "density_c": nnz / c_numpy.size,
                                "implementation": name,
                                "seconds": seconds,
                                "finch_recommends": recommendation,
                            }
                        )
                        output.flush()
                        print(
                            f"{dataset} shape={rows}x{cols} nnz={nnz:>7} "
                            f"k={k:>4} {name:<19} "
                            f"seconds={seconds:.6g}"
                        )


def format_axes(axis, k_values, matrices):
    axis.set_xticks(range(len(k_values)), k_values, rotation=45)
    axis.set_yticks(
        range(len(matrices)),
        [
            f"{name}\n{rows}×{cols}, density={nnz / (rows * cols):.2%}"
            for name, rows, cols, nnz in matrices
        ],
    )
    axis.set_xlabel("K")


def plot_baseline_speedups(medians, k_values, matrices, output_dir):
    baselines = ("numpy", "scipy", "pydata_sparse")
    speedups = {}

    for finch in ("finch_autoscheduled", "best_fixed_finch"):
        for baseline in baselines:
            values = np.empty((len(matrices), len(k_values)))
            for row, matrix in enumerate(matrices):
                for column, k in enumerate(k_values):
                    if finch == "best_fixed_finch":
                        finch_time = min(
                            medians[(k, matrix, "finch_fused")],
                            medians[(k, matrix, "finch_unfused")],
                        )
                    else:
                        finch_time = medians[(k, matrix, finch)]
                    values[row, column] = np.log2(
                        medians[(k, matrix, baseline)] / finch_time
                    )
            speedups[(finch, baseline)] = values

    limit = max(max(np.max(np.abs(values)) for values in speedups.values()), 1e-12)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    for finch, label, filename in (
        ("finch_autoscheduled", "Autoscheduled Finch", "sddmm_autoscheduled_speedups.png"),
        ("best_fixed_finch", "Best Fixed Finch", "sddmm_best_fixed_speedups.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(20, 11), constrained_layout=True)
        for column, baseline in enumerate(baselines):
            axis = axes[column]
            values = speedups[(finch, baseline)]
            image = axis.imshow(
                values, origin="lower", aspect="auto", cmap="RdBu", norm=norm
            )
            axis.set_title(f"{label} vs {baseline.replace('_', ' ').title()}")
            format_axes(axis, k_values, matrices)
            if column == 0:
                axis.set_ylabel("Matrix C (Finch data)")

            for row in range(len(matrices)):
                for cell_column in range(len(k_values)):
                    speedup = 2 ** values[row, cell_column]
                    number = (
                        f"{speedup:.0f}×"
                        if speedup >= 10
                        else f"{speedup:.1f}×"
                        if speedup >= 0.1
                        else f"{speedup:.2f}×"
                    )
                    axis.text(
                        cell_column,
                        row,
                        number,
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white" if abs(values[row, cell_column]) > limit * 0.55 else "black",
                    )

        colorbar = figure.colorbar(image, ax=axes, shrink=0.75)
        colorbar.set_label("log₂(baseline runtime / Finch runtime)")
        colorbar.ax.axhline(0, color="black", linewidth=1)
        figure.suptitle(f"SDDMM speedup of {label} over Python baselines")
        figure.savefig(output_dir / filename, dpi=200)
        plt.close(figure)


def plot_finch_comparison(medians, recommendations, k_values, matrices, output_dir):
    shape = (len(matrices), len(k_values))
    fixed_comparison = np.empty(shape)
    auto_comparison = np.empty(shape)
    recommendation_values = np.empty(shape)
    fixed_labels = np.empty(shape, dtype=object)
    auto_labels = np.empty(shape, dtype=object)
    recommendation_labels = np.empty(shape, dtype=object)

    for row, matrix in enumerate(matrices):
        for column, k in enumerate(k_values):
            fused = medians[(k, matrix, "finch_fused")]
            unfused = medians[(k, matrix, "finch_unfused")]
            autoscheduled = medians[(k, matrix, "finch_autoscheduled")]
            fixed = min(fused, unfused)
            fixed_comparison[row, column] = np.log2(unfused / fused)
            auto_comparison[row, column] = np.log2(fixed / autoscheduled)

            if fused <= unfused:
                fixed_winner = "fused"
                fixed_labels[row, column] = f"F {unfused / fused:.1f}×"
            else:
                fixed_winner = "unfused"
                fixed_labels[row, column] = f"U {fused / unfused:.1f}×"

            if autoscheduled <= fixed:
                auto_labels[row, column] = f"A {fixed / autoscheduled:.1f}×"
            else:
                auto_labels[row, column] = f"F {autoscheduled / fixed:.1f}×"

            recommendation = recommendations[(k, matrix)]
            recommendation_values[row, column] = 1 if recommendation == "fused" else -1
            recommendation_labels[row, column] = (
                f"{'F' if recommendation == 'fused' else 'U'}"
                f"{'✓' if recommendation == fixed_winner else '✗'}"
            )

    limit = max(np.max(np.abs(fixed_comparison)), np.max(np.abs(auto_comparison)), 1e-12)
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, axes = plt.subplots(1, 3, figsize=(20, 11), constrained_layout=True)

    for axis, values, labels, title in (
        (axes[0], fixed_comparison, fixed_labels, "Fused vs Unfused"),
        (axes[1], auto_comparison, auto_labels, "Autoscheduled vs Best Fixed"),
    ):
        image = axis.imshow(
            values, origin="lower", aspect="auto", cmap="RdBu", norm=norm
        )
        axis.set_title(title)
        format_axes(axis, k_values, matrices)
        for row in range(len(matrices)):
            for column in range(len(k_values)):
                axis.text(
                    column,
                    row,
                    labels[row, column],
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if abs(values[row, column]) > limit * 0.55 else "black",
                )

    axes[0].set_ylabel("Matrix C (Finch data)")
    axes[2].imshow(
        recommendation_values,
        origin="lower",
        aspect="auto",
        cmap=ListedColormap(("#ff0000", "#0066ff")),
        vmin=-1,
        vmax=1,
    )
    axes[2].set_title("Autoscheduler Recommendation")
    format_axes(axes[2], k_values, matrices)
    for row in range(len(matrices)):
        for column in range(len(k_values)):
            axes[2].text(
                column,
                row,
                recommendation_labels[row, column],
                ha="center",
                va="center",
                fontsize=6,
            )

    colorbar = figure.colorbar(
        image, ax=axes, shrink=0.75, location="right"
    )
    colorbar.set_label("log₂(strategy 2 / strategy 1)")
    figure.suptitle("SDDMM Finch strategy comparison")
    figure.savefig(output_dir / "sddmm_finch_comparison.png", dpi=200)
    plt.close(figure)


def plot_results(output_dir):
    measurements = defaultdict(list)
    recommendations = {}
    with (output_dir / "sddmm_results.csv").open(newline="") as results:
        reader = csv.DictReader(results)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(
                "CSV has no Finch dataset identities. Run without --plot-only "
                "to benchmark the bundled matrices."
            )
        for row in reader:
            matrix = (
                row["dataset"], int(row["rows"]), int(row["cols"]), int(row["nnz_c"])
            )
            key = (int(row["k"]), matrix, row["implementation"])
            measurements[key].append(float(row["seconds"]))
            if row["finch_recommends"]:
                recommendations[(int(row["k"]), matrix)] = row[
                    "finch_recommends"
                ]

    medians = {key: np.median(values) for key, values in measurements.items()}
    k_values = sorted({key[0] for key in medians})
    matrices = [
        matrix
        for matrix in sorted({key[1] for key in medians})
        if all(
            (k, matrix, name) in medians and (k, matrix) in recommendations
            for k in k_values
            for name in IMPLEMENTATIONS
        )
    ]
    if not matrices:
        raise ValueError("No complete matrix results to plot; finish the benchmark run first.")
    plot_baseline_speedups(medians, k_values, matrices, output_dir)
    plot_finch_comparison(medians, recommendations, k_values, matrices, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "graph" / "sddmm_results",
        help="Directory for the CSV and PNG plots (default: graph/sddmm_results)",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Plot the existing CSV without running benchmarks",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=2048,
        help="Maximum rows and columns taken from each Finch matrix (default: 2048)",
    )
    args = parser.parse_args()
    if args.max_size < 1:
        parser.error("--max-size must be positive")
    output_path = args.output_dir / "sddmm_results.csv"
    if args.plot_only and not output_path.is_file():
        parser.error(f"CSV not found: {output_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not args.plot_only:
            run_benchmarks(output_path, args.max_size)
        plot_results(args.output_dir)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
