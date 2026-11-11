"""Compare estimators using Finch's shared benchmark cases.

The three real matrices are loaded by finch.tests.stats_cases.load_matrix_market
from Finch's bundled test data. Structured and random cases use its shared
generators.
"""

from __future__ import annotations

import argparse
import json
import logging
from functools import partial
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from matplotlib import colormaps
from matplotlib.figure import Figure

import finch as ft
from finch.autoschedule.tensor_stats import (
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finch.autoschedule.tensor_stats.sampling_stats import (
    SamplingStats,
    SamplingStatsFactory,
)
from finch.tests.stats_cases import (
    BLOCK_COUNT,
    DENSE_MATRIX_SIZE,
    KERNELS,
    RANDOM_DENSITY,
    RANDOM_MATRIX_SIZE,
    SAMPLE_NNZ,
    SEED,
    SPARSE_MATRIX_SIZE,
    load_matrix_market,
    make_banded,
    make_kernel_estimator,
    make_kernel_stats,
    make_striped,
    make_triangular,
    make_uniform_random,
)

GEOMEAN_DATASETS = ("ct20stif", "roadNet-PA", "soc-sign-epinions")
SAMPLING = ("sh3", "schlosser")
DATASETS = [
    ("Banded", partial(make_banded, SPARSE_MATRIX_SIZE)),
    ("Triangular", partial(make_triangular, DENSE_MATRIX_SIZE)),
    ("Striped", partial(make_striped, DENSE_MATRIX_SIZE)),
    ("Uniform Random", make_uniform_random),
    ("ct20stif", partial(load_matrix_market, "ct20stif.mtx")),
    ("roadNet-PA", partial(load_matrix_market, "roadNet-PA.mtx")),
    ("soc-sign-epinions", partial(load_matrix_market, "soc-sign-epinions.mtx")),
]


def make_models():
    models = {
        "Dense": DenseStatsFactory(),
        "Uniform": UniformStatsFactory(),
        "DC": DCStatsFactory(),
        "VP": VPStatsFactory(),
        "LP": LPStatsFactory(),
        "Blocked-Uniform": BlockedUniformStatsFactory(block_count=BLOCK_COUNT),
    }
    for estimator in SAMPLING:
        sampling = SamplingStatsFactory(sample_nnz=SAMPLE_NNZ, estimator=estimator)
        sampling._rng = np.random.default_rng(SEED)
        models[f"Sampling_{estimator}"] = sampling
    return models


def actual_nonzeros(matrix, batch_rows=128):
    # Boolean products count reachability without large path counts. Row batches
    # bound intermediate storage even when powers of a sparse graph become dense.
    pattern = matrix.astype(bool)
    actual = dict.fromkeys(KERNELS, 0)
    actual["Hadamard"] = int(pattern.count_nonzero())
    for start in range(0, pattern.shape[0], batch_rows):
        rows = pattern[start : start + batch_rows]
        squared = rows @ pattern
        actual["SpGEMM"] += int(squared.count_nonzero())
        actual["SpGEMM2"] += int((squared @ pattern).count_nonzero())
        actual["Triangle Counting"] += int(squared.multiply(rows).count_nonzero())
    return actual


def accuracy_results():
    results = {kernel: {} for kernel in KERNELS}
    q_errors = {model: [] for model in make_models()}
    datasets = {}
    for dataset, make_matrix in DATASETS:
        a = sps.csr_array(make_matrix())
        datasets[dataset] = {
            "shape": list(a.shape),
            "nnz": int(a.nnz),
            "source": (
                f"finch.tests.data/{dataset}.mtx"
                if dataset in GEOMEAN_DATASETS
                else "finch.tests.stats_cases"
            ),
        }
        tensor = ft.asarray(a)
        logging.getLogger(__name__).info("Computing reference counts for %s", dataset)
        actual = actual_nonzeros(a)
        sampling_stats: dict[str, SamplingStats] = {}
        # Fresh factories isolate sampling masks between datasets.
        for model, factory in make_models().items():
            logging.getLogger(__name__).info("Estimating %s with %s", dataset, model)
            match factory:
                case SamplingStatsFactory():
                    # Sampling variants use identical budgets and seeds; only
                    # the estimator applied to the materialized sketch differs.
                    if not sampling_stats:
                        kernel_stats = make_kernel_stats(factory, tensor)
                        sampling_stats = {
                            kernel: kernel_stats(kernel) for kernel in KERNELS
                        }
                        for kernel, stats in sampling_stats.items():
                            _, nnz, _, _ = stats.scan(needs_freq=True)
                            logging.getLogger(__name__).info(
                                "%s / %s: %s retained nonzeros", dataset, kernel, nnz
                            )
                    estimates = {}
                    for kernel, cached_stats in sampling_stats.items():
                        stats = factory.copy(cached_stats)
                        stats.estimator = factory.estimator
                        estimates[kernel] = stats.estimate_non_fill_values()
                case _:
                    estimate_kernel = make_kernel_estimator(factory, tensor)
                    estimates = {kernel: estimate_kernel(kernel) for kernel in KERNELS}
            for kernel, estimate in estimates.items():
                estimate = float(estimate)
                # Clamp only for ratios, retaining the raw counts in the results.
                ratio = max(estimate, 1) / max(actual[kernel], 1)
                q_error = max(ratio, 1 / ratio)
                if dataset in GEOMEAN_DATASETS:
                    q_errors[model].append(q_error)
                results[kernel].setdefault(dataset, {})[model] = {
                    "actual_nnz": actual[kernel],
                    "estimated_nnz": round(estimate, 6),
                    "ratio": round(ratio, 6),
                    "q_error": round(q_error, 6),
                }
    return {
        "dense_matrix_size": DENSE_MATRIX_SIZE,
        "sparse_matrix_size": SPARSE_MATRIX_SIZE,
        "random_matrix_size": RANDOM_MATRIX_SIZE,
        "random_density": RANDOM_DENSITY,
        "datasets": datasets,
        "block_count": BLOCK_COUNT,
        "seed": SEED,
        "sample_nnz": SAMPLE_NNZ,
        "geomean_datasets": list(GEOMEAN_DATASETS),
        "results": results,
        "geomean_q_error": {
            model: round(float(np.exp(np.mean(np.log(errors)))), 6)
            for model, errors in q_errors.items()
        },
    }


def plot_accuracy(results):
    fig = Figure(figsize=(18, 16), layout="constrained")
    axes = fig.subplots(len(KERNELS), 1)
    models = list(make_models())
    datasets = [name for name, _ in DATASETS]
    x = np.arange(len(datasets))
    width = 0.8 / len(models)
    for ax, kernel in zip(axes, KERNELS, strict=True):
        for index, model in enumerate(models):
            values = np.log2(
                [results[kernel][dataset][model]["ratio"] for dataset in datasets]
            )
            positions = x + (index - (len(models) - 1) / 2) * width
            color = colormaps["tab20"](index / 19)
            ax.bar(
                positions,
                np.clip(values, -10, 10),
                width=width,
                color=color,
                label=model,
                zorder=3,
            )
            for position, value in zip(positions, values, strict=True):
                if abs(value) < 0.1:
                    ax.plot(position, 0, marker="x", color=color, markersize=5)
                elif abs(value) > 10:
                    ax.text(
                        position,
                        np.sign(value) * 9.5,
                        f"{value:+.1f}",
                        ha="center",
                        va="center",
                        fontsize=6,
                    )
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=kernel, ylabel="log2(estimated / actual nnz)", ylim=(-10, 10))
        ax.set_xticks(x, datasets, rotation=20, ha="right")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.suptitle(f"Sparsity estimator accuracy — sampling budget {SAMPLE_NNZ:,} nnz")
    return fig


def plot_geomean(geomeans):
    fig = Figure(figsize=(14, 6), layout="constrained")
    ax = fig.subplots()
    models = list(make_models())
    x = np.arange(len(models))
    ax.bar(
        x,
        [geomeans[model] for model in models],
        color=[colormaps["tab20"](i / 19) for i in x],
    )
    ax.axhline(1, color="black", linestyle="--", label="Perfect estimate")
    ax.set_yscale("log", base=2)
    ax.set_xticks(x, models, rotation=35, ha="right", fontsize=9)
    ax.set(
        ylabel="Geometric mean q-error",
        title="Estimator accuracy — real matrices only\n"
        + ", ".join(GEOMEAN_DATASETS)
        + f" ({len(KERNELS)} kernels each)",
    )
    ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "graph" / "stats_comparison",
        help="Directory for JSON results and PNG plots (default: graph/stats_comparison)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = accuracy_results()
    output = args.output_dir / "accuracy.json"
    output.write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    logging.getLogger(__name__).info("Saved %s", output)
    for view, fig in (
        ("accuracy", plot_accuracy(results["results"])),
        ("geomean", plot_geomean(results["geomean_q_error"])),
    ):
        output = args.output_dir / f"{view}.png"
        fig.savefig(output, format="png", dpi=100)
        logging.getLogger(__name__).info("Saved %s", output)


if __name__ == "__main__":
    main()
