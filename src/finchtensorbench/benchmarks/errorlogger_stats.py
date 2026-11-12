"""Plot estimator errors for Finch's shared matrix multiplication cases.

Run from the repository root:

    pixi run python src/finchtensorbench/benchmarks/errorlogger_stats.py

This saves one combined graph of PLOT_DATASETS (luxembourg_osm, ca-GrQc, bcsstk39),
side by side, to graph/errorlogger_results/.
"""

import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from scipy.io import mmread

import finch as ft
from finch.autoschedule import LogicExecutor, LogicNormalizer
from finch.autoschedule.compiler import CompilerFormLowerer, LogicCompiler
from finch.autoschedule.factorizer.galley_factorizer.galley_optimize import (
    GalleyLogicFactorizer,
)
from finch.autoschedule.formatter import GalleyFormatter
from finch.autoschedule.loop_orderer import DefaultLoopOrderer
from finch.autoschedule.tensor_stats.errorlogger_stats import (
    ErrorLoggerStatsFactory,
    ErrorReport,
)
from finch.compile_jl.compiler import FinchJLCompiler
from finch.compile_jl.julia import julia_available
from finch.finch_logic import LogicSimplify
from finch.autoschedule.tensor_stats import (
    BlockedStatsFactory,
    BlockedUniformStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    SamplingStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
)
from finch.tests.stats_cases import BLOCK_COUNT, DATASETS, SAMPLE_NNZ, SEED


DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def load_local_matrix(name, shape, nnz):
    """Load a SuiteSparse matrix stored in the repository's data/ directory."""
    matrix = sps.csr_array(mmread(DATA_DIR / f"{name}.mtx"), dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data.fill(1.0)
    if matrix.shape != shape or matrix.nnz != nnz:
        raise ValueError(f"Unexpected {name} shape or nonzero count")
    return matrix


LOCAL_DATASETS = {
    "luxembourg_osm": ((114599, 114599), 239332),
}
DATASETS = {
    **DATASETS,
    **{
        name: partial(load_local_matrix, name, shape, nnz)
        for name, (shape, nnz) in LOCAL_DATASETS.items()
    },
}

# Datasets drawn side by side in the combined graph, left to right.
PLOT_DATASETS = ["luxembourg_osm", "ca-GrQc", "bcsstk39"]


def make_sampling(estimator):
    sampling = SamplingStatsFactory(sample_nnz=SAMPLE_NNZ, estimator=estimator)
    sampling._rng = np.random.default_rng(SEED)
    return sampling


def make_statistics():
    return [
        DenseStatsFactory(),
        UniformStatsFactory(),
        DCStatsFactory(),
        VPStatsFactory(),
        LPStatsFactory(),
        BlockedUniformStatsFactory(block_count=BLOCK_COUNT),
        make_sampling("sh3"),
        make_sampling("schlosser"),
    ]


def run_dataset(dataset):
    """Run A @ A for one dataset and return the estimator error log."""
    tensor = ft.asarray(sps.csr_array(DATASETS[dataset]()))
    logger = ErrorLoggerStatsFactory(*make_statistics())
    jl_load = GalleyLogicFactorizer(
        LogicSimplify(
            DefaultLoopOrderer(
                GalleyFormatter(CompilerFormLowerer(LogicCompiler(FinchJLCompiler())))
            )
        ),
        optimizer="bfs",
    )
    schedule = LogicNormalizer(LogicExecutor(jl_load, stats_factory=logger))
    with ft.with_default_scheduler(schedule):
        ft.compute(ft.defer(tensor) @ ft.defer(tensor))
    return logger.log


def dataset_slug(dataset):
    return dataset.lower().replace(" ", "_").replace("-", "_")


def plot_dataset(dataset, log, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"errorlogger_{dataset_slug(dataset)}_matmul.png"
    # ErrorReport.plot saves and closes its figure without exposing a title option.
    savefig = Figure.savefig

    def save_with_dataset_title(figure, *args, **kwargs):
        figure.axes[0].set_title(f"Estimate / Exact Distributions ({dataset})")
        return savefig(figure, *args, **kwargs)

    with patch.object(Figure, "savefig", save_with_dataset_title):
        ErrorReport(log).plot(str(output))
    print(f"Saved {output}")


def plot_combined(logs, output_dir):
    """Draw every dataset as one panel of a single figure with shared axes."""
    figure, axes = plt.subplots(
        1, len(logs), figsize=(7 * len(logs) + 2, 6), sharey=True, squeeze=False
    )
    axes = axes[0]
    limits = []
    for axis, (dataset, log) in zip(axes, logs.items()):
        # ErrorReport.plot draws into plt.subplots() and then saves and closes;
        # hand it this panel instead and keep the figure open.
        with (
            patch.object(plt, "subplots", return_value=(figure, axis)),
            patch.object(Figure, "savefig"),
            patch.object(plt, "close"),
        ):
            ErrorReport(log).plot("unused")
        limits.append(axis.get_xlim())
        axis.set_title(dataset)
        axis.set_xlabel("")
        axis.set_ylabel("")
    left = min(low for low, _ in limits)
    right = max(high for _, high in limits)
    for axis in axes:
        axis.set_xlim(left, right)
    for axis in axes[1:]:
        axis.tick_params(labelleft=False)
    figure.suptitle("Estimate / Exact Distributions (A @ A)")
    figure.supxlabel(
        "Estimate / exact (log scale; 1 = exact)    "
        "Blue < 1: underestimation; orange > 1: overestimation"
    )
    axes[0].set_ylabel("Estimator\n(n = number of prediction requests)")
    # Leave room for the overflow labels drawn past each panel's right edge.
    figure.subplots_adjust(wspace=0.35)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "_".join(dataset_slug(dataset) for dataset in logs)
    output = output_dir / f"errorlogger_{name}_matmul.png"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=PLOT_DATASETS,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "graph" / "errorlogger_results",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="save one plot per dataset instead of one combined graph",
    )
    args = parser.parse_args()
    if args.separate:
        for dataset in args.datasets:
            plot_dataset(dataset, run_dataset(dataset), args.output_dir)
    else:
        logs = {dataset: run_dataset(dataset) for dataset in args.datasets}
        plot_combined(logs, args.output_dir)


if __name__ == "__main__":
    main()
