"""Plot estimator errors for Finch's shared matrix multiplication cases.

Run directly to save plots in graph/errorlogger_results.
"""

import argparse
from pathlib import Path

import scipy.sparse as sps

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
from finch.tests.stats_cases import DATASETS, make_models

def plot_dataset(dataset, output_dir):
    tensor = ft.asarray(sps.csr_array(DATASETS[dataset]()))
    factories = tuple(make_models().values())
    logger = ErrorLoggerStatsFactory(*factories)
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
    name = dataset.lower().replace(" ", "_").replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"errorlogger_{name}_matmul.png"
    ErrorReport(logger.log).plot(str(output))
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASETS,
        default=("Uniform Random", "ca-GrQc"),
        help="Finch cases to plot (default: 'Uniform Random' and ca-GrQc)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "graph" / "errorlogger_results",
        help="Directory for PNG plots (default: graph/errorlogger_results)",
    )
    args = parser.parse_args()
    if not julia_available():
        parser.error("Error logger plots require Finch's Julia backend (finch-tensor[julia]).")
    for dataset in args.datasets:
        plot_dataset(dataset, args.output_dir)


if __name__ == "__main__":
    main()
