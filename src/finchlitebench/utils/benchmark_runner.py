import argparse
import time
from collections import defaultdict

import os
import numpy as np

import finchlite as fl
from finchlite.autoschedule import (
    DefaultLogicFormatter,
    LogicExecutor,
    LogicNormalizer,
    LogicStandardizer,
)
from finchlite.autoschedule.compiler import LogicCompiler
from finchlite.autoschedule.galley_optimize import GalleyLogicalOptimizer
from finchlite.autoschedule.tensor_stats import (
    BlockedStatsFactory,
    DatabaseStatsFactory,
    DCStatsFactory,
    DenseStatsFactory,
    UniformStatsFactory,
)
from finchlite.finch_notation.interpreter import NotationInterpreter

from ..benchmarks.cg import (
    benchmark_cg,
    dg_cg_sparse_1,
    dg_cg_sparse_2,
    dg_cg_sparse_3,
    dg_cg_sparse_4,
    dg_cg_sparse_5,
    dg_cg_sparse_6,
    dg_cg_sparse_7,
    dg_cg_sparse_8,
)
from ..benchmarks.matmul import (
    benchmark_matmul,
    benchmark_matmul_stats,
    dg_matmul_dense_large,
    dg_matmul_dense_small,
    dg_matmul_sparse_large,
    dg_matmul_sparse_small,
)


def galley_scheduler(stats_factory):
    return LogicNormalizer(
        LogicExecutor(
            GalleyLogicalOptimizer(
                LogicStandardizer(
                    DefaultLogicFormatter(LogicCompiler(NotationInterpreter()))
                )
            ),
            stats_factory=stats_factory,
        )
    )


def blocked_factory(inner_factory):
    factory = BlockedStatsFactory({}, inner_factory)
    factory.blocks_per_dim = defaultdict(lambda: 8)
    return factory

BENCHMARK_DICT = {
    "matmul": benchmark_matmul,
    "cg": benchmark_cg,
}

STATS_FACTORY_DICT = {
    "DenseStats": DenseStatsFactory(),
    "UniformStats": UniformStatsFactory(),
    "DatabaseStats": DatabaseStatsFactory(),
    "DCStats": DCStatsFactory(),
    "BlockedDenseStats": blocked_factory(DenseStatsFactory()),
    "BlockedUniformStats": blocked_factory(UniformStatsFactory()),
    "BlockedDatabaseStats": blocked_factory(DatabaseStatsFactory()),
    "BlockedDCStats": blocked_factory(DCStatsFactory()),
}

STATS_BENCHMARK_DICT = {
    "matmul": benchmark_matmul_stats,
}

REFERENCE_DICT = {
    "matmul": lambda A, B: A @ B,
}

DATA_GENERATOR_DICT = {
    "matmul": {
        "matmul_dense_small": dg_matmul_dense_small,
        "matmul_dense_large": dg_matmul_dense_large,
        "matmul_sparse_small": dg_matmul_sparse_small,
        "matmul_sparse_large": dg_matmul_sparse_large,
    },
    "cg": {
        "cg_sparse_1": dg_cg_sparse_1,
        "cg_sparse_2": dg_cg_sparse_2,
        "cg_sparse_3": dg_cg_sparse_3,
        "cg_sparse_4": dg_cg_sparse_4,
        "cg_sparse_5": dg_cg_sparse_5,
        "cg_sparse_6": dg_cg_sparse_6,
        "cg_sparse_7": dg_cg_sparse_7,
        "cg_sparse_8": dg_cg_sparse_8,
    },
}


def run_stats_benchmark(stats_factory, stats_benchmark_fn, benchmark_data_generator, iters, reference_fn=None):
    execution_times = []
    relative_errors = []
    for _ in range(iters):
        data = benchmark_data_generator()
        start = time.perf_counter()
        estimated_nnz = stats_benchmark_fn(stats_factory, *data)
        end = time.perf_counter()
        execution_times.append(end - start)
        if reference_fn is not None:
            actual_nnz = np.count_nonzero(reference_fn(*data))
            relative_errors.append((estimated_nnz - actual_nnz) / actual_nnz * 100)
    print(f"Stats estimation took an average of {sum(execution_times) / len(execution_times)} seconds")
    return execution_times, relative_errors

def save_benchmark_results(results_folder, execution_times, relative_errors, scheduler, benchmark, data_generator):
    folder_path = os.path.join(results_folder, data_generator)
    os.makedirs(folder_path, exist_ok=True)

    filename = os.path.join(folder_path, f"{scheduler}.csv")

    with open(filename, "w") as f:
        f.write("Scheduler,Benchmark,Data Generator,Iteration,ExecutionTime,RelativeError\n")
        for i, (t, e) in enumerate(zip(execution_times, relative_errors), start=1):
            f.write(f"{scheduler},{benchmark},{data_generator},{i},{t},{e}\n")

def main(
    benchmarks=None,
    benchmark_names=None,
    data_generators=None,
    data_generator_names=None,
    iters=None,
    results_folder=None,
    args=None,
):
    collected_benchmarks = BENCHMARK_DICT.copy()
    if benchmarks is not None:
        collected_benchmarks.update(benchmarks)

    collected_data_generators = {k: v.copy() for k, v in DATA_GENERATOR_DICT.items()}
    if data_generators is not None:
        for benchmark_name, generators in data_generators.items():
            collected_data_generators[benchmark_name].update(generators)

    parser = argparse.ArgumentParser(description="Run finch-lite benchmark")
    parser.add_argument("--benchmark", default=["all"], nargs="*")
    parser.add_argument("--data-generator", default=["all"], nargs="*")
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--results-folder", default="results")
    parser.add_argument("--stats", action="store_true", help="Run stats estimation benchmarks")
    args = parser.parse_args(args)

    if benchmark_names is None:
        benchmark_names = list(collected_benchmarks.keys()) if args.benchmark == ["all"] else args.benchmark
    if data_generator_names is None:
        if args.data_generator == ["all"]:
            data_generator_names = [
                name for generators in collected_data_generators.values() for name in generators
            ]
        else:
            data_generator_names = args.data_generator
    if results_folder is None:
        results_folder = args.results_folder
    if iters is None:
        iters = args.iterations

    data_generator_names = set(data_generator_names)

    if args.stats:
        for factory_name, stats_factory in STATS_FACTORY_DICT.items():
            for benchmark_name in benchmark_names:
                if benchmark_name not in STATS_BENCHMARK_DICT:
                    continue
                stats_benchmark = STATS_BENCHMARK_DICT[benchmark_name]
                for dg_name, dg in collected_data_generators[benchmark_name].items():
                    if dg_name not in data_generator_names:
                        continue
                    reference_fn = REFERENCE_DICT.get(benchmark_name)
                    print(f"Stats {benchmark_name} | {factory_name} | {dg_name}")
                    execution_times, relative_errors = run_stats_benchmark(stats_factory, stats_benchmark, dg, iters, reference_fn)
                    save_benchmark_results(results_folder, execution_times, relative_errors, factory_name, benchmark_name, dg_name)

if __name__ == "__main__":
    main()
