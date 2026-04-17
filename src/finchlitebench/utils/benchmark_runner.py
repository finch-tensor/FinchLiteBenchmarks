import argparse
import time
from collections import defaultdict

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

from ..benchmarks.matmul import (
    benchmark_matmul,
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


SCHEDULER_DICT = {
    "default": fl.get_default_scheduler(),
    "DenseStats": galley_scheduler(DenseStatsFactory()),
    "UniformStats": galley_scheduler(UniformStatsFactory()),
    "DatabaseStats": galley_scheduler(DatabaseStatsFactory()),
    "DCStats": galley_scheduler(DCStatsFactory()),
    "BlockedDenseStats": galley_scheduler(blocked_factory(DenseStatsFactory())),
    "BlockedUniformStats": galley_scheduler(blocked_factory(UniformStatsFactory())),
    "BlockedDatabaseStats": galley_scheduler(blocked_factory(DatabaseStatsFactory())),
    "BlockedDCStats": galley_scheduler(blocked_factory(DCStatsFactory())),
}

BENCHMARK_DICT = {
    "matmul": benchmark_matmul,
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
}


def run_benchmark(ctx, benchmark_function, benchmark_data_generator, iters, reference_fn=None):
    execution_times = []
    relative_errors = []
    for _ in range(iters):
        data = benchmark_data_generator()
        start = time.perf_counter()
        result = benchmark_function(ctx, *data)
        end = time.perf_counter()
        execution_times.append(end - start)
        if reference_fn is not None:
            ref = reference_fn(*data)
            actual_nnz = np.count_nonzero(ref)
            computed_nnz = np.count_nonzero(np.asarray(result))
            relative_errors.append((computed_nnz - actual_nnz) / actual_nnz * 100)
    print(f"Benchmark took an average of {sum(execution_times) / len(execution_times)} seconds")
    return execution_times, relative_errors


def save_benchmark_results(results_folder, execution_times, relative_errors, scheduler, benchmark, data_generator):
    filename = f"{results_folder}/{scheduler}_{benchmark}_{data_generator}.csv"
    with open(filename, "w") as f:
        f.write("Scheduler,Benchmark,Data Generator,Iteration,ExecutionTime,RelativeError\n")
        for i, (t, e) in enumerate(zip(execution_times, relative_errors), start=1):
            f.write(f"{scheduler},{benchmark},{data_generator},{i},{t},{e}\n")


def main(
    schedulers=None,
    scheduler_names=None,
    benchmarks=None,
    benchmark_names=None,
    data_generators=None,
    data_generator_names=None,
    iters=None,
    results_folder=None,
    args=None,
):
    collected_schedulers = SCHEDULER_DICT.copy()
    if schedulers is not None:
        collected_schedulers.update(schedulers)

    collected_benchmarks = BENCHMARK_DICT.copy()
    if benchmarks is not None:
        collected_benchmarks.update(benchmarks)

    collected_data_generators = {k: v.copy() for k, v in DATA_GENERATOR_DICT.items()}
    if data_generators is not None:
        for benchmark_name, generators in data_generators.items():
            collected_data_generators[benchmark_name].update(generators)

    parser = argparse.ArgumentParser(description="Run finch-lite benchmark")
    parser.add_argument("--scheduler", default=["all"], nargs="*")
    parser.add_argument("--benchmark", default=["all"], nargs="*")
    parser.add_argument("--data-generator", default=["all"], nargs="*")
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--results-folder", default="results")
    args = parser.parse_args(args)

    if scheduler_names is None:
        scheduler_names = list(SCHEDULER_DICT.keys()) if args.scheduler == ["all"] else args.scheduler
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
    for scheduler_name in scheduler_names:
        ctx = collected_schedulers[scheduler_name]
        for benchmark_name in benchmark_names:
            benchmark = collected_benchmarks[benchmark_name]
            for dg_name, dg in collected_data_generators[benchmark_name].items():
                if dg_name not in data_generator_names:
                    continue
                reference_fn = REFERENCE_DICT.get(benchmark_name)
                print(f"Running {benchmark_name} | {scheduler_name} | {dg_name}")
                execution_times, relative_errors = run_benchmark(ctx, benchmark, dg, iters, reference_fn)
                save_benchmark_results(
                    results_folder, execution_times, relative_errors, scheduler_name, benchmark_name, dg_name
                )


if __name__ == "__main__":
    main()
