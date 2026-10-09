"""Time SDDMM implementations and write every measurement to CSV.

Run the complete sweep with::

    pixi run python -m finchtensorbench.benchmarks.sddmm

Compilation, scheduler setup, input conversion, and warmup calls are excluded
from the recorded execution times. Results are flushed after every measurement.
"""

from __future__ import annotations

import csv
import gc
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import sparse

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


def _make_inputs(k: int, nnz: int, seed: int = 42) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    sps.csr_matrix,
    sparse.COO,
    object,
    object,
    object,
]:
    rng = np.random.default_rng(seed + nnz + k)
    a = rng.random((2048, k))
    b = rng.random((k, 2048))
    flat_indices = rng.choice(2048 * 2048, size=nnz, replace=False)
    row_indices, column_indices = np.divmod(flat_indices, 2048)
    c_scipy = sps.csr_matrix(
        (rng.random(nnz) + 1.0, (row_indices, column_indices)),
        shape=(2048, 2048),
        dtype=np.float64,
    )
    c_scipy.indices = c_scipy.indices.astype(np.int64)
    c_scipy.indptr = c_scipy.indptr.astype(np.int64)
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


def main():
    output_path = Path("sddmm_results.csv")
    completed = Counter()
    if output_path.exists():
        with output_path.open(newline="") as existing:
            for row in csv.DictReader(existing):
                completed[
                    (int(row["nnz_c"]), int(row["k"]), row["implementation"])
                ] += 1

    with output_path.open("a", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "k",
                "nnz_c",
                "density_c",
                "implementation",
                "seconds",
                "finch_recommends",
            ),
        )
        if output.tell() == 0:
            writer.writeheader()
        for nnz in (2**i for i in range(1, 23)):
            for k in (2**j for j in range(1, 13)):
                if all(
                    completed[(nnz, k, name)] >= 5
                    for name in (
                        "finch_fused",
                        "finch_unfused",
                        "finch_autoscheduled",
                        "numpy",
                        "scipy",
                        "pydata_sparse",
                    )
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
                ) = _make_inputs(k, nnz)
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
                    remaining = 5 - completed[(nnz, k, name)]
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
                            f"nnz={nnz:>7} k={k:>4} {name:<19} "
                            f"seconds={seconds:.6g}"
                        )


if __name__ == "__main__":
    main()
