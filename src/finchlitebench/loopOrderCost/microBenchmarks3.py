import argparse
import csv
import gc
import itertools
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sps

from finchlitebench.loopOrderCost.microBenchmark import RECORDER as SHARED_RECORDER
from finchlitebench.loopOrderCost.microBenchmark import FixedLoopOrderer, Recorder, scheduler

import finch as fl
from finch.compile_jl.julia import get_jl

SIZES = (128, 500, 1000)
FIXED_DENSITY = 1 / 256  # A..C use the same density
F_DENSITIES = (0.01, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0)
LOOP_ORDERS = tuple("".join(order) for order in itertools.permutations("ijklm"))
ORDER_SEED = 42
WARMUP = 1
RUNS = 20
OUT = Path("microBenchmark3_confirm_larger.csv")

FIELDS = (
    "size",
    "fixed_density",
    "f_density",
    "nnz_F",
    "loop_order",
    "executed_order",
    "est_cost",
    "seconds",
    "plan_seconds",
    "median_seconds",
    "filtered_mean_seconds",
    "filtered_run_count",
    *(f"run_{run}_seconds" for run in range(1, RUNS + 1)),
)


class CachedRecorder(Recorder):
    """Retain the bound kernel call so timed runs bypass the planner."""

    def __call__(self, *args):
        mod, *rest = super().__call__(*args)

        def main(*kernel_args):
            self.execute = lambda: mod.main(*kernel_args)
            return self.execute()

        return SimpleNamespace(main=main), *rest


RECORDER = CachedRecorder(SHARED_RECORDER.backend)


def chain(ctx, *mats):
    return fl.compute(fl.einsum("ij,jk,kl,lm->im", *mats), ctx=ctx)


def main():
    jl = get_jl()
    cases = []
    for n in SIZES:
        rng = np.random.default_rng([n, 0])
        fixed = [
            sps.random(
                n,
                n,
                density=FIXED_DENSITY,
                format="csr",
                random_state=rng,
                data_rvs=np.ones,
                dtype=np.float64,
            )
            for _ in range(3)
        ]
        for density_id, density in enumerate(F_DENSITIES):
            print(f"Preparing {n}x{n} inputs: F density {density:.6g}", flush=True)
            rng = np.random.default_rng([n, density_id, 1])
            mats = [
                *fixed,
                sps.random(
                    n,
                    n,
                    density=density,
                    format="csr",
                    random_state=rng,
                    data_rvs=np.ones,
                    dtype=np.float64,
                ),
            ]
            # Match Julia's integer writes to the CSR index-buffer type.
            for m in mats:
                m.indices = m.indices.astype(np.int64, copy=False)
                m.indptr = m.indptr.astype(np.int64, copy=False)
            inputs = [fl.defer(fl.asarray(m)) for m in mats]
            cases.append((n, density, mats[-1].nnz, inputs))

    total = len(cases) * len(LOOP_ORDERS)
    orders = list(LOOP_ORDERS)
    np.random.default_rng(ORDER_SEED).shuffle(orders)
    print(
        f"Fully fused ABCF: {len(cases)} density/size cases x "
        f"{len(orders)} permutations = {total} schedules; "
        f"{WARMUP} warmup; mean and median of {RUNS} cached executions; "
        f"shuffle seed={ORDER_SEED}",
        flush=True,
    )
    with OUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()

        for order_id, order in enumerate(orders):
            # Reuse this order across densities without keeping 120 kernels.
            RECORDER.kernels.clear()
            ctx = scheduler(FixedLoopOrderer(RECORDER, order))
            for case_id, (n, density, nnz_f, inputs) in enumerate(cases, 1):
                progress = order_id * len(cases) + case_id
                print(
                    f"[{progress}/{total}] {n}x{n} F density={density:.6g} "
                    f"nnz={nnz_f} order={order}",
                    flush=True,
                )
                RECORDER.reset()
                print("    planning + initial execution", flush=True)
                start = time.perf_counter()
                chain(ctx, *inputs)
                # One-time setup outside the initial kernel execution.
                plan_seconds = time.perf_counter() - start - RECORDER.seconds
                print("    Python + Julia garbage collection", flush=True)
                gc.collect()
                jl.seval("GC.gc()")
                for _ in range(WARMUP):
                    print("    warmup", flush=True)
                    RECORDER.execute()
                kernel = []
                for run in range(1, RUNS + 1):
                    print(f"    timed run {run}/{RUNS}", flush=True)
                    RECORDER.seconds = 0.0
                    RECORDER.execute()
                    kernel.append(RECORDER.seconds)
                    print(f"    {kernel[-1] * 1e3:.3f} ms", flush=True)
                if RECORDER.orders != [order] or not RECORDER.fused:
                    raise RuntimeError(
                        f"Expected fully fused {order}, got {RECORDER.orders}"
                    )
                q1, q3 = np.percentile(kernel, [25, 75])
                cutoff = max(3 * np.median(kernel), q3 + 3 * (q3 - q1))
                filtered = np.asarray(kernel)[np.asarray(kernel) <= cutoff]
                row = {
                    "size": n,
                    "fixed_density": FIXED_DENSITY,
                    "f_density": density,
                    "nnz_F": nnz_f,
                    "loop_order": order,
                    "executed_order": RECORDER.orders[0],
                    "est_cost": RECORDER.cost,
                    "seconds": float(np.mean(kernel)),
                    "plan_seconds": plan_seconds,
                    "median_seconds": float(np.median(kernel)),
                    "filtered_mean_seconds": float(np.mean(filtered)),
                    "filtered_run_count": len(filtered),
                    **{
                        f"run_{run}_seconds": seconds
                        for run, seconds in enumerate(kernel, 1)
                    },
                }
                writer.writerow(row)
                file.flush()
                print(
                    f"    mean {row['seconds'] * 1e3:.3f} ms; "
                    f"median {row['median_seconds'] * 1e3:.3f} ms; "
                    f"filtered mean {row['filtered_mean_seconds'] * 1e3:.3f} ms "
                    f"({row['filtered_run_count']}/{RUNS} runs); "
                    f"cost {row['est_cost']:.6g}",
                    flush=True,
                )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark one matrix size")
    parser.add_argument("--size", type=int, choices=SIZES, required=True)
    args = parser.parse_args()
    SIZES = (args.size,)
    OUT = Path(f"microBenchmark3_confirm_{args.size}_20runs.csv")
    main()
