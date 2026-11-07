import csv
import itertools
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from finchlitebench.loopOrderCost.microBenchmark import RECORDER, FixedLoopOrderer, scheduler

import finch as fl

SIZES = (28,)
FIXED_DENSITY = 1 / 256  # A..C use the same density
F_DENSITIES = (0.01, 0.20, 1.0)
LOOP_ORDERS = tuple("".join(order) for order in itertools.permutations("ijklm"))
WARMUP = 1
RUNS = 2
OUT = Path("chain_microbenchmark.csv")

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
)


def chain(ctx, *mats):
    return fl.compute(fl.einsum("ij,jk,kl,lm->im", *mats), ctx=ctx)


def main():
    cases = []
    for n in SIZES:
        for density_id, density in enumerate(F_DENSITIES):
            print(f"Preparing {n}x{n} inputs: F density {density:.6g}", flush=True)
            rng = np.random.default_rng([n, density_id])
            mats = [
                sps.random(
                    n,
                    n,
                    density=d,
                    format="csr",
                    random_state=rng,
                    data_rvs=np.ones,
                    dtype=np.float64,
                )
                for d in [FIXED_DENSITY] * 3 + [density]
            ]
            # Match Julia's integer writes to the CSR index-buffer type.
            for m in mats:
                m.indices = m.indices.astype(np.int64, copy=False)
                m.indptr = m.indptr.astype(np.int64, copy=False)
            inputs = [fl.defer(fl.asarray(m)) for m in mats]
            cases.append((n, density, mats[-1].nnz, inputs))

    total = len(cases) * len(LOOP_ORDERS)
    print(
        f"Fully fused ABCF: {len(cases)} density/size cases x "
        f"{len(LOOP_ORDERS)} permutations = {total} schedules; mean of {RUNS} runs",
        flush=True,
    )
    with OUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        file.flush()

        for order_id, order in enumerate(LOOP_ORDERS):
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
                for _ in range(WARMUP):
                    print("    warmup", flush=True)
                    chain(ctx, *inputs)
                kernel, wall = [], []
                for run in range(1, RUNS + 1):
                    print(f"    timed run {run}/{RUNS}", flush=True)
                    RECORDER.reset()
                    start = time.perf_counter()
                    chain(ctx, *inputs)
                    wall.append(time.perf_counter() - start)
                    kernel.append(RECORDER.seconds)
                if RECORDER.orders != [order] or not RECORDER.fused:
                    raise RuntimeError(
                        f"Expected fully fused {order}, got {RECORDER.orders}"
                    )
                row = {
                    "size": n,
                    "fixed_density": FIXED_DENSITY,
                    "f_density": density,
                    "nnz_F": nnz_f,
                    "loop_order": order,
                    "executed_order": RECORDER.orders[0],
                    "est_cost": RECORDER.cost,
                    "seconds": float(np.mean(kernel)),
                    "plan_seconds": float(
                        np.mean([w - k for w, k in zip(wall, kernel, strict=True)])
                    ),
                }
                writer.writerow(row)
                file.flush()
                print(
                    f"    mean {row['seconds'] * 1e3:.3f} ms; "
                    f"cost {row['est_cost']:.6g}",
                    flush=True,
                )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
