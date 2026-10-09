import math
from statistics import median
from time import perf_counter

import numpy as np
import scipy.sparse as sps

import matplotlib.pyplot as plt
import pandas as pd

import finch.interface as ft
from finch.autoschedule.default_schedulers import COMPILE_JULIA_GALLEY
from finch.autoschedule.tensor_stats import UniformStatsFactory
from finch.tensor.fiber_tensor import FiberTensor


def chain(a, b, c, sched, order):
    if order == "(AB)C":
        ab = ft.compute(a @ b, ctx=sched)
        return ft.compute(ft.defer(ab) @ c, ctx=sched)
    if order == "A(BC)":
        bc = ft.compute(b @ c, ctx=sched)
        return ft.compute(a @ ft.defer(bc), ctx=sched)
    return ft.compute(a @ b @ c, ctx=sched)


def calc_time(order, a, b, c, sched, reps):
    sched.ctx.cached_kernels.clear()
    chain(a, b, c, sched, order)
    times = []
    for _ in range(reps):
        start = perf_counter()
        chain(a, b, c, sched, order)
        times.append(perf_counter() - start)
    return median(times)

def make_plot(df, exps, b_exp, path):
    top = exps[-1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(exps, df["(AB)C"], "k-s", lw=1.5, ms=4, label="(AB)C")
    ax.plot(
        exps,
        df["A(BC)"],
        "--^",
        color="#3b3833",
        lw=1.5,
        ms=4,
        label="A(BC)",
    )
    ax.plot(
        exps,
        df["Finch"],
        "-o",
        color="#00b1f7",
        lw=2,
        ms=4,
        label="Finch",
    )
    ax.set_yscale("log")
    times = df[["Finch", "(AB)C", "A(BC)"]]
    lo = math.floor(math.log10(times.min().min()))
    hi = math.ceil(math.log10(times.max().max()))
    ticks = [m * 10.0**e for e in range(lo, hi) for m in (1, 3)] + [10.0**hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.grid(axis="y", color="#e5e5e3", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xticks(exps[::3])
    ax.set_xticklabels([f"A: 2^{a}\nC: 2^{top + 1 - a}" for a in exps[::3]])
    ax.set_xlabel(f"nnz(A) grows, nnz(C) shrinks   (nnz(B) fixed at 2^{b_exp})")
    ax.set_ylabel("runtime (seconds, log scale)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
 


def run_path(size=2048, reps=20):
    jl_galley_sched = COMPILE_JULIA_GALLEY
    jl_galley_sched.ctx.stats_factory = UniformStatsFactory()

    top = int(math.log2(size * size))
    exps = list(range(1, top + 1))
    b_exp = top - 5
    rng = np.random.default_rng(0)
    rows = []

    for a_exp, c_exp in zip(exps, reversed(exps), strict=True):
        mats = []
        for nnz in (2**a_exp, 2**b_exp, 2**c_exp):
            flat = rng.choice(size * size, size=nnz, replace=False)
            r, c = np.divmod(flat, size)
            mat = sps.csr_matrix((rng.random(nnz), (r, c)), shape=(size, size))
            mat.indices = mat.indices.astype(np.int64)
            mat.indptr = mat.indptr.astype(np.int64)
            mats.append(FiberTensor.from_scipy_csr(mat))
        a, b, c = map(ft.defer, mats)

        row = {
            "nnz_A": 2**a_exp,
            "nnz_B": 2**b_exp,
            "nnz_C": 2**c_exp,
            "(AB)C": calc_time("(AB)C", a, b, c, jl_galley_sched, reps),
            "A(BC)": calc_time("A(BC)", a, b, c, jl_galley_sched, reps),
            "Finch": calc_time("Finch", a, b, c, jl_galley_sched, reps),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"computes_path_{size}.csv", index=False)
    make_plot(df, exps, b_exp, f"computes_path_{size}.png")


if __name__ == "__main__":
    run_path()