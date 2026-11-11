import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import finch as ft
from finch.autoschedule import COMPILE_JULIA_GALLEY
from finch.tensor.scalar import ConstantScalar

from finchtensorbench.benchmarks.microbenchmarks.util import feature_matrix, paper_spgemm
from finchtensorbench.benchmarks.microbenchmarks.util.scheduler_trace import (
    capture_scheduler,
    configured_scheduler,
    finish_capture,
)
from finchtensorbench.benchmarks.microbenchmarks.util.search_trace import capture_search

ROWS = 5000
COLS = 5000
SDDMM_ROWS = 10000  # SDDMM uses larger matrices than chain matmul
SDDMM_COLS = 10000
SDDMM_DENSITIES = (0.00001, 0.0001, 0.001, 0.01)
CHAIN_AB_DENSITY = 0.01  # A and B in A @ B @ C
CHAIN_DENSITIES = (0.000001, 0.00001, 0.0001, 0.001, 0.01)  # C
SEED = 42


def random_csr(shape, density, seed):
    matrix = sps.csr_array(sps.random(
        *shape, density=density, format="csr",
        random_state=np.random.default_rng(seed), data_rvs=np.ones,
    ))
    matrix.indices = matrix.indices.astype(np.intp)
    matrix.indptr = matrix.indptr.astype(np.intp)
    return matrix


######################### SDDMM ##########################

def sddmm(a, b, c):
    return ft.compute(ft.defer(a) * (ft.defer(b) @ ft.defer(c)))

def sddmm_inputs(k: int, c_scipy: sps.csr_array, seed: int):
    rng = np.random.default_rng(seed + k)
    rows, cols = c_scipy.shape
    a = rng.random((rows, k))
    b = rng.random((k, cols))
    return (
        ft.asarray(a),
        ft.asarray(b),
        ft.asarray(c_scipy),
    )

def sddmm_cases():
    results = []
    for density in SDDMM_DENSITIES:
        mask = random_csr((SDDMM_ROWS, SDDMM_COLS), density, SEED + 2)
        b, c, mask_finch = sddmm_inputs(128, mask, SEED)
        results.append(((mask_finch, b, c), mask.nnz / (SDDMM_ROWS * SDDMM_COLS)))
    return results


######################### Chain Matmul ##########################

def matmul(a, b, c):
    return ft.compute(ft.defer(a) @ ft.defer(b) @ ft.defer(c))

def chain_cases():
    results = []
    shape = (ROWS, COLS)
    a = ft.asarray(random_csr(shape, CHAIN_AB_DENSITY, SEED))
    b = ft.asarray(random_csr(shape, CHAIN_AB_DENSITY, SEED + 1))
    for density in CHAIN_DENSITIES:
        c = random_csr(shape, density, SEED + 2)
        results.append(((a, b, ft.asarray(c)), c.nnz / (ROWS * COLS)))
    return results


######################### Gustavson / Outer Product ##########################

def spgemm(a, b):
    return ft.compute(ft.defer(a) @ ft.defer(b).T)


def spgemm_cases():
    results = []
    for shared in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
        a, b = paper_spgemm.make_inputs(shared)
        a = ft.defer(ft.asarray(a.T)).T
        b = ft.defer(ft.asarray(b.T)).T
        results.append(((a, b), shared))
    return results


######################### Matrix Features ##########################

def matrix_features(a, count):
    # Row Lp norms, p = 1..count. The values are nonnegative, so no abs is needed.
    x = ft.defer(a)
    outputs = []
    for p in feature_matrix.POWERS[:count]:
        powered = x if p == 1 else x ** ConstantScalar(np.int64(p))
        total = ft.sum(powered, axis=1)
        outputs.append(total if p == 1 else total ** ConstantScalar(np.float64(1.0 / p)))
    return ft.compute(tuple(outputs))


def matrix_feature_cases():
    matrix = feature_matrix.make_matrix()
    matrix.indices = matrix.indices.astype(np.intp)
    matrix.indptr = matrix.indptr.astype(np.intp)
    a = ft.asarray(matrix)
    return [(f"features_{count}", (a, count))
            for count in range(1, len(feature_matrix.POWERS) + 1)]


######################### Main ##########################

def warm_case(name, function, inputs, parameter, *, capture=False):
    workspaces = {"gustavson": True, "outer": False}.get(name)
    if name.startswith("features_"):
        label = f"features={inputs[1]}"
    elif workspaces is None:
        label = f"density={parameter:.6g}"
    else:
        mode = "enabled" if workspaces else "disabled"
        label = f"shared_k={parameter}, workspace insertion={mode}"
    print(f"Warming {name}: {label}", flush=True)
    if capture:
        scheduler, trace = capture_scheduler(COMPILE_JULIA_GALLEY, workspaces=workspaces)
    else:
        scheduler = configured_scheduler(COMPILE_JULIA_GALLEY, workspaces=workspaces)
        trace = None
    if capture and workspaces is not None:
        trace.events.insert(0, (
            "Paper SpGEMM dataset",
            f"Operation: A @ B.T; base input shapes: {inputs[0].shape}, {inputs[1].shape}.\n"
            f"Original artifact: {paper_spgemm.SOURCE}\n"
            f"{paper_spgemm.ACTIVE_COLUMNS} active columns per input, "
            f"{paper_spgemm.NONZEROS_PER_COLUMN} nonzeros per active column, "
            f"{parameter} shared columns.\n"
            "The gustavson/outer names identify the workspace-on/off experiment slots; "
            "the scheduler chooses the actual loop order and storage in both.",
        ))
    with ft.with_default_scheduler(scheduler):
        if capture:
            with capture_search(trace.events):
                function(*inputs)
        else:
            function(*inputs)
    if capture:
        scheduler = finish_capture(scheduler)
    return scheduler, trace, label


def cases():
    benchmarks = {
        "sddmm": (sddmm, sddmm_cases),
        "chain": (matmul, chain_cases),
    }
    results = []
    for name, (function, make_cases) in benchmarks.items():
        for inputs, density in make_cases():
            results.append((name, function, inputs, density))
    for inputs, shared_k in spgemm_cases():
        for name in ("gustavson", "outer"):
            results.append((name, spgemm, inputs, shared_k))
    for name, inputs in matrix_feature_cases():
        results.append((name, matrix_features, inputs, feature_matrix.DENSITY))
    return results


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("graph/scripts/finch_result.csv"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(("benchmark", "density", "active_k", "repeat", "seconds", "implementation", "workspaces"))
        output.flush()
        for name, function, inputs, parameter in cases():
            is_spgemm = name in ("gustavson", "outer")
            density = "" if is_spgemm else parameter
            active_k = parameter if is_spgemm else ""
            samples = []
            scheduler, _, label = warm_case(name, function, inputs, parameter)
            workspace_mode = ("enabled" if name == "gustavson" else "disabled") if is_spgemm else ""
            implementation = f"Workspace {workspace_mode}" if is_spgemm else ""
            with ft.with_default_scheduler(scheduler):
                for trial in range(5):
                    gc.collect()
                    start = time.perf_counter()
                    result = function(*inputs)
                    seconds = time.perf_counter() - start
                    del result
                    writer.writerow((name, density, active_k, trial, seconds, implementation, workspace_mode))
                    output.flush()
                    samples.append(seconds)
            median = float(np.median(samples))
            print(f"{name}: {label}, median={median:.6g}s", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
