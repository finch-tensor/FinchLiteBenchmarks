import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import galley_jl_python as gl
from galley_jl_python.julia import jl

from finchtensorbench.benchmarks.microbenchmarks.util import feature_matrix, paper_spgemm

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
    return gl.compute(gl.lazy(a) * (gl.lazy(b) @ gl.lazy(c)))


def sddmm_inputs(k, mask, seed):
    rng = np.random.default_rng(seed + k)
    rows, cols = mask.shape
    a = rng.random((rows, k))
    b = rng.random((k, cols))
    return gl.asarray(a), gl.asarray(b), gl.asarray(mask)


def sddmm_cases():
    results = []
    for density in SDDMM_DENSITIES:
        mask = random_csr((SDDMM_ROWS, SDDMM_COLS), density, SEED + 2)
        b, c, mask_galley = sddmm_inputs(128, mask, SEED)
        results.append(((mask_galley, b, c), mask.nnz / (SDDMM_ROWS * SDDMM_COLS)))
    return results


######################### Chain Matmul ##########################

def matmul(a, b, c):
    return gl.compute(gl.lazy(a) @ gl.lazy(b) @ gl.lazy(c))


def chain_cases():
    results = []
    shape = (ROWS, COLS)
    a = gl.asarray(random_csr(shape, CHAIN_AB_DENSITY, SEED))
    b = gl.asarray(random_csr(shape, CHAIN_AB_DENSITY, SEED + 1))
    for density in CHAIN_DENSITIES:
        c = random_csr(shape, density, SEED + 2)
        results.append(((a, b, gl.asarray(c)), c.nnz / (ROWS * COLS)))
    return results


######################### Gustavson / Outer Product ##########################

def spgemm(a, b):
    return gl.compute(gl.lazy(a) @ gl.lazy(b).T)


def spgemm_cases():
    results = []
    for shared in (1, 4, 16, 64, 256, 1024, 4096, 16384, 65536):
        a, b = paper_spgemm.make_inputs(shared)
        a = gl.lazy(gl.asarray(a.T)).T
        b = gl.lazy(gl.asarray(b.T)).T
        results.append(((a, b), shared))
    return results


######################### Matrix Features ##########################

@gl.compiled()
def matrix_features(a, count):
    # Row Lp norms, p = 1..count. The values are nonnegative, so no abs is needed.
    outputs = []
    for p in feature_matrix.POWERS[:count]:
        total = gl.sum(a if p == 1 else a ** float(p), axis=1)
        outputs.append(total if p == 1 else total ** (1.0 / p))
    return tuple(outputs)


def matrix_feature_cases():
    matrix = feature_matrix.make_matrix()
    matrix.indices = matrix.indices.astype(np.intp)
    matrix.indptr = matrix.indptr.astype(np.intp)
    a = gl.asarray(matrix)
    return [(f"features_{count}", (a, count)) for count in (1, 2, 3, 4)]


######################### Main ##########################

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
        results.append(("spgemm", spgemm, inputs, shared_k))
    for name, inputs in matrix_feature_cases():
        results.append((name, matrix_features, inputs, feature_matrix.DENSITY))
    return results


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("graph/scripts/galley_result.csv"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("benchmark", "density", "active_k", "repeat", "seconds"))
        for name, function, inputs, parameter in cases():
            is_spgemm = name == "spgemm"
            density = "" if is_spgemm else parameter
            active_k = parameter if is_spgemm else ""
            label = f"active_k={active_k}" if is_spgemm else f"density={density:.6g}"
            print(f"Warming {name}: {label}", flush=True)
            # Match Finch's cache=True: compile during warmup, then reuse the plan.
            jl.Galley.set_scheduler_b(jl.Galley.galley_scheduler(cache=True))
            function(*inputs)
            samples = []
            for trial in range(5):
                gc.collect()
                start = time.perf_counter()
                result = function(*inputs)
                seconds = time.perf_counter() - start
                del result
                writer.writerow((name, density, active_k, trial, seconds))
                output.flush()
                samples.append(seconds)
            median = float(np.median(samples))
            print(f"{name}: {label}, median={median:.6g}s", flush=True)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    run()
