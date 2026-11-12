import argparse
import csv
import gc
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

import galley_jl_python as gl
from galley_jl_python.julia import jl

from finchtensorbench.benchmarks.scripts.util import paper_spgemm

ROWS = 5000
COLS = 5000
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
    for density in (0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0):
        mask = random_csr((ROWS, COLS), density, SEED + 2)
        b, c, mask_galley = sddmm_inputs(128, mask, SEED)
        results.append(((mask_galley, b, c), mask.nnz / (ROWS * COLS)))
    return results


######################### Chain Matmul ##########################

def matmul(a, b, c):
    return gl.compute(gl.lazy(a) @ gl.lazy(b) @ gl.lazy(c))


def chain_cases():
    results = []
    shape = (ROWS, COLS)
    a = gl.asarray(random_csr(shape, 0.1, SEED))
    b = gl.asarray(random_csr(shape, 0.1, SEED + 1))
    for density in (0.00001, 0.0001, 0.001, 0.01, 0.1):
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
    denominator = float(a.shape[1])
    mean = gl.sum(a, axis=1) / denominator
    outputs = [mean]
    if count >= 2:
        # E[x^2] - mean^2: avoids broadcasting `mean` against the sparse matrix.
        variance = gl.sum(a * a, axis=1) / denominator - mean * mean
        outputs.append(gl.sqrt(variance))
    if count >= 3:
        outputs.append(gl.max(a, axis=1))
    if count >= 4:
        outputs.append(gl.min(a, axis=1))
    return tuple(outputs)


def matrix_feature_cases():
    matrix = sps.csr_array(sps.random(
        ROWS, COLS, density=0.01, format="csr",
        random_state=np.random.default_rng(SEED),
    ))
    matrix.indices = matrix.indices.astype(np.intp)
    matrix.indptr = matrix.indptr.astype(np.intp)
    a = gl.asarray(matrix)
    density = matrix.nnz / (ROWS * COLS)
    return [(f"features_{count}", (a, count), density) for count in (1, 2, 3, 4)]


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
    for name, inputs, density in matrix_feature_cases():
        results.append((name, matrix_features, inputs, density))
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
