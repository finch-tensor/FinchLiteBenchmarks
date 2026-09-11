from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

import finch as fl
from finch import ffuncs
from finch.autoschedule.tensor_stats import (
    DCStatsFactory,
    DenseStatsFactory,
    LPStatsFactory,
    UniformStatsFactory,
    VPStatsFactory,
    SamplingStatsFactory,
)
from finch.finch_logic import Field

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "tests" / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

i, j, k, l, m, n = (Field(name) for name in ("i", "j", "k", "l", "m", "n"))

def load_snap_matrix(path):
    mat = scipy.io.loadmat(path)
    matrix = mat["Problem"]["A"][0, 0].tocsr()
    matrix.data = np.ones(matrix.nnz, dtype=np.float64)
    return fl.asarray(matrix)

KERNELS = ["Hadamard", "SpGEMM", "SpGEMM2", "SpGEMM3", "SpGEMM4", "Triangle Counting"]
DISPLAY_KERNELS = ["Hadamard", "SpGEMM", "SpGEMM2", "Triangle Counting"]

DATASETS = {
    "ca-GrQc": lambda: load_snap_matrix(DATA_DIR / "ca-GrQc.mat"),
    "ca-HepTh": lambda: load_snap_matrix(DATA_DIR / "ca-HepTh.mat"),
    "web-NotreDame": lambda: load_snap_matrix(DATA_DIR / "web-NotreDame.mat"),
    "Boeing/ct20stif": lambda: load_snap_matrix(DATA_DIR / "ct20stif.mat"),
    "ca-HepPh": lambda: load_snap_matrix(DATA_DIR / "ca-HepPh.mat"),
    "Boeing/bcsstk39": lambda: load_snap_matrix(DATA_DIR / "bcsstk39.mat"),
}

STATS_MODEL = {
    "Dense" : DenseStatsFactory(),
    "Uniform" : UniformStatsFactory(),
    "DC" : DCStatsFactory(),
    "VP" : VPStatsFactory(),
    "LP" : LPStatsFactory(),
    "Sampling" : None,
}

def result_paths(model_name, sampling_name=None):
    output_name = model_name
    if model_name == "Sampling" and sampling_name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", sampling_name):
            raise ValueError(
                "sampling name may contain only letters, numbers, '.', '-', and '_'"
            )
        output_name = f"{model_name}_{sampling_name}"
    return (
        RESULTS_DIR / f"{output_name}_stats_results.csv",
        RESULTS_DIR / f"{output_name}_stats_accuracy.png",
    )

DATASET_COLORS = {
    "ca-GrQc": "#4878CF",
    "ca-HepTh": "#6ACC65",
    "web-NotreDame": "#D65F5F",
    "Boeing/ct20stif": "#FF9F0A",
    "ca-HepPh": "#9B59B6",
    "Boeing/bcsstk39": "#8C564B",
}

########################################### Kernels #################################################

def actual_hadamard(A, B):
    return max(int(A.multiply(B).count_nonzero()), 1)


def actual_spgemm(A, B):
    C = A.astype(bool).astype(float) @ B.astype(bool).astype(float)
    return max(int(C.count_nonzero()), 1)


def actual_spgemm2(A, B):
    C = (A.astype(bool).astype(float) @ B.astype(bool).astype(float) > 0).astype(float)
    D = C @ B.astype(bool).astype(float)
    return max(int(D.count_nonzero()), 1)


def actual_spgemm3(A, B, C, D):
    AB = (A.astype(bool).astype(float) @ B.astype(bool).astype(float) > 0).astype(float)
    ABC = (AB @ C.astype(bool).astype(float) > 0).astype(float)
    ABCD = ABC @ D.astype(bool).astype(float)
    return max(int(ABCD.count_nonzero()), 1)


def actual_spgemm4(A, B, C, D, E):
    AB = (A.astype(bool).astype(float) @ B.astype(bool).astype(float) > 0).astype(float)
    ABC = (AB @ C.astype(bool).astype(float) > 0).astype(float)
    ABCD = (ABC @ D.astype(bool).astype(float) > 0).astype(float)
    ABCDE = ABCD @ E.astype(bool).astype(float)
    return max(int(ABCDE.count_nonzero()), 1)


def actual_triangle(A):
    C = A.astype(bool).astype(float) @ A.astype(bool).astype(float)
    return max(int(C.multiply(A).count_nonzero()), 1)


#stats estimate
def est_hadamard(factory,tns_a,tns_b):
    s_a = factory(tns_a,(i,j))
    s_b = factory(tns_b,(i,j))
    return factory.mapjoin(ffuncs.mul, s_a, s_b).estimate_non_fill_values()


def est_spgemm(factory, tns_a, tns_b):
    s_a = factory(tns_a, (i, k))
    s_b = factory(tns_b, (k, j))
    mm  = factory.mapjoin(ffuncs.mul, s_a, s_b)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm).estimate_non_fill_values()


def est_spgemm2(factory, tns_a, tns_b):
    s_a  = factory(tns_a, (i, l))
    s_b1 = factory(tns_b, (l, k))
    mm1  = factory.aggregate(ffuncs.add, 0.0, (l,),
                             factory.mapjoin(ffuncs.mul, s_a, s_b1))
    s_b2 = factory(tns_b, (k, j))
    mm2  = factory.mapjoin(ffuncs.mul, mm1, s_b2)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm2).estimate_non_fill_values()


def est_spgemm3(factory, tns_a, tns_b, tns_c, tns_d):
    s_a = factory(tns_a, (i, m))
    s_b = factory(tns_b, (m, l))
    mm1 = factory.aggregate(ffuncs.add, 0.0, (m,),
                            factory.mapjoin(ffuncs.mul, s_a, s_b))
    s_c = factory(tns_c, (l, k))
    mm2 = factory.aggregate(ffuncs.add, 0.0, (l,),
                            factory.mapjoin(ffuncs.mul, mm1, s_c))
    s_d = factory(tns_d, (k, j))
    mm3 = factory.mapjoin(ffuncs.mul, mm2, s_d)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm3).estimate_non_fill_values()


def est_spgemm4(factory, tns_a, tns_b, tns_c, tns_d, tns_e):
    s_a = factory(tns_a, (i, n))
    s_b = factory(tns_b, (n, m))
    mm1 = factory.aggregate(ffuncs.add, 0.0, (n,),
                            factory.mapjoin(ffuncs.mul, s_a, s_b))
    s_c = factory(tns_c, (m, l))
    mm2 = factory.aggregate(ffuncs.add, 0.0, (m,),
                            factory.mapjoin(ffuncs.mul, mm1, s_c))
    s_d = factory(tns_d, (l, k))
    mm3 = factory.aggregate(ffuncs.add, 0.0, (l,),
                            factory.mapjoin(ffuncs.mul, mm2, s_d))
    s_e = factory(tns_e, (k, j))
    mm4 = factory.mapjoin(ffuncs.mul, mm3, s_e)
    return factory.aggregate(ffuncs.add, 0.0, (k,), mm4).estimate_non_fill_values()


def est_triangle(factory, tns_a):
    s_a1 = factory(tns_a, (i, k))
    s_a2 = factory(tns_a, (k, j))
    mm   = factory.aggregate(ffuncs.add, 0.0, (k,),
                             factory.mapjoin(ffuncs.mul, s_a1, s_a2))
    s_a3 = factory(tns_a, (i, j))
    return factory.mapjoin(ffuncs.mul, mm, s_a3).estimate_non_fill_values()

############################################# Plot ################################################

def save_csv(results, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "kernel",
                "dataset",
                "model",
                "actual_nnz",
                "estimated_nnz",
                "ratio",
                "time_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(results)


def plot(results, out_path, model_name=None):
    dataset_names = list(DATASETS)
    ratios = {(row["kernel"], row["dataset"]): row["ratio"] for row in results}
    x = np.arange(len(dataset_names))
    y_min, y_max = -4, 10
    fig, axes = plt.subplots(
        len(DISPLAY_KERNELS), 1, figsize=(18, 5 * len(DISPLAY_KERNELS)), squeeze=False
    )

    for ax, kernel in zip(axes[:, 0], DISPLAY_KERNELS):
        values = [ratios.get((kernel, dataset), np.nan) for dataset in dataset_names]
        log_ratios = [
            np.log2(value) if not np.isnan(value) and value > 0 else np.nan
            for value in values
        ]
        displayed = [
            np.clip(value, y_min, y_max) if not np.isnan(value) else np.nan
            for value in log_ratios
        ]
        bars = ax.bar(
            x,
            displayed,
            width=0.65,
            color=[DATASET_COLORS[dataset] for dataset in dataset_names],
        )
        for bar, value in zip(bars, log_ratios):
            if np.isnan(value):
                continue
            x_center = bar.get_x() + bar.get_width() / 2
            if value > y_max:
                ax.text(x_center, y_max * 0.95, f"+{value:.1f}", ha="center", va="top")
            elif value < y_min:
                ax.text(x_center, y_min * 0.95, f"{value:.1f}", ha="center", va="bottom")
            elif abs(value) < 0.1:
                ax.plot(x_center, 0, marker="x", color="black", markersize=10, zorder=5)

        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=4)
        ax.set_title(kernel, fontweight="bold")
        ax.set_ylabel("log2(estimated / actual nnz)")
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=25, ha="right")
        ax.set_ylim(y_min, y_max)
        ax.grid(axis="y", linestyle=":", alpha=0.5)

    if model_name is None:
        model_name = results[0]["model"] if results else "Stats"
    fig.suptitle(
        f"{model_name} Stats Accuracy on Full Sparse Matrices",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(
    model_name,
    sampling_name=None,
    sample_prob=0.5,
    estimator="uj1",
):
    run_start = perf_counter()
    if model_name == "Sampling":
        factory = SamplingStatsFactory(sample_prob=sample_prob, estimator=estimator)
        if sampling_name is None:
            sampling_name = f"{estimator}_p{sample_prob:g}"
        plot_model_name = f"Sampling ({estimator}, probability={sample_prob:g})"
        csv_path, plot_path = result_paths(model_name, sampling_name)
    else:
        factory = STATS_MODEL[model_name]
        plot_model_name = model_name
        csv_path, plot_path = result_paths(model_name)
    results = []
    kernel_times = {kernel: 0.0 for kernel in KERNELS}
    estimators_warmed_up = False

    for ds_name, ds_fn in DATASETS.items():
        print(f" dataset :{ds_name}")
        tns_a = ds_fn()
        A = tns_a.to_scipy()
        print(
            f"  shape={A.shape}, size={A.shape[0] * A.shape[1]:,}, "
            f"nnz={A.nnz:,}"
        )
        estimators = {
            "Hadamard": lambda: est_hadamard(factory, tns_a, tns_a),
            "SpGEMM": lambda: est_spgemm(factory, tns_a, tns_a),
            "SpGEMM2": lambda: est_spgemm2(factory, tns_a, tns_a),
            # "SpGEMM3": lambda: est_spgemm3(factory, tns_a, tns_a, tns_a, tns_a),
            # "SpGEMM4": lambda: est_spgemm4(factory, tns_a, tns_a, tns_a, tns_a, tns_a),
            "Triangle Counting": lambda: est_triangle(factory, tns_a),
        }
        actuals = {
            "Hadamard": lambda: actual_hadamard(A, A),
            "SpGEMM": lambda: actual_spgemm(A, A),
            "SpGEMM2": lambda: actual_spgemm2(A, A),
            # "SpGEMM3": lambda: actual_spgemm3(A, A, A, A),
            # "SpGEMM4": lambda: actual_spgemm4(A, A, A, A, A),
            "Triangle Counting": lambda: actual_triangle(A),
        }

        if not estimators_warmed_up:
            print("  Warming up estimators once...")
            for kernel, estimate in estimators.items():
                try:
                    estimate()
                except Exception as error:
                    print(f"  {model_name} {kernel} warm-up error: {error}")
            estimators_warmed_up = True

        for kernel, estimate in estimators.items():
            kernel_start = perf_counter()
            actual_nnz = actuals[kernel]()
            try:
                estimated_nnz = estimate()
            except Exception as error:
                kernel_times[kernel] += perf_counter() - kernel_start
                print(f"{model_name} {kernel} on {ds_name} errors: {error}")
                continue
            elapsed = perf_counter() - kernel_start
            kernel_times[kernel] += elapsed
            ratio = max(estimated_nnz, 1) / actual_nnz
            print(
                f"  {kernel}: estimated={estimated_nnz:.6g}, "
                f"actual={actual_nnz}, ratio={ratio:.6g}, "
                f"time={elapsed:.6f}s"
            )
            results.append(
                {
                    "kernel": kernel,
                    "dataset": ds_name,
                    "model": model_name,
                    "actual_nnz": actual_nnz,
                    "estimated_nnz": estimated_nnz,
                    "ratio": ratio,
                    "time_seconds": elapsed,
                }
            )
            save_csv(results, csv_path)
        if results:
            plot(results, plot_path, plot_model_name)

    print("\nTiming summary:")
    for kernel in DISPLAY_KERNELS:
        print(f"  {kernel} total: {kernel_times[kernel]:.6f}s")
    print(f"  Whole computation: {perf_counter() - run_start:.6f}s")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model", choices=STATS_MODEL,)
    parser.add_argument("--sampling-name")
    parser.add_argument("--sample-prob", type=float, default=0.5)
    parser.add_argument("--estimator", default="uj1")
    args = parser.parse_args()

    if not 0.0 < args.sample_prob <= 1.0:
        parser.error("--sample-prob must be greater than 0 and at most 1")
    print("Running benchmarks...")
    run(
        args.model,
        args.sampling_name,
        sample_prob=args.sample_prob,
        estimator=args.estimator,
    )
