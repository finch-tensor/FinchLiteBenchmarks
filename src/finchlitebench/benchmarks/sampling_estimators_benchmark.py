from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import scipy.io
import finchlite as fl
import finchlite.algebra.ffuncs as ffuncs
from finchlite.autoschedule.tensor_stats import (BlockedStats,BlockedStatsFactory,DCStatsFactory,DenseStatsFactory,DenseStatsFactory,UniformStatsFactory, DatabaseStatsFactory,SamplingStats,SamplingStatsFactory)
from finchlite.finch_logic import Field
import csv

#Defining fields 
i,j,k,l = Field("i"), Field("j"),Field("k"), Field("l")

ESTIMATOR_MODELS = {
    "uj1":       SamplingStatsFactory(sample_prob=0.25, estimator="uj1"),
    "sj1":       SamplingStatsFactory(sample_prob=0.25, estimator="sj1"),
    "uj2":       SamplingStatsFactory(sample_prob=0.25, estimator="uj2"),
    "schlosser": SamplingStatsFactory(sample_prob=0.25, estimator="schlosser"),
    "sh2":       SamplingStatsFactory(sample_prob=0.25, estimator="sh2"),
    "sh3":       SamplingStatsFactory(sample_prob=0.25, estimator="sh3"),
    "good1":       SamplingStatsFactory(sample_prob=0.25, estimator="good1"),

}

ESTIMATOR_COLORS = {
    "uj1": "#E41A1C",
    "sj1": "#FF7F00",
    "uj2": "#4DAF4A",
    "schlosser": "#00BCD4",
    "sh2": "#A65628",
    "sh3": "#F781BF",
    "good1":  "#F3F781",
}

KERNELS = ["Hadamard", "SpGEMM", "SpGEMM2", "Triangle Counting"]

#creating dataset
def make_diagonal(n):
    return np.eye(n,dtype=np.float64)

def make_tridiagonal(n):
    A = np.eye(n,k=0) + np.eye(n,k=1) + np.eye(n,k=-1)
    return (A>0).astype(np.float64)

def make_banded(n,bw=5):
    r,c = np.indices((n,n))
    return (np.abs(r - c) <= bw).astype(np.float64)

def make_triangular(n):
    return np.triu(np.ones((n, n), dtype=np.float64))

def make_striped(n):
    A = np.zeros((n,n),dtype=np.float64)
    A[:,::5] = 1
    return A

def load_snap_matrix(path,n_max=50):
    mat = scipy.io.loadmat(path)
    A_sparse = mat["Problem"]["A"][0,0][:n_max,:n_max]
    A_dense = A_sparse.toarray()
    A_dense = np.ascontiguousarray((A_dense !=0).astype(np.float64))
    return A_dense

#dataset
N = 100

DATASETS = {
    #Structured matrices
    "Diagonal":    lambda: make_diagonal(N),
    "Tridiagonal": lambda: make_tridiagonal(N),
    "Banded":      lambda: make_banded(N),
    "Triangular":  lambda: make_triangular(N),
    "Striped":     lambda: make_striped(N),
  
    #snap matrices
    "ca-GrQc" : lambda:load_snap_matrix("tests/data/ca-GrQc.mat"),
    "ca-HepTh": lambda: load_snap_matrix("tests/data/ca-HepTh.mat"),
    "web-NotreDame":lambda: load_snap_matrix("tests/data/web-NotreDame.mat"),
}


#actual compute

def actual_hadamard(A,B):
    return max(int(np.count_nonzero(A*B)),1)

def actual_spgemm(A,B):
    C = A.astype(bool).astype(float)@ B.astype(bool).astype(float)
    return max(int(np.count_nonzero(C)),1)

def actual_spgemm2(A,B):
    C = (A.astype(bool).astype(float)@ B.astype(bool).astype(float)>0).astype(float)
    D = C @ B.astype(bool).astype(float)
    return max(int(np.count_nonzero(D)),1)

def actual_triangle(A):
    C = A.astype(bool).astype(float)@ A.astype(bool).astype(float)
    return max(int(np.count_nonzero(C*A)),1)

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
 
def est_triangle(factory, tns_a):
    s_a1 = factory(tns_a, (i, k))
    s_a2 = factory(tns_a, (k, j))
    mm   = factory.aggregate(ffuncs.add, 0.0, (k,),
                             factory.mapjoin(ffuncs.mul, s_a1, s_a2))
    s_a3 = factory(tns_a, (i, j))
    return factory.mapjoin(ffuncs.mul, mm, s_a3).estimate_non_fill_values()

def run_estimators():
    results = defaultdict(lambda: defaultdict(dict))

    for ds_name, ds_fn in DATASETS.items():
        print(f" dataset: {ds_name}")
        A = ds_fn()
        B = ds_fn()
        tns_a = fl.asarray(A)
        tns_b = fl.asarray(B)

        actual_vals = {
            "Hadamard": actual_hadamard(A, B),
            "SpGEMM": actual_spgemm(A, B),
            "SpGEMM2": actual_spgemm2(A, B),
            "Triangle Counting": actual_triangle(A),
        }

        for model_name, factory in ESTIMATOR_MODELS.items():
            factory._masks.clear()
            try:
                ests = {
                    "Hadamard": est_hadamard(factory, tns_a, tns_b),
                    "SpGEMM": est_spgemm(factory, tns_a, tns_b),
                    "SpGEMM2": est_spgemm2(factory, tns_a, tns_b),
                    "Triangle Counting":est_triangle(factory, tns_a),
                }
                for kernel, est in ests.items():
                    ratio = max(est, 1) / actual_vals[kernel]
                    results[kernel][ds_name][model_name] = ratio
            except Exception as e:
                print(f"  {model_name} on {ds_name} error: {e}")

    return results


def plot_estimators(results, out_path="results/estimator_comparison.png"):
    model_names = list(ESTIMATOR_MODELS.keys())
    dataset_names = list(DATASETS.keys())
    n_models = len(model_names)
    bar_w = 0.13
    x = np.arange(len(dataset_names))
    Y_MAX_LOG = 5
    Y_MIN_LOG = -5

    fig, axes = plt.subplots(len(KERNELS), 1, figsize=(18, 4.5 * len(KERNELS)))

    for ax, kernel in zip(axes, KERNELS):
        kernel_data = results.get(kernel, {})

        for mi, model_name in enumerate(model_names):
            ratios = [
                kernel_data.get(ds, {}).get(model_name, np.nan)
                for ds in dataset_names
            ]
            offset = (mi - n_models / 2 + 0.5) * bar_w
            log_ratios = [
                np.log2(r) if (not np.isnan(r) and r > 0) else np.nan
                for r in ratios
            ]
            display_ratios = [
                np.clip(r, Y_MIN_LOG, Y_MAX_LOG) if not np.isnan(r) else np.nan
                for r in log_ratios
            ]
            bars = ax.bar(x + offset, display_ratios, width=bar_w,
                          color=ESTIMATOR_COLORS.get(model_name, "gray"),
                          label=model_name, zorder=3)

            # overflow annotations
            for bar, log_r in zip(bars, log_ratios):
                if not np.isnan(log_r) and log_r > Y_MAX_LOG:
                    ax.text(bar.get_x() + bar.get_width()/2, Y_MAX_LOG*0.95,
                            f"+{log_r:.1f}", ha="center", va="top",
                            fontsize=7, color="black", fontweight="bold")
                elif not np.isnan(log_r) and log_r < Y_MIN_LOG:
                    ax.text(bar.get_x() + bar.get_width()/2, Y_MIN_LOG*0.95,
                            f"{log_r:.1f}", ha="center", va="bottom",
                            fontsize=7, color="black", fontweight="bold")

            # star for perfect
            for bar, log_r in zip(bars, log_ratios):
                if not np.isnan(log_r) and abs(log_r) < 0.1:
                    ax.plot(bar.get_x() + bar.get_width()/2, 0,
                            marker="x", color=ESTIMATOR_COLORS.get(model_name, "gray"),
                            markersize=10, zorder=5)

        ax.axhline(0, color="black", linewidth=1.5, linestyle="--", zorder=4,
                   label="Perfect estimate")
        ax.set_ylabel("log2(estimated / true nnz)", fontsize=9)
        ax.set_title(kernel, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=30, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
        ax.set_ylim(Y_MIN_LOG, Y_MAX_LOG)
        ax.axhline(1,  color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.axhline(-1, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.axhline(2,  color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
        ax.axhline(-2, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)

        if kernel == KERNELS[0]:
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                      fontsize=8, ncol=1, framealpha=0.9,
                      title=f"Estimator\n(sample_prob=0.5)",
                      borderaxespad=0)

    fig.suptitle(
        "Sampling Estimator Comparison : sample_prob=0.5\n",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


def plot_geomean_by_estimator(results, out_path="results/estimator_geomean.png"):
    model_names = list(ESTIMATOR_MODELS.keys())
    geomeans = {}
    for model_name in model_names:
        ratios = []
        for kernel, by_dataset in results.items():
            for ds_name, by_model in by_dataset.items():
                if model_name in by_model:
                    ratios.append(by_model[model_name])
        if not ratios:
            geomeans[model_name] = float("nan")
            continue
        geomeans[model_name] = 2 ** np.mean(np.log2(ratios))
 
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(geomeans.keys())
    vals = [np.log2(geomeans[n]) for n in names]
    colors = ["tab:red", "tab:orange", "tab:green", "tab:cyan", "tab:brown", "tab:pink"]
    bars = ax.bar(names, vals, color=colors[: len(names)])
    for bar,name in zip(bars,names):
        bar.set_label(name)
    ref_line = ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Perfect estimate")
    ax.set_ylabel("log2(geomean estimated / true) across all kernels & datasets")
    ax.set_title("Geometric mean of different estimators")
    ax.legend(handles=[ref_line] + list(bars), loc="upper left", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved {out_path}")
    return geomeans

def save_results_csv(results, out_path="results/estimator_comparison.csv"):
    fieldnames = ["kernel", "dataset", "estimator", "ratio", "log2_ratio"]
 
    rows = []
    for kernel, by_dataset in results.items():
        for ds_name, by_model in by_dataset.items():
            for model_name, ratio in by_model.items():
                log2_ratio = np.log2(ratio) if ratio > 0 else float("nan")
                rows.append({
                    "kernel": kernel,
                    "dataset": ds_name,
                    "estimator": model_name,
                    "ratio": ratio,
                    "log2_ratio": log2_ratio,
                })
 
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
 
    print(f"Saved {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    print("Running estimator comparison")
    est_results = run_estimators()
    plot_estimators(est_results, "results/estimator_comparison.png")
    plot_geomean_by_estimator(est_results, "results/estimator_geomean.png")
    save_results_csv(est_results, "results/estimator_comparison.csv")