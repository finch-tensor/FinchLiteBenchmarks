import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm


def format_axes(axis, k_values, nnz_values):
    axis.set_xticks(range(len(k_values)), k_values, rotation=45)
    axis.set_yticks(
        range(len(nnz_values)),
        [f"2^{int(np.log2(nnz / 2048**2))}" for nnz in nnz_values],
    )
    axis.set_xlabel("K")


def plot_baseline_speedups(medians, k_values, nnz_values):
    baselines = ("numpy", "scipy", "pydata_sparse")
    speedups = {}

    for finch in ("finch_autoscheduled", "best_fixed_finch"):
        for baseline in baselines:
            values = np.empty((len(nnz_values), len(k_values)))
            for row, nnz in enumerate(nnz_values):
                for column, k in enumerate(k_values):
                    if finch == "best_fixed_finch":
                        finch_time = min(
                            medians[(k, nnz, "finch_fused")],
                            medians[(k, nnz, "finch_unfused")],
                        )
                    else:
                        finch_time = medians[(k, nnz, finch)]
                    values[row, column] = np.log2(
                        medians[(k, nnz, baseline)] / finch_time
                    )
            speedups[(finch, baseline)] = values

    limit = max(np.max(np.abs(values)) for values in speedups.values())
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

    for finch, label, filename in (
        ("finch_autoscheduled", "Autoscheduled Finch", "sddmm_autoscheduled_speedups.png"),
        ("best_fixed_finch", "Best Fixed Finch", "sddmm_best_fixed_speedups.png"),
    ):
        figure, axes = plt.subplots(1, 3, figsize=(20, 11), constrained_layout=True)
        for column, baseline in enumerate(baselines):
            axis = axes[column]
            values = speedups[(finch, baseline)]
            image = axis.imshow(
                values, origin="lower", aspect="auto", cmap="RdBu", norm=norm
            )
            axis.set_title(f"{label} vs {baseline.replace('_', ' ').title()}")
            format_axes(axis, k_values, nnz_values)
            if column == 0:
                axis.set_ylabel("Density of C")

            for row in range(len(nnz_values)):
                for cell_column in range(len(k_values)):
                    speedup = 2 ** values[row, cell_column]
                    number = (
                        f"{speedup:.0f}×"
                        if speedup >= 10
                        else f"{speedup:.1f}×"
                        if speedup >= 0.1
                        else f"{speedup:.2f}×"
                    )
                    axis.text(
                        cell_column,
                        row,
                        number,
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white" if abs(values[row, cell_column]) > limit * 0.55 else "black",
                    )

        colorbar = figure.colorbar(image, ax=axes, shrink=0.75)
        colorbar.set_label("log₂(baseline runtime / Finch runtime)")
        colorbar.ax.axhline(0, color="black", linewidth=1)
        figure.suptitle(f"SDDMM speedup of {label} over Python baselines")
        figure.savefig(filename, dpi=200)
        plt.close(figure)


def plot_finch_comparison(medians, recommendations, k_values, nnz_values):
    shape = (len(nnz_values), len(k_values))
    fixed_comparison = np.empty(shape)
    auto_comparison = np.empty(shape)
    recommendation_values = np.empty(shape)
    fixed_labels = np.empty(shape, dtype=object)
    auto_labels = np.empty(shape, dtype=object)
    recommendation_labels = np.empty(shape, dtype=object)

    for row, nnz in enumerate(nnz_values):
        for column, k in enumerate(k_values):
            fused = medians[(k, nnz, "finch_fused")]
            unfused = medians[(k, nnz, "finch_unfused")]
            autoscheduled = medians[(k, nnz, "finch_autoscheduled")]
            fixed = min(fused, unfused)
            fixed_comparison[row, column] = np.log2(unfused / fused)
            auto_comparison[row, column] = np.log2(fixed / autoscheduled)

            if fused <= unfused:
                fixed_winner = "fused"
                fixed_labels[row, column] = f"F {unfused / fused:.1f}×"
            else:
                fixed_winner = "unfused"
                fixed_labels[row, column] = f"U {fused / unfused:.1f}×"

            if autoscheduled <= fixed:
                auto_labels[row, column] = f"A {fixed / autoscheduled:.1f}×"
            else:
                auto_labels[row, column] = f"F {autoscheduled / fixed:.1f}×"

            recommendation = recommendations[(k, nnz)]
            recommendation_values[row, column] = 1 if recommendation == "fused" else -1
            recommendation_labels[row, column] = (
                f"{'F' if recommendation == 'fused' else 'U'}"
                f"{'✓' if recommendation == fixed_winner else '✗'}"
            )

    limit = max(np.max(np.abs(fixed_comparison)), np.max(np.abs(auto_comparison)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    figure, axes = plt.subplots(1, 3, figsize=(20, 11), constrained_layout=True)

    for axis, values, labels, title in (
        (axes[0], fixed_comparison, fixed_labels, "Fused vs Unfused"),
        (axes[1], auto_comparison, auto_labels, "Autoscheduled vs Best Fixed"),
    ):
        image = axis.imshow(
            values, origin="lower", aspect="auto", cmap="RdBu", norm=norm
        )
        axis.set_title(title)
        format_axes(axis, k_values, nnz_values)
        for row in range(len(nnz_values)):
            for column in range(len(k_values)):
                axis.text(
                    column,
                    row,
                    labels[row, column],
                    ha="center",
                    va="center",
                    fontsize=5,
                    color="white" if abs(values[row, column]) > limit * 0.55 else "black",
                )

    axes[0].set_ylabel("Density of C")
    axes[2].imshow(
        recommendation_values,
        origin="lower",
        aspect="auto",
        cmap=ListedColormap(("#ff0000", "#0066ff")),
        vmin=-1,
        vmax=1,
    )
    axes[2].set_title("Autoscheduler Recommendation")
    format_axes(axes[2], k_values, nnz_values)
    for row in range(len(nnz_values)):
        for column in range(len(k_values)):
            axes[2].text(
                column,
                row,
                recommendation_labels[row, column],
                ha="center",
                va="center",
                fontsize=6,
            )

    colorbar = figure.colorbar(
        image, ax=axes, shrink=0.75, location="right"
    )
    colorbar.set_label("log₂(strategy 2 / strategy 1)")
    figure.suptitle("SDDMM Finch strategy comparison")
    figure.savefig("sddmm_finch_comparison.png", dpi=200)
    plt.close(figure)


def main():
    measurements = defaultdict(list)
    recommendations = {}
    with Path("sddmm_results.csv").open(newline="") as results:
        for row in csv.DictReader(results):
            key = (int(row["k"]), int(row["nnz_c"]), row["implementation"])
            measurements[key].append(float(row["seconds"]))
            if row["finch_recommends"]:
                recommendations[(int(row["k"]), int(row["nnz_c"]))] = row[
                    "finch_recommends"
                ]

    medians = {key: np.median(values) for key, values in measurements.items()}
    k_values = sorted({key[0] for key in medians})
    nnz_values = sorted({key[1] for key in medians})
    plot_baseline_speedups(medians, k_values, nnz_values)
    plot_finch_comparison(medians, recommendations, k_values, nnz_values)


if __name__ == "__main__":
    main()
