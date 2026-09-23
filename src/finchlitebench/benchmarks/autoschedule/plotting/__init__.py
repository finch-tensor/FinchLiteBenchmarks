"""Shared plotting for autoscheduler benchmark summaries."""

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


LOOP_ORDER = ("default", "greedy", "bfs")
LOGIC_ORDER = ("default", "galley-greedy", "galley-bfs", "galley-dfs")


def plot_results(name, results_dir, argv=None):
    """Plot median timings and interquartile ranges from a summary CSV."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments:
        csv_path = Path(arguments[0])
    else:
        files = sorted(results_dir.glob(f"{name}_*_summary.csv"))
        if not files:
            raise FileNotFoundError(f"no {name}_*_summary.csv in {results_dir}")
        csv_path = files[-1]

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        configurations = sorted(
            {(row["loop"], row["logic"]) for row in mode_rows},
            key=lambda item: (
                LOGIC_ORDER.index(item[1]),
                LOOP_ORDER.index(item[0]),
            ),
        )
        workloads = sorted({row["workload"] for row in mode_rows})
        rows_by_key = {
            (row["workload"], row["loop"], row["logic"]): row for row in mode_rows
        }

        x = np.arange(len(configurations), dtype=float)
        width = 0.8 / len(workloads)
        figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
        metrics = (
            ("logic_s", "Logic optimizer"),
            ("loop_s", "Loop orderer"),
            ("pipeline_s", "Complete mode pipeline"),
        )

        for axis, (metric, title) in zip(axes, metrics, strict=True):
            for index, workload in enumerate(workloads):
                selected = [
                    rows_by_key[(workload, loop, logic)]
                    for loop, logic in configurations
                ]
                median = np.array(
                    [float(row[f"{metric}_median"]) for row in selected]
                )
                q1 = np.array([float(row[f"{metric}_q1"]) for row in selected])
                q3 = np.array([float(row[f"{metric}_q3"]) for row in selected])
                positions = x - 0.4 + width / 2 + index * width
                bars = axis.bar(
                    positions,
                    median,
                    width,
                    yerr=np.vstack((median - q1, q3 - median)),
                    capsize=2,
                    label=workload,
                )
                # Extend the default/default bar height across the chart so
                # every configuration can be compared with the baseline.
                axis.hlines(
                    median[0],
                    positions[0],
                    x[-1] + 0.4,
                    colors=[bars.patches[0].get_facecolor()],
                    linestyles="--",
                    linewidth=1,
                    alpha=0.8,
                )

            axis.set_title(title)
            axis.set_ylabel("seconds (median, IQR)")
            axis.set_xticks(x)
            axis.set_xticklabels(
                [f"{loop}\n{logic}" for loop, logic in configurations],
                rotation=45,
                ha="right",
                fontsize=8,
            )
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", linestyle=":", alpha=0.4)

        axes[0].legend(title="workload", fontsize=8)
        figure.suptitle(f"{mode}: {csv_path.name}")
        output = csv_path.with_name(
            csv_path.stem.removesuffix("_summary") + f"_{mode}.png"
        )
        figure.savefig(output, dpi=150)
        plt.close(figure)
        print(f"wrote {output}")
