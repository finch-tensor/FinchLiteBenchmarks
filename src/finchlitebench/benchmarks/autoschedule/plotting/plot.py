"""Plot every benchmark result currently in the results folder."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LOOP_ORDER = ("default", "greedy", "bfs", "dfs")
LOGIC_ORDER = ("default", "galley-greedy", "galley-bfs", "galley-dfs")
PIPELINE_TITLES = {
    "scheduler": "Complete planning time",
    "end-to-end": "Complete time (planning + execution)",
}


def _configuration(row):
    # Older CSVs have no stats/pipeline columns.
    return (
        row.get("stats", ""),
        row.get("pipeline", ""),
        row["loop"],
        row["logic"],
    )


if __name__ == "__main__":
    results = sorted(RESULTS_DIR.glob("*_raw.csv"))
    if not results:
        raise FileNotFoundError(f"no raw CSV files in {RESULTS_DIR}")

    print(f"plotting {len(results)} results from {RESULTS_DIR}")
    for csv_path in results:
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

        for mode in sorted({row["mode"] for row in rows}):
            mode_rows = [row for row in rows if row["mode"] == mode]
            configurations = sorted(
                {_configuration(row) for row in mode_rows},
                key=lambda item: (
                    item[0],
                    item[1],
                    LOGIC_ORDER.index(item[3]),
                    LOOP_ORDER.index(item[2]),
                ),
            )
            workloads = sorted({row["workload"] for row in mode_rows})
            samples = {}
            for row in mode_rows:
                key = (row["workload"], _configuration(row))
                samples.setdefault(key, []).append(row)

            x = np.arange(len(configurations), dtype=float)
            width = 0.8 / len(workloads)
            figure, axes = plt.subplots(
                3, 1, figsize=(14, 11), constrained_layout=True
            )
            metrics = (
                ("logic_s", "Logic optimizer planning time"),
                ("loop_s", "Loop orderer planning time"),
                ("pipeline_s", PIPELINE_TITLES[mode]),
            )

            for axis, (metric, title) in zip(axes, metrics, strict=True):
                for index, workload in enumerate(workloads):
                    q1, median, q3 = np.array(
                        [
                            np.percentile(
                                [
                                    float(row[metric])
                                    for row in samples[workload, config]
                                ],
                                (25, 50, 75),
                            )
                            for config in configurations
                        ]
                    ).T
                    positions = x - 0.4 + width / 2 + index * width
                    bars = axis.bar(
                        positions,
                        median,
                        width,
                        yerr=np.vstack((median - q1, q3 - median)),
                        capsize=2,
                        label=workload,
                    )
                    # Dashed baseline at the default/default bar's height.
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
                    [
                        f"{stats}/{pipeline}\nloop-optimizer: {loop}\n"
                        f"logic-optimizer: {logic}"
                        for stats, pipeline, loop, logic in configurations
                    ],
                    rotation=45,
                    ha="right",
                    fontsize=8,
                )
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", linestyle=":", alpha=0.4)

            axes[0].legend(title="workload", fontsize=8)
            figure.suptitle(f"{mode}: {csv_path.name}")
            output = csv_path.with_name(
                csv_path.stem.removesuffix("_raw") + f"_{mode}.png"
            )
            figure.savefig(output, dpi=150)
            plt.close(figure)
            print(f"wrote {output}")
