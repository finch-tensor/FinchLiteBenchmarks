import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot fully fused chain orders and Spearman rho"
    )
    parser.add_argument(
        "csv", type=Path, nargs="?", default=Path("chain_microbenchmark.csv")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.csv.open(newline="") as file:
        for row in csv.DictReader(file):
            groups[int(row["size"]), float(row["f_density"])].append(row)
    if not groups:
        raise ValueError("The benchmark CSV has no measurements yet")
    output = args.out or args.csv.parent / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    columns = min(3, len(groups))
    fig, axes = plt.subplots(
        (len(groups) + columns - 1) // columns,
        columns,
        squeeze=False,
        figsize=(5.2 * columns, 4.2 * ((len(groups) + columns - 1) // columns)),
        layout="constrained",
    )
    summaries = []
    for ax, ((size, density), rows) in zip(
        axes.flat, sorted(groups.items()), strict=False
    ):
        costs = np.array([float(row["est_cost"]) for row in rows])
        seconds = np.array([float(row["seconds"]) for row in rows])
        if len(rows) < 2 or len(set(costs)) < 2 or len(set(seconds)) < 2:
            rho = p_value = float("nan")
        else:
            rho, p_value = spearmanr(costs, seconds)
        summaries.append(
            {
                "size": size,
                "f_density": density,
                "nnz_F": int(rows[0]["nnz_F"]),
                "samples": len(rows),
                "spearman_rho": rho,
                "p_value": p_value,
                "min_seconds": float(seconds.min()),
                "median_seconds": float(np.median(seconds)),
                "max_seconds": float(seconds.max()),
            }
        )
        ax.scatter(costs, seconds * 1000, s=14, alpha=0.5)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("log")
        ax.set_xlabel("Reported cost (model units)")
        ax.set_ylabel("Mean kernel runtime (ms)")
        ax.set_title(
            f"{size}x{size}; F density={density:.6g}\n"
            f"Spearman rho={rho:.3f}; {len(rows)} measured orders"
        )
        ax.grid(True, alpha=0.25)
        print(f"{size}x{size} F={density:.6g}: rho={rho:.6g}, n={len(rows)}")
    for ax in list(axes.flat)[len(groups) :]:
        ax.set_visible(False)
    fig.savefig(output / "chain_spearman.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for size in sorted({row["size"] for row in summaries}):
        points = [row for row in summaries if row["size"] == size]
        densities = [row["f_density"] for row in points]
        (line,) = ax.plot(
            densities,
            [row["median_seconds"] * 1000 for row in points],
            "o-",
            label=f"{size}x{size}: median across orders",
        )
        ax.fill_between(
            densities,
            [row["min_seconds"] * 1000 for row in points],
            [row["max_seconds"] * 1000 for row in points],
            alpha=0.15,
            color=line.get_color(),
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("F density")
    ax.set_ylabel("Mean kernel runtime (ms)")
    ax.set_title("Fully fused chain: all loop orders; shaded range is min to max")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output / "chain_runtime.png", dpi=180)
    plt.close(fig)

    summary = output / "chain_spearman.csv"
    with summary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"wrote graphs and {summary}")


if __name__ == "__main__":
    main()
