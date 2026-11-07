import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ORDERS = ("ijk", "ikj", "jik", "jki", "kij", "kji", "autoscheduled")


def heatmaps(rows, orders, metric, title, path, *, cmap, norm):
    q_values = sorted({row["q_exp"] for row in rows})
    p_values = sorted({row["p_exp"] for row in rows})
    r_values = sorted({row["r_exp"] for row in rows})
    fig, axes = plt.subplots(
        len(q_values),
        len(orders),
        figsize=(3.1 * len(orders), 3.1 * len(q_values)),
        squeeze=False,
        layout="constrained",
    )
    for qi, qe in enumerate(q_values):
        for oi, order in enumerate(orders):
            grid = np.full((len(p_values), len(r_values)), np.nan)
            for row in rows:
                if row["q_exp"] == qe and row["schedule"] == order:
                    grid[p_values.index(row["p_exp"]), r_values.index(row["r_exp"])] = (
                        row[metric]
                    )
            ax = axes[qi, oi]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, norm=norm)
            ax.set_title(f"{order}; B nnz = 2^{qe}")
            ax.set_xticks(range(0, len(r_values), 4), r_values[::4])
            ax.set_yticks(range(0, len(p_values), 4), p_values[::4])
            ax.set_xlabel("log2(C nnz)")
            ax.set_ylabel("log2(A nnz)")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes.ravel().tolist(), label=metric.replace("_", " "))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot SDDMM runtimes and cost agreement"
    )
    parser.add_argument(
        "csv", type=Path, nargs="?", default=Path("sddmm_microbenchmark.csv")
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    with args.csv.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The benchmark CSV has no measurements yet")
    if len({(row["M"], row["N"], row["K"]) for row in rows}) != 1:
        raise ValueError("Plot one matrix shape at a time")
    for row in rows:
        row["schedule"] = (
            row["orderer"] if row["variant"] == "forced" else "autoscheduled"
        )
        for name in ("p", "q", "r"):
            row[f"{name}_exp"] = round(
                np.log2(float(row[name]) * int(row["M"]) * int(row["N"]))
            )
        row["kernel_ms"] = float(row["seconds"]) * 1000
        row["cost_agreement"] = float(row["matches_bruteforce"].lower() == "true")
    output = args.out or args.csv.parent / "graphs"
    output.mkdir(parents=True, exist_ok=True)
    plt.switch_backend("Agg")
    positive = [row["kernel_ms"] for row in rows if row["kernel_ms"] > 0]
    heatmaps(
        rows,
        ORDERS,
        "kernel_ms",
        "SDDMM kernel runtime (ms)",
        output / "sddmm_runtime.png",
        cmap="viridis",
        norm=LogNorm(vmin=min(positive), vmax=max(positive)),
    )
    auto = [row for row in rows if row["schedule"] == "autoscheduled"]
    if auto:
        agreement = np.mean([row["cost_agreement"] for row in auto])
        heatmaps(
            auto,
            ("autoscheduled",),
            "cost_agreement",
            f"Autoscheduler agrees with brute-force cost winner: {agreement:.1%}",
            output / "sddmm_cost_agreement.png",
            cmap="RdYlGn",
            norm=plt.Normalize(0, 1),
        )
        print(f"Cost-model agreement: {agreement:.1%} ({len(auto)} configurations)")


if __name__ == "__main__":
    main()
