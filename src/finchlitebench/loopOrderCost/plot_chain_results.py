import csv
import itertools
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_archibe"
OUT = ROOT / "filtered"
SOURCES = tuple(Path(arg).resolve() for arg in sys.argv[1:]) or (
    OUT / "microBenchmark3_64_5runs_filtered.csv",
    OUT / "microBenchmark3_128_20runs_filtered.csv",
    OUT / "microBenchmark3_500_20runs_filtered.csv",
)
if sys.argv[1:]:
    OUT = SOURCES[0].parent
BLUE, ORANGE = "#2563a6", "#d05d16"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False})
expected_orders = {"".join(order) for order in itertools.permutations("ijklm")}
summaries = []

for source in SOURCES:
    with source.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    run_fields = sorted(
        (field for field in rows[0] if field.startswith("run_") and field.endswith("_seconds")),
        key=lambda field: int(field.split("_")[1]),
    )
    groups = defaultdict(list)
    for row in rows:
        assert row["executed_order"] == row["loop_order"]
        raw = [float(row[field]) for field in run_fields if row[field]]
        if "retained_runs" in row:
            assert len(raw) == int(row["retained_runs"])
        assert np.isclose(float(row["seconds"]), np.mean(raw), rtol=1e-9, atol=1e-12)
        assert np.isclose(float(row["median_seconds"]), np.median(raw), rtol=1e-9, atol=1e-12)
        groups[float(row["f_density"])].append(row)
    sizes = {int(row["size"]) for row in rows}
    assert len(sizes) == 1
    size = sizes.pop()
    assert sorted(groups) == [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    for group in groups.values():
        orders = {row["loop_order"] for row in group}
        assert 0 < len(group) == len(orders) and orders <= expected_orders
    complete = len(rows) == len(groups) * len(expected_orders)
    rank_limit = max(len(group) for group in groups.values())
    all_costs = np.array([float(row["est_cost"]) for row in rows])
    all_ms = np.array([float(row["seconds"]) * 1000 for row in rows])
    assert np.all(np.isfinite(all_costs)) and np.all(all_costs > 0)
    assert np.all(np.isfinite(all_ms)) and np.all(all_ms > 0)
    cost_limits = (all_costs.min() / 1.5, all_costs.max() * 1.5)
    time_limits = (all_ms.min() / 1.5, all_ms.max() * 1.5)

    for kind in ("lines", "scatter"):
        fig, axes = plt.subplots(3, 3, figsize=(17.5, 12.5))
        fig.subplots_adjust(left=0.06, right=0.945, bottom=0.105, top=0.865,
                            wspace=0.58 if kind == "lines" else 0.28, hspace=0.65)
        title = "Estimated cost and mean execution time" if kind == "lines" else "Estimated cost versus mean execution time"
        if not complete:
            title += f" (partial: {len(rows)}/{len(groups) * len(expected_orders)} schedules)"
        fig.suptitle(f"{size} × {size}: {title}", fontsize=20, y=0.978)
        handles = []
        for ax, (density, group) in zip(axes.flat, sorted(groups.items())):
            group = sorted(group, key=lambda row: (float(row["est_cost"]), row["loop_order"]))
            costs = np.array([float(row["est_cost"]) for row in group])
            ms = np.array([float(row["seconds"]) * 1000 for row in group])
            rho, p_value = spearmanr(costs, ms)
            assert np.isclose(rho, np.corrcoef(rankdata(costs), rankdata(ms))[0, 1])
            ax.set_ylim(time_limits if kind == "scatter" else cost_limits)
            ax.set_yscale("log")
            if kind == "lines":
                right = ax.twinx()
                right.set_yscale("log")
                right.set_ylim(time_limits)
                x = np.arange(1, len(group) + 1)
                handles = [
                    ax.plot(x, costs, color=BLUE, linewidth=1.9, label="Estimated cost (left axis)")[0],
                    right.plot(x, ms, color=ORANGE, linewidth=1.25, label="Mean execution time (right axis)")[0],
                ]
                ax.set_xlim(1, max(2, rank_limit))
                ax.set_xticks([1, 30, 60, 90, 120] if complete else np.unique(np.linspace(1, rank_limit, 5, dtype=int)))
                ax.set_xlabel("Schedule rank by estimated cost" if complete else "Measured schedule rank by estimated cost")
                ax.set_ylabel("Estimated cost", color=BLUE)
                right.set_ylabel("Mean execution time (ms)", color=ORANGE)
                ax.tick_params(axis="y", colors=BLUE)
                right.tick_params(axis="y", colors=ORANGE)
                panel_title = f"F density {density:.0%}  ({int(group[0]['nnz_F']):,} nonzeros)"
                if not complete:
                    panel_title += f"\nn = {len(group)} measured orders"
                ax.set_title(panel_title, fontsize=12, pad=11)
            else:
                ax.scatter(costs, ms, color=BLUE, s=24, alpha=0.67, edgecolors="white", linewidths=0.25)
                ax.set_xscale("log")
                ax.set_xlim(cost_limits)
                ax.set_xlabel("Estimated cost")
                ax.set_ylabel("Mean execution time (ms)")
                ax.set_title(f"F density {density:.0%}\nSpearman ρ = {rho:.3f}; n = {len(group)}", fontsize=12, pad=10)
                summaries.append({"source": source.name, "size": size, "f_density": density,
                                  "orders": len(group), "spearman_rho": float(rho), "p_value": float(p_value)})
                print(f"{source.name} F={density:.0%}: rho={rho:.6f}; n={len(group)}")
            ax.grid(axis="y" if kind == "lines" else "both", alpha=0.2)
            ax.set_axisbelow(True)
        for ax in axes.flat[len(groups):]:
            ax.set_visible(False)
        if kind == "lines":
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.942), ncol=2, frameon=False)
        for suffix in ("png", "svg"):
            fig.savefig(OUT / f"{source.stem}_{kind}.{suffix}", dpi=180, facecolor="white")
        plt.close(fig)

with (OUT / "spearman_summary.csv").open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
    writer.writeheader()
    writer.writerows(summaries)

fig, ax = plt.subplots(figsize=(10, 5.7), layout="constrained")
for size, (color, marker) in zip(
    sorted({row["size"] for row in summaries}),
    itertools.cycle(zip((BLUE, ORANGE, "#15806b"), ("o", "s", "^"))),
):
    points = [row for row in summaries if row["size"] == size]
    ax.plot([row["f_density"] * 100 for row in points], [row["spearman_rho"] for row in points],
            color=color, marker=marker, linewidth=2, markersize=6, label=f"{size} × {size}")
summary_title = "Cost versus mean execution time: Spearman correlation"
if any(row["orders"] < len(expected_orders) for row in summaries):
    summary_title += "\nIncludes partial results"
ax.set_title(summary_title, fontsize=16, pad=16)
ax.set_xlabel("F density (%)")
ax.set_ylabel("Spearman ρ across measured loop orders")
ax.set_xticks([1, 10, 20, 40, 60, 80, 100])
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.2)
ax.legend(title="Matrix size", frameon=False)
for suffix in ("png", "svg"):
    fig.savefig(OUT / f"spearman_by_density.{suffix}", dpi=180, facecolor="white")
plt.close(fig)
print(f"Saved graphs and correlation summary in {OUT}")
