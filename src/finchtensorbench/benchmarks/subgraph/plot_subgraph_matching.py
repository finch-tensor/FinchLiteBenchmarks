"""Box-and-whisker plots of subgraph-matching results CSVs.

    pixi run python -m finchtensorbench.benchmarks.subgraph.plot_subgraph_matching \
        graph/subgraph_matching/human_<stamp>.csv [more CSVs ...] [--out PNG]

One panel per runtime (cached-plan execution, fresh plan + execution, first
call); within a panel, one group per query family plus "All", one box per
scheduler config.  Boxes span the quartiles, whiskers reach 1.5 IQR, and every
query is drawn as a dot.  By default only queries that finished with the
correct count are plotted and the rest are counted in the panel's footnote;
``--cap`` draws failures at a time cap instead, and ``--paired`` keeps only
queries every config got right.  Several CSVs (e.g.
finch and Galley.jl on one dataset) are drawn together.  Writes
``<first csv stem>_boxplot.png`` next to the first CSV unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import (  # noqa: E402
    FuncFormatter,
    LogLocator,
    NullFormatter,
)

from finchtensorbench.benchmarks.subgraph.results import DONE, read_rows  # noqa: E402

# Categorical palette, one colour per config in order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

METRICS = (
    ("exec_median_s", "Execution with a cached plan (median of runs)"),
    ("plan_total_s", "Fresh plan + execution (stats, optimize, lower; warm Julia)"),
    ("first_s", "First call (cold: includes Julia compilation of new kernels)"),
)


def _seconds(value, _pos=None):
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if value >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} s"


MAX_FAMILY_GROUPS = 8
MAX_WIDTH_IN = 18


def _family_key(family):
    shape, _, size = family.partition("_")
    return (shape, int(size) if size.isdigit() else 0)


def plot(paths, out=None, paired=False, palette=None, cap=None):
    rows = [row for path in paths for row in read_rows(path)]
    configs = list(dict.fromkeys(r["config"] for r in rows))
    series = palette or SERIES
    dropped = 0
    if paired:
        # Only queries every config finished with a correct count, so every box
        # is over the same queries (a config that fails the hard ones is not
        # flattered by them being left out of its box only).
        # Pairing is within each (dataset, family), over the configs that ran
        # there, so a config run on only some datasets doesn't empty the rest.
        ran = defaultdict(set)  # (dataset, family) -> configs
        good = defaultdict(set)  # (dataset, query) -> configs that got it right
        for r in rows:
            ran[(r["dataset"], r["family"])].add(r["config"])
            if r["status"] in DONE:
                good[(r["dataset"], r["query"])].add(r["config"])
        family = {(r["dataset"], r["query"]): r["family"] for r in rows}
        common = {
            key for key, configs in good.items()
            if configs >= ran[(key[0], family[key])]
        }
        dropped = len(set(family) - common)
        rows = [r for r in rows if (r["dataset"], r["query"]) in common
                or r["status"] not in DONE]
    families = sorted({r["family"] for r in rows}, key=_family_key)
    # Many families (e.g. AIDS: Chain_3 ... Tree_12) are grouped by shape.
    by_shape = len(families) > MAX_FAMILY_GROUPS

    def group_of(row):
        return row["family"].split("_")[0] if by_shape else row["family"]

    groups = [*dict.fromkeys(group_of({"family": f}) for f in families), "All"]
    ok = defaultdict(list)  # (metric, config, group) -> values
    failed = Counter()  # (config, status)
    for r in rows:
        if r["status"] not in DONE:
            failed[(r["config"], r["status"])] += 1
            if cap is None:
                continue
        for metric, _ in METRICS:
            if r["status"] in DONE and r.get(metric, "") != "":
                value = float(r[metric])
            elif cap is not None:
                # A failed query (or a missing time) counts as the time cap,
                # so every config's box covers every query it was given.
                value = cap
            else:
                continue
            if cap is not None:
                value = min(value, cap)
            for group in (group_of(r), "All"):
                ok[(metric, r["config"], group)].append(value)

    width = 0.8 / len(configs)
    fig, axes = plt.subplots(
        len(METRICS), 1,
        figsize=(min(MAX_WIDTH_IN, max(7, 1.0 * len(groups) * len(configs) + 2)), 11),
        facecolor=SURFACE,
    )  # fmt: skip
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        ax.set_facecolor(SURFACE)
        for c, config in enumerate(configs):
            color = series[c % len(series)]
            for g, group in enumerate(groups):
                values = ok.get((metric, config, group), [])
                if not values:
                    continue
                x = g + (c - (len(configs) - 1) / 2) * width
                ax.boxplot(
                    values, positions=[x], widths=width * 0.8, whis=1.5,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=color + "33", edgecolor=color, lw=1.2),
                    medianprops=dict(color=color, lw=2),
                    whiskerprops=dict(color=color, lw=1),
                    capprops=dict(color=color, lw=1),
                )  # fmt: skip
                jitter = rng.uniform(-width * 0.25, width * 0.25, len(values))
                ax.scatter(x + jitter, values, s=9, color=color, alpha=0.55,
                           linewidths=0, zorder=3)  # fmt: skip
        ax.set_yscale("log")
        if cap is not None:
            ax.axhline(cap, color=TEXT_2, lw=0.8, ls="--", zorder=1)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
        ax.yaxis.set_major_formatter(FuncFormatter(_seconds))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(range(len(groups)))
        counts = Counter(
            group_of(r) for r in rows if r["config"] == configs[0]
            and (cap is not None or not paired or r["status"] in DONE)
        )
        counts["All"] = sum(counts.values())
        ax.set_xticklabels(
            [f"{g}\nn={counts[g]}" for g in groups], fontsize=8, color=TEXT_2
        )
        ax.axvline(len(groups) - 1.5, color=GRID, lw=1)
        ax.set_ylabel("time (log scale)", color=TEXT_2, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=TEXT)
        ax.grid(True, axis="y", which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_2, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=series[i % len(series)] + "33",
                      edgecolor=series[i % len(series)])  # fmt: skip
        for i in range(len(configs))
    ]
    fig.legend(handles, configs, loc="upper right", ncol=len(configs), frameon=False,
               fontsize=9, labelcolor=TEXT)  # fmt: skip
    dataset = rows[0]["dataset"] if rows else ""
    fig.suptitle(f"{' vs '.join(configs)} on {dataset}: runtime per query",
                 x=0.01, ha="left", fontsize=12, color=TEXT)  # fmt: skip
    note = "; ".join(f"{c}: {n} {s}" for (c, s), n in sorted(failed.items()))
    which = (
        f"Failed queries (timeout, oom, crash, error, wrong) are drawn at the "
        f"{_seconds(cap)} cap (dashed line). " if cap is not None
        else f"Paired: only queries every system finished correctly ({dropped} "
        "queries left out). " if paired else "Correct results only. "
    )
    footnote = textwrap.fill(
        which + (f"Failures - {note}." if note else "All queries finished correctly."),
        width=int(fig.get_figwidth() * 17),
    )
    lines = footnote.count("\n") + 1
    fig.text(0.01, 0.005, footnote, fontsize=8, color=TEXT_2, va="bottom")
    fig.tight_layout(rect=(0, 0.01 + 0.012 * lines, 1, 0.97))
    if out is None:
        out = Path(paths[0]).with_name(Path(paths[0]).stem + "_boxplot.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="only plot queries every config finished correctly",
    )
    parser.add_argument(
        "--cap",
        type=float,
        help="draw failed queries at this many seconds (e.g. the timeout) "
        "instead of leaving them out; times above it are clipped to it",
    )
    args = parser.parse_args(argv)
    plot(args.csv, args.out, paired=args.paired, cap=args.cap)


if __name__ == "__main__":
    main()
