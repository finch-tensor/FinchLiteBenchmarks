"""Box-and-whisker plots of subgraph-matching results CSVs.

    pixi run python -m finchtensorbench.benchmarks.subgraph.plot_subgraph_matching \
        graph/subgraph_matching/human_<stamp>.csv [more CSVs ...] [--out PNG]

One panel per runtime (cached-plan execution, fresh plan + execution, first
call); within a panel, one group per query family plus "All", one box per
scheduler config.  Boxes span the quartiles, whiskers reach 1.5 IQR, and every
query is drawn as a dot.  By default only queries that finished with the
correct count are plotted and the rest are counted in a note printed with
the figure;
``--cap`` draws failures at a time cap instead, and ``--paired`` keeps only
queries every config got right.  Several CSVs (e.g.
finch and Galley.jl on one dataset) are drawn together.  Writes
``<first csv stem>_boxplot.png`` (and a PDF) next to the first CSV unless
``--out`` is given.
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

# The microbenchmark plots' colours (microbenchmarks/plot.py), one per system.
# Several configs of one system (e.g. Crow's Nest with LP and DC statistics)
# take its shades in turn, and are hatched after the first.
COLORS = {"Crow's Nest": ("#2563eb", "#0ea5e9", "#1e3a8a"), "Galley": ("#ea580c",)}
HATCHES = ("", "///", "...", "xxx")
MUTED = "#64748b"
STYLE = {"font.size": 14, "axes.titlesize": 17,
         "figure.facecolor": "white", "savefig.facecolor": "white",
         "pdf.fonttype": 42, "hatch.linewidth": 1.5}
DATASET_NAMES = {"dblp": "DBLP", "youtube": "YouTube", "human": "Human",
                 "aids": "AIDS", "AIDS": "AIDS"}

METRICS = (
    ("exec_median_s", "Execution Time"),
    ("plan_total_s", "Fresh Plan + Execution"),
    ("first_s", "Compilation Time"),
)
METRIC_NOTES = {
    "plan_total_s": "Fresh plan + execution: statistics, optimization and lowering "
                    "with Julia warm.",
    "first_s": "First call: includes Julia compilation of new kernels.",
}


def _seconds(value, _pos=None):
    for scale, unit in ((1, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if value >= scale:
            return f"{value / scale:g} {unit}"
    return f"{value:g} s"


MAX_FAMILY_GROUPS = 8
MAX_WIDTH_IN = 18
# The space between datasets on a shared axis, in groups.
DATASET_GAP = 0.6

# Our scheduler configs are shown as SYSTEM; Galley.jl runs (config "galley.jl")
# as "Galley", as in the microbenchmark plots.
SYSTEM = "Crow's Nest"
JULIA_NAMES = {"galley.jl": "Galley"}
STATS_NAMES = {"dc": "DC", "lp": "LP", "fd": "FD", "vp": "VP", "bu": "BU"}
# The statistics of the presets of autoschedule.PRESETS, which runs record by
# name (kept here so plotting doesn't import finch).
PRESET_STATS = {
    "default": "fd",
    "galley": "dc",
    "galley-cost": "dc",
    "galley-lp": "lp",
    "galley-sampling": "sampling",
}


def display_names(configs):
    """The name each config is shown with: SYSTEM for ours, with the statistics
    it plans with where several of ours are drawn together, or the whole config
    where that still doesn't tell them apart."""
    ours = [c for c in configs if not c.endswith(".jl")]
    names = {c: JULIA_NAMES.get(c, c) for c in configs}
    if len(ours) == 1:
        names[ours[0]] = SYSTEM
    elif ours:
        for c in ours:
            parts = c.split("/")
            stats = parts[3] if len(parts) >= 4 else PRESET_STATS.get(c, c)
            names[c] = f"{SYSTEM} ({STATS_NAMES.get(stats, stats)})"
        if len({names[c] for c in ours}) < len(ours):
            names.update({c: f"{SYSTEM} ({c})" for c in ours})
    return names


def listed(items):
    """"A", "A and B", or "A, B and C"."""
    return " and ".join(filter(None, (", ".join(items[:-1]), items[-1])))


def config_styles(configs, names):
    """Each config's (colour, hatch): the next shade of its system's colour,
    and a hatch for every config of that system after the first."""
    palette = plt.get_cmap("tab10")
    styles, seen = {}, Counter()
    for i, c in enumerate(configs):
        system = SYSTEM if names[c].startswith(SYSTEM) else names[c]
        shades = COLORS.get(system, (palette(i % 10),))
        k = seen[system]
        styles[c] = (shades[k % len(shades)], HATCHES[k % len(HATCHES)])
        seen[system] += 1
    return styles


def plot(paths, out=None, paired=False, cap=None, metrics=None, side_by_side=False):
    with plt.rc_context(STYLE):
        return _plot(paths, out, paired, cap, metrics, side_by_side)


def _plot(paths, out, paired, cap, metrics, side_by_side):
    # The panels to draw, by metric name (all of METRICS by default).
    panels = [m for m in METRICS if metrics is None or m[0] in metrics]
    rows = [row for path in paths for row in read_rows(path)]
    configs = list(dict.fromkeys(r["config"] for r in rows))
    names = display_names(configs)
    styles = config_styles(configs, names)
    dropped = 0
    if paired:
        # Only queries every config finished with a correct count, so every box
        # is over the same queries (a config that fails the hard ones is not
        # flattered by them being left out of its box only).
        # Pairing is within each dataset, over the configs that ran on it, so a
        # config run on only some datasets doesn't empty the rest; a query a
        # config never ran (e.g. one family Galley.jl skipped) is left out too.
        ran = defaultdict(set)  # dataset -> configs
        good = defaultdict(set)  # (dataset, query) -> configs that got it right
        for r in rows:
            ran[r["dataset"]].add(r["config"])
            if r["status"] in DONE:
                good[(r["dataset"], r["query"])].add(r["config"])
        queries = {(r["dataset"], r["query"]) for r in rows}
        common = {key for key, configs in good.items() if configs >= ran[key[0]]}
        dropped = len(queries - common)
        rows = [r for r in rows if (r["dataset"], r["query"]) in common
                or r["status"] not in DONE]
    # Every dataset shares each panel's axes: its families and "All", then the
    # next dataset's, with the dataset's name below its groups.
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    multi = len(datasets) > 1

    def family_order(f):
        # Order families by shape, then size (Chain_3 before Chain_12).
        size = f.partition("_")[2]
        return (f.partition("_")[0], int(size) if size.isdigit() else 0)

    groups_of = {}  # dataset -> its groups, then "All"
    by_shape = {}  # dataset -> whether its families are grouped by shape
    for d in datasets:
        families = sorted(
            {r["family"] for r in rows if r["dataset"] == d}, key=family_order
        )
        # Many families (e.g. AIDS: Chain_3 ... Tree_12) are grouped by shape.
        by_shape[d] = len(families) > MAX_FAMILY_GROUPS
        groups_of[d] = [*dict.fromkeys(
            f.split("_")[0] if by_shape[d] else f for f in families), "All"]

    def group_of(row):
        family = row["family"]
        return family.split("_")[0] if by_shape[row["dataset"]] else family

    ok = defaultdict(list)  # (metric, dataset, config, group) -> values
    failed = Counter()  # (dataset, config, status)
    for r in rows:
        if r["status"] not in DONE:
            failed[(r["dataset"], r["config"], r["status"])] += 1
            if cap is None:
                continue
        for metric, _ in panels:
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
                ok[(metric, r["dataset"], r["config"], group)].append(value)

    def shown(dataset):
        return DATASET_NAMES.get(dataset, dataset)

    width = 0.8 / len(configs)
    # Each (dataset, group)'s x position, with a gap between datasets.
    slots, x = {}, 0.0
    for d in datasets:
        for group in groups_of[d]:
            slots[(d, group)] = x
            x += 1
        x += DATASET_GAP
    if side_by_side:
        # One panel per metric in a row, each wide enough for its group labels.
        fig_width = len(panels) * max(7, 1.0 * len(slots)) + 1
    else:
        fig_width = min(MAX_WIDTH_IN, max(7, 0.9 * len(slots) * len(configs) + 2))
    note = "; ".join(
        f"{f'{shown(d)} ' if multi else ''}{names[c]}: {n} {s}"
        for (d, c, s), n in sorted(failed.items())
    )
    which = (
        f"Failed queries (timeout, oom, crash, error, wrong) are drawn at the "
        f"{_seconds(cap)} cap (dashed line). " if cap is not None
        else f"Paired: only queries every system finished correctly ({dropped} "
        "queries left out). " if paired else "Correct results only. "
    )
    notes = "".join(f" {METRIC_NOTES[m]}" for m, _ in panels if m in METRIC_NOTES)
    footnote = textwrap.fill(
        which + (f"Failures - {note}." if note else "All queries finished correctly.")
        + notes,
        width=100,
    )
    # The panels on top; below them, a strip of their own for the legend, so
    # the constrained layout never overlaps it.
    strip = 0.5  # inches
    panels_height = 5.5 if side_by_side else 1.0 + 3.2 * len(panels)
    fig = plt.figure(figsize=(fig_width, panels_height + strip), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=(panels_height, strip))
    if side_by_side:
        axes = top.subplots(1, len(panels), squeeze=False)[0, :]
    else:
        axes = top.subplots(len(panels), 1, squeeze=False)[:, 0]
    counts = Counter(
        (r["dataset"], group_of(r)) for r in rows
        if r["config"] == configs[0]
        and (cap is not None or not paired or r["status"] in DONE)
    )
    for d in datasets:
        counts[(d, "All")] = sum(n for (e, _), n in counts.items() if e == d)
    rng = np.random.default_rng(0)
    for ax, (metric, title) in zip(axes, panels, strict=True):
        for c, config in enumerate(configs):
            color, hatch = styles[config]
            for (dataset, group), center in slots.items():
                values = ok.get((metric, dataset, config, group), [])
                if not values:
                    continue
                x = center + (c - (len(configs) - 1) / 2) * width
                ax.boxplot(
                    values, positions=[x], widths=width * 0.8, whis=1.5,
                    showfliers=False, patch_artist=True,
                    boxprops=dict(facecolor=(*matplotlib.colors.to_rgb(color), 0.2),
                                  edgecolor=color, lw=1.5, hatch=hatch),
                    medianprops=dict(color=color, lw=2),
                    whiskerprops=dict(color=color, lw=1.2),
                    capprops=dict(color=color, lw=1.2),
                )  # fmt: skip
                jitter = rng.uniform(-width * 0.25, width * 0.25, len(values))
                ax.scatter(x + jitter, values, s=9, color=color, alpha=0.55,
                           linewidths=0, zorder=3)  # fmt: skip
        ax.set_yscale("log")
        if cap is not None:
            ax.axhline(cap, color=MUTED, lw=0.8, ls="--", zorder=1)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=20))
        ax.yaxis.set_major_formatter(FuncFormatter(_seconds))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xticks(list(slots.values()),
                      [f"{g}\nn={counts[(d, g)]}" for d, g in slots], fontsize=12)
        ax.set_xlim(-0.6, max(slots.values()) + 0.6)
        for i, d in enumerate(datasets):
            # A light line before the dataset's "All", a dark one between datasets.
            ax.axvline(slots[(d, "All")] - 0.5, color="#cbd5e1", lw=1)
            if i:
                ax.axvline(slots[(d, groups_of[d][0])] - (1 + DATASET_GAP) / 2,
                           color="#94a3b8", lw=1.5)  # fmt: skip
        if multi:
            # Each dataset's name, centred below its groups.
            names_axis = ax.secondary_xaxis("bottom")
            names_axis.set_xticks(
                [np.mean([slots[(d, g)] for g in groups_of[d]]) for d in datasets],
                [shown(d) for d in datasets], fontsize=16, fontweight="bold",
            )
            names_axis.tick_params(length=0, pad=40)
            names_axis.spines["bottom"].set_visible(False)
        ax.set_ylabel("Runtime")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        plt.Rectangle((0, 0), 1, 1, edgecolor=styles[c][0], lw=1.5,
                      facecolor=(*matplotlib.colors.to_rgb(styles[c][0]), 0.2),
                      hatch=styles[c][1])  # fmt: skip
        for c in configs
    ]
    bottom.legend(handles, [names[c] for c in configs], loc="center",
                  ncol=len(configs), frameon=False, fontsize=15)  # fmt: skip
    fig.suptitle(
        f"Subgraph Matching: {listed([shown(d) for d in datasets])}",
        fontsize=22, fontweight="bold",
    )
    if out is None:
        out = Path(paths[0]).with_name(Path(paths[0]).stem + "_boxplot.png")
    # A vector PDF beside the PNG (or whatever --out names).
    out = Path(out)
    for path in dict.fromkeys((out, out.with_suffix(".pdf"))):
        fig.savefig(path, dpi=300)
        print(f"wrote {path}")
    plt.close(fig)
    print(footnote)
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
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=[m for m, _ in METRICS],
        help="draw only these panels (default: all)",
    )
    parser.add_argument(
        "--side-by-side",
        action="store_true",
        help="put the panels in one row instead of stacking them",
    )
    args = parser.parse_args(argv)
    plot(args.csv, args.out, paired=args.paired, cap=args.cap, metrics=args.metrics,
         side_by_side=args.side_by_side)


if __name__ == "__main__":
    main()
