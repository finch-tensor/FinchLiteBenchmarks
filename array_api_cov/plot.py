#!/usr/bin/env python3
"""Bar plot of the pass rates that pass_rate.py recorded in reports/.

    envs/finch/bin/python plot.py                # writes reports/pass_rate.png

One bar per library with a report, in pass_rate.LIBRARIES order, scored as in
pass_rate.py's table: on the tests that use sparse arrays, and on the whole
suite. pass_rate.py runs this after every run.
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

from pass_rate import LIBRARIES, REPORTS, on_arrays, report_path, score  # noqa: E402

SURFACE, TEXT, MUTED, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", default=REPORTS / "pass_rate.png")
    args = p.parse_args()

    suite = json.loads((REPORTS / "suite.json").read_text())  # written by pass_rate.py
    rows = [(name, score(json.loads(report_path(name).read_text()), suite))
            for name in LIBRARIES if report_path(name).exists()]
    if not rows:
        raise SystemExit(f"no reports in {REPORTS}; run pass_rate.py first")
    names = [name for name, _ in rows][::-1]  # barh draws bottom up

    plt.rcParams.update({"font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT})
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(10, 1.5 + 0.42 * len(rows)),
                             facecolor=SURFACE, layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)
    panels = [("Tests on sparse arrays", 1, sum(map(on_arrays, suite))),
              ("Whole suite", 0, len(suite))]
    for ax, (title, column, n) in zip(axes, panels):
        pct = [100 * cells[column][0] / cells[column][1] for _, cells in rows][::-1]
        ax.barh(names, pct, height=0.45, color=BAR, zorder=2)
        for y, value in enumerate(pct):
            ax.text(value + 1.5, y, f"{value:.1f}%", va="center", clip_on=False)
        ax.set_title(f"{title} ({n})", loc="left", fontsize=10, fontweight="bold")
        ax.set_facecolor(SURFACE)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.xaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.tick_params(length=0, labelcolor=MUTED)
        ax.tick_params(axis="y", labelcolor=TEXT, labelsize=10)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(MUTED)
    fig.suptitle("array-api-tests passed with sparse inputs", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.savefig(args.output, dpi=200, facecolor=SURFACE)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
