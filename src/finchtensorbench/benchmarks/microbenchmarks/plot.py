
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from finchtensorbench.benchmarks.microbenchmarks.util import feature_matrix

MAX_NORMS = len(feature_matrix.POWERS)
DPI = 300  # for the PNGs
# Each plot is saved as a vector PDF, plus a PNG for quick viewing.
FORMATS = ("pdf", "png")

PLOTS = {
    "sddmm": ("SDDMM", "Mask density"),
    "chain": ("Matrix chain ABC", "C density"),
    "spgemm": ("Sparse matrix multiply", "Active k-indices"),
    "matrix_features": ("Row Lp norms (50k x 50k CSR, 1% dense)", "Norms per row"),
}
COLORS = {
    "Crow's Nest": "#2563eb",
    "Galley": "#ea580c",
    "Scorch": "#0d9488",
    "SparseAuto": "#9333ea",
}
# The stem of each system's result CSV in graph/scripts, as its runner writes it.
RESULT_STEMS = {
    "Crow's Nest": "finch",
    "Galley": "galley",
    "Scorch": "scorch",
    "SparseAuto": "sparseauto",
}
# The names the plots show for the systems their inputs are labeled with.
DISPLAY_NAMES = {"Finch": "Crow's Nest"}
# Plots that leave a system's results out. SparseAuto cannot schedule the row
# norms (powers and roots are not products of tensors), and its SpGEMM schedule
# drops products when a row of A has several nonzeros (it never adds the
# workspace a sparse output needs), so run_sparseauto times TACO's default
# kernels for both; those are not SparseAuto's results.
EXCLUDED = {"SparseAuto": {"spgemm", "matrix_features"}}


def load_results(inputs):
    results = {name: defaultdict(lambda: defaultdict(list)) for name in PLOTS}
    for backend, path in inputs:
        with path.open(newline="") as source:
            for line, row in enumerate(csv.DictReader(source), start=2):
                name = row["benchmark"].strip().lower()
                variant = row.get("implementation", "").strip()
                if variant.lower() == backend.lower():
                    variant = ""
                if name in ("gustavson", "outer", "spgemm"):
                    family = "spgemm"
                    point = float(row["active_k"])
                    if name != "spgemm":
                        variant = variant or name.title()
                elif name.startswith("features_"):
                    family = "matrix_features"
                    point = int(name.rsplit("_", 1)[1])
                elif name in ("matrix_features", "features", "stats"):
                    family = "matrix_features"
                    point = int(row["num_stats"])
                elif name in ("sddmm", "chain"):
                    family = name
                    point = float(row["density"])
                else:
                    raise ValueError(f"{path}:{line}: unknown benchmark {name!r}")
                seconds = float(row["seconds"])
                if not np.isfinite(seconds) or seconds <= 0:
                    raise ValueError(f"{path}:{line}: seconds must be finite and positive")
                if not np.isfinite(point) or point <= 0:
                    raise ValueError(f"{path}:{line}: sweep values must be positive")
                if family == "matrix_features" and point not in range(1, MAX_NORMS + 1):
                    raise ValueError(f"{path}:{line}: num_stats must be between 1 and {MAX_NORMS}")
                shown = DISPLAY_NAMES.get(backend, backend)
                if family in EXCLUDED.get(shown, ()):
                    continue
                results[family][(shown, variant)][point].append(seconds * 1000)
    return results


def draw_plot(axis, family, series, colors):
    title, xlabel = PLOTS[family]
    axis.set(title=title, xlabel=xlabel, ylabel="Runtime (ms)")
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    if not series:
        axis.text(0.5, 0.5, "No results available", transform=axis.transAxes,
                  ha="center", va="center", color="#64748b")
        return

    width = 0.8 / len(series)
    for index, ((backend, variant), points) in enumerate(series.items()):
        x = np.array(sorted(points))
        median = np.array([np.median(points[point]) for point in x])
        label = f"{backend} ({variant})" if variant else backend
        color = colors[backend]
        if family == "matrix_features":
            positions = x - 1 + (index - (len(series) - 1) / 2) * width
            axis.bar(positions, median, width, color=color, label=label, alpha=0.9)
        else:
            # The run without workspaces is dashed, with square markers, and the
            # run with them has diamonds, which show beneath other systems' circles.
            no_workspace = any(word in variant.lower() for word in ("outer", "disabled"))
            marker = "s" if no_workspace else "D" if variant else "o"
            axis.plot(x, median, label=label, color=color, linewidth=2, marker=marker,
                      linestyle="--" if no_workspace else "-", markersize=5)

    if family == "matrix_features":
        counts = sorted({int(point) for points in series.values() for point in points})
        axis.set_xticks([c - 1 for c in counts],
                        [f"{c}\nL1" if c == 1 else f"{c}\n+ L{c}" for c in counts])
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", alpha=0.2)
    else:
        axis.set(xscale="log", yscale="log")
        axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=13)


def plot_results(results, output):
    output.mkdir(parents=True, exist_ok=True)
    backends = list(dict.fromkeys(
        backend for series in results.values() for backend, _ in series
    ))
    palette = plt.get_cmap("tab10")
    colors = {name: COLORS.get(name, palette(index % 10))
              for index, name in enumerate(backends)}
    with plt.rc_context({"font.size": 14, "axes.titlesize": 17,
                         "figure.facecolor": "white", "savefig.facecolor": "white",
                         "pdf.fonttype": 42}):
        overview, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        overview.suptitle("Microbenchmarks", fontsize=22, fontweight="bold")
        for axis, (family, series) in zip(axes.flat, results.items(), strict=True):
            draw_plot(axis, family, series, colors)
            figure, single_axis = plt.subplots(figsize=(8, 5), layout="constrained")
            draw_plot(single_axis, family, series, colors)
            for fmt in FORMATS:
                figure.savefig(output / f"{family}.{fmt}", dpi=DPI)
            plt.close(figure)
        for fmt in FORMATS:
            overview.savefig(output / f"all_benchmarks.{fmt}", dpi=DPI)
        plt.close(overview)


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", action="append", metavar="LABEL=CSV",
                        help="Repeat for each system or result file")
    parser.add_argument("--output", type=Path, default=Path("graph/scripts"))
    args = parser.parse_args(argv)
    if args.input:
        inputs = []
        for item in args.input:
            label, separator, path = item.partition("=")
            if not separator or not label or not path:
                parser.error("--input must have the form LABEL=CSV")
            inputs.append((label, Path(path)))
    else:
        inputs = [(name, Path(f"graph/scripts/{stem}_result.csv"))
                  for name, stem in RESULT_STEMS.items()]
        inputs = [(name, path) for name, path in inputs if path.is_file()]
    if not inputs:
        parser.error("No timing CSVs found; run Finch or provide --input LABEL=CSV")
    results = load_results(inputs)
    if not any(results.values()):
        parser.error("The input CSVs contain no timing rows")
    plot_results(results, args.output)
    print(f"Saved four plots and an overview (PDF and PNG) to {args.output}")


if __name__ == "__main__":
    run()
