"""Plot every benchmark summary currently in the results folder."""

from pathlib import Path

if __package__:
    from . import plot_results
else:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from plotting import plot_results


RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


if __name__ == "__main__":
    summaries = sorted(RESULTS_DIR.glob("*_summary.csv"))
    if not summaries:
        raise FileNotFoundError(f"no summary CSV files in {RESULTS_DIR}")

    print(f"plotting {len(summaries)} summaries from {RESULTS_DIR}")
    for summary in summaries:
        plot_results("", RESULTS_DIR, [str(summary)])
