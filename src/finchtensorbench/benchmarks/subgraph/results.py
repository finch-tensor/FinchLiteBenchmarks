"""Helpers shared by the subgraph-matching runners, plots and comparisons."""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

# A query finished with a correct count (or one with no ground truth to check).
DONE = ("ok", "unchecked")
STATUSES = ("ok", "unchecked", "wrong", "timeout", "oom", "error", "crash")
# What a --plan-only run can record.
PLAN_STATUSES = ("planned", "timeout", "oom", "error", "crash")


def read_rows(path: str | Path) -> list[dict]:
    """The rows of a results CSV, as written by either runner."""
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def geomean(values) -> float:
    """Geometric mean of the positive values (numbers or numeric strings);
    missing, empty and non-positive values are skipped."""
    xs = [float(v) for v in values if v not in (None, "") and float(v) > 0]
    return math.exp(statistics.fmean(map(math.log, xs))) if xs else float("nan")


def print_summary(rows: list[dict]) -> None:
    """Status counts and geometric-mean times of finished queries, per config."""
    planned = any(r["status"] == "planned" for r in rows)
    statuses = PLAN_STATUSES if planned else STATUSES
    finished = ("planned",) if planned else DONE
    header = "".join(f"{s:>10}" for s in statuses)
    print(f"\n{'config':38}{header}  geomean exec  geomean plan  (finished)")
    for config in dict.fromkeys(r["config"] for r in rows):
        mine = [r for r in rows if r["config"] == config]
        done = [r for r in mine if r["status"] in finished]
        counts = "".join(f"{sum(r['status'] == s for r in mine):10}" for s in statuses)
        print(
            f"{config:38}{counts}  "
            f"{geomean(r.get('exec_median_s') for r in done):11.4g}s  "
            f"{geomean(r.get('plan_total_s') for r in done):11.4g}s"
        )
