"""Compare a subgraph-matching run against earlier runs, by query family.

    pixi run python -m finchtensorbench.benchmarks.compare_subgraph_runs \
        [NEW.csv] --base OLD.csv [--base OTHER.csv ...]

NEW defaults to the newest ``<dataset>_<stamp>.csv`` of ``--dataset`` in
``--dir``; the run may still be going, since its CSV is written row by row.
Each ``--base`` (another finch run, or a Galley.jl run from
``galley_jl_subgraph``) is compared on the queries both finished: the
geometric mean of the median execution and of plan + execution time, the
speedup NEW gives over it, and how many queries got faster or slower by more
than ``--tolerance``. The queries which either run failed are listed.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

DONE = ("ok", "unchecked")


def load(path: Path) -> dict[str, dict]:
    with path.open(newline="") as file:
        return {row["query"]: row for row in csv.DictReader(file)}


def newest(directory: Path, dataset: str) -> Path:
    runs = [
        p
        for p in directory.glob(f"{dataset}_*.csv")
        if re.fullmatch(rf"{dataset}_\d{{8}}_\d{{6}}", p.stem)
    ]
    if not runs:
        raise SystemExit(f"no {dataset}_<stamp>.csv in {directory}")
    return max(runs, key=lambda p: p.stem)


def geo(values: list[float]) -> float:
    values = [max(v, 1e-9) for v in values]
    return math.exp(sum(map(math.log, values)) / len(values)) if values else math.nan


def label(path: Path, rows: dict[str, dict]) -> str:
    config = next(iter(rows.values()), {}).get("config", "?")
    return f"{path.name} [{config}]"


def compare(new: dict[str, dict], base: dict[str, dict], tolerance: float) -> None:
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for query, row in new.items():
        other = base.get(query)
        if other and row["status"] in DONE and other["status"] in DONE:
            groups[row["family"]].append((row, other))
            groups["All"].append((row, other))
    header = (
        f"  {'family':<12}{'n':>5}{'exec new':>11}{'exec base':>11}{'speedup':>9}"
        f"{'faster':>8}{'slower':>8}{'plan+exec new':>15}{'base':>9}"
    )
    print(header)
    for family in sorted(groups, key=lambda f: (f == "All", f)):
        pairs = groups[family]
        new_exec = [float(a["exec_median_s"]) for a, _ in pairs]
        base_exec = [float(b["exec_median_s"]) for _, b in pairs]
        ratios = [b / a for a, b in zip(new_exec, base_exec, strict=True)]
        faster = sum(r > 1 + tolerance for r in ratios)
        slower = sum(r < 1 / (1 + tolerance) for r in ratios)
        new_plan = geo([float(a["plan_total_s"]) for a, _ in pairs])
        base_plan = geo([float(b["plan_total_s"]) for _, b in pairs])
        print(
            f"  {family:<12}{len(pairs):>5}{geo(new_exec):>10.4g}s{geo(base_exec):>10.4g}s"
            f"{geo(ratios):>8.2f}x{faster:>8}{slower:>8}{new_plan:>14.4g}s{base_plan:>8.4g}s"
        )
    worst = sorted(
        (float(a["exec_median_s"]) / float(b["exec_median_s"]), a["query"])
        for a, b in groups["All"]
    )[-3:]
    if worst and worst[-1][0] > 1 + tolerance:
        print(
            "  most slowed down: "
            + ", ".join(f"{q} {r:.2f}x" for r, q in reversed(worst) if r > 1 + tolerance)
        )
    changed = [
        f"{q}: {base[q]['status']} -> {row['status']}"
        for q, row in new.items()
        if q in base and (row["status"] in DONE) != (base[q]["status"] in DONE)
    ]
    if changed:
        print("  finished in one run only: " + "; ".join(changed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("new", type=Path, nargs="?")
    parser.add_argument("--base", type=Path, action="append", default=[])
    parser.add_argument("--dataset", default="youtube")
    parser.add_argument("--dir", type=Path, default=Path("graph/subgraph_matching"))
    parser.add_argument(
        "--tolerance", type=float, default=0.2, help="relative change counted (0.2)"
    )
    args = parser.parse_args(argv)
    path = args.new or newest(args.dir, args.dataset)
    new = load(path)
    status = Counter(row["status"] for row in new.values())
    print(f"{label(path, new)}: {len(new)} queries so far; " + ", ".join(
        f"{n} {s}" for s, n in status.most_common()
    ))
    failed = [f"{q} ({r['status']})" for q, r in new.items() if r["status"] not in DONE]
    if failed:
        print("  failed: " + ", ".join(failed))
    for base_path in args.base:
        base = load(base_path)
        print(f"\nvs {label(base_path, base)}  (speedup = base / new, geomean)")
        compare(new, base, args.tolerance)


if __name__ == "__main__":
    main()
