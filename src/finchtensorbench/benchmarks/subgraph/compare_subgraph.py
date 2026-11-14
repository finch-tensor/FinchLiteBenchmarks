"""Compare systems on subgraph-matching results CSVs, query by query.

    pixi run python -m finchtensorbench.benchmarks.subgraph.compare_subgraph \\
        graph/subgraph_matching/dblp_<stamp>.csv graph/subgraph_matching/dblp_galleyjl_<stamp>.csv \\
        [--baseline galley.jl] [--metric exec_median_s] [--penalty 600]

A system is a scheduler config (``galley.jl`` for Galley.jl runs).  The
baseline defaults to the config of the last CSV given.  With no CSVs, the
newest finch and Galley.jl runs of ``--dataset`` in ``--dir`` are compared; a
run may still be going, since CSVs are written row by row.

For each dataset and family, and each system against the baseline:

* ``paired``: queries both finished with a correct (or unchecked) count; the
  geometric mean of system/baseline, a 95% bootstrap interval for it, the median
  ratio, and how many queries the system is faster on.  Below 1 means the
  system is faster.
* ``penalized``: every query of the family, a failure (timeout, oom, crash,
  error, wrong) costing ``--penalty`` seconds, so a system is not flattered by
  dropping the queries it fails (PAR-k when the penalty is k x the timeout).
* the status counts of both.

A query missing from one system's CSV entirely (it was never run) is left out
of both comparisons, and counted as ``missing``.
"""

from __future__ import annotations

import argparse
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from finchtensorbench.benchmarks.subgraph.results import DONE, geomean, read_rows


def load(paths):
    """{system: {(dataset, query): row}} from the CSVs; a system is a config."""
    rows: dict[str, dict] = defaultdict(dict)
    for path in paths:
        for row in read_rows(path):
            dataset = row["dataset"].removesuffix("_ext")
            rows[row["config"]][(dataset, row["query"])] = row
    return rows


def newest(directory: Path, pattern: str) -> Path:
    """The newest CSV in *directory* whose stem fully matches *pattern*."""
    runs = [p for p in directory.glob("*.csv") if re.fullmatch(pattern, p.stem)]
    if not runs:
        raise SystemExit(f"no CSV matching {pattern} in {directory}")
    return max(runs, key=lambda p: p.stem[-15:])


def value(row, metric):
    if row["status"] in DONE and row.get(metric):
        return float(row[metric])
    return None


def bootstrap(ratios, reps=2000, seed=0):
    """95% percentile interval of the geometric mean of *ratios*."""
    if len(ratios) < 2:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    logs = [math.log(r) for r in ratios]
    means = sorted(
        statistics.fmean(rng.choices(logs, k=len(logs))) for _ in range(reps)
    )
    return math.exp(means[int(0.025 * reps)]), math.exp(means[int(0.975 * reps)])


def compare(system, baseline, metric, penalty):
    groups = defaultdict(list)
    for key in system.keys() | baseline.keys():
        a, b = system.get(key), baseline.get(key)
        family = (a or b)["family"]
        groups[(key[0], family)].append((key, a, b))
    for (dataset, family), items in sorted(groups.items()):
        present = [(a, b) for _, a, b in items if a is not None and b is not None]
        paired = [
            (value(a, metric), value(b, metric))
            for a, b in present
            if value(a, metric) and value(b, metric)
        ]
        ratios = [x / y for x, y in paired]
        lo, hi = bootstrap(ratios)

        def cost(row):
            v = value(row, metric)
            return v if v is not None else penalty

        penalized = [cost(a) / cost(b) for a, b in present]
        yield {
            "dataset": dataset,
            "family": family,
            "n_paired": len(ratios),
            "paired_geomean": geomean(ratios),
            "ci_low": lo,
            "ci_high": hi,
            "median_ratio": statistics.median(ratios) if ratios else float("nan"),
            "faster": sum(r < 1 for r in ratios),
            "n_penalized": len(penalized),
            "penalized_geomean": geomean(penalized),
            "missing": len(items) - len(present),
            "status": dict(Counter(a["status"] for a, _ in present)),
            "baseline_status": dict(Counter(b["status"] for _, b in present)),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path, nargs="*")
    parser.add_argument("--dataset", help="with no CSVs: compare this dataset's newest runs")
    parser.add_argument("--dir", type=Path, default=Path("graph/subgraph_matching"))
    parser.add_argument(
        "--baseline", help="config to compare against (default: the last CSV's)"
    )
    parser.add_argument("--systems", nargs="+", help="configs to compare (default: all)")
    parser.add_argument("--metric", default="exec_median_s")
    parser.add_argument(
        "--penalty",
        type=float,
        default=600.0,
        help="seconds a failed query costs in the penalized comparison",
    )
    parser.add_argument("--out", type=Path, help="also write the table as CSV")
    args = parser.parse_args(argv)

    if not args.csv:
        if not args.dataset:
            parser.error("give CSVs, or --dataset to compare its newest runs")
        stamp = r"_\d{8}_\d{6}"
        args.csv = [newest(args.dir, f"{args.dataset}{stamp}"),
                    newest(args.dir, f"{args.dataset}_galleyjl{stamp}")]  # fmt: skip
        print("comparing " + " and ".join(p.name for p in args.csv))
    rows = load(args.csv)
    if args.baseline is None:
        args.baseline = read_rows(args.csv[-1])[0]["config"]
    if args.baseline not in rows:
        parser.error(f"no rows for baseline {args.baseline!r}; have {sorted(rows)}")
    systems = args.systems or [s for s in rows if s != args.baseline]
    table = []
    print(f"{args.metric}: system / {args.baseline} (below 1: system is faster)")
    print(
        f"{'system':32} {'dataset/family':22} {'n':>4} {'paired':>7} "
        f"{'95% CI':>15} {'median':>7} {'faster':>7} {'penal.':>7}  statuses"
    )
    for name in systems:
        for r in compare(rows[name], rows[args.baseline], args.metric, args.penalty):
            table.append({"system": name, "baseline": args.baseline, **r})
            print(
                f"{name:32} {r['dataset'] + '/' + r['family']:22} {r['n_paired']:>4} "
                f"{r['paired_geomean']:>7.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                f"{r['median_ratio']:>7.3f} {r['faster']:>4}/{r['n_paired']:<3}"
                f"{r['penalized_geomean']:>7.3f}  {r['status']} vs {r['baseline_status']}"
                + (f"  missing {r['missing']}" if r["missing"] else "")
            )
    if args.out:
        with args.out.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
