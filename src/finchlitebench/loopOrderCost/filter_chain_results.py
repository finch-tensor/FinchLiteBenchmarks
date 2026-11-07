import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_archibe"
OUT = ROOT / "filtered"
SOURCES = tuple(Path(arg).resolve() for arg in sys.argv[1:]) or tuple(
    ROOT / name for name in ("microBenchmark3_64_5runs.csv", "microBenchmark3_128_20runs.csv", "microBenchmark3_500_20runs.csv")
)
if sys.argv[1:]:
    OUT = SOURCES[0].parent / "filtered"


def keep_timings(samples):
    samples = np.asarray(samples, dtype=float)
    q1, q3 = np.percentile(samples, [25, 75])
    cutoff = max(3 * np.median(samples), q3 + 3 * (q3 - q1))
    return samples <= cutoff, float(cutoff)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    removed = []
    for source in SOURCES:
        name = source.name
        original_hash = hashlib.sha256(source.read_bytes()).digest()
        with source.open(newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            fields = reader.fieldnames
            rows = list(reader)
        run_fields = sorted(
            (field for field in fields if field.startswith("run_") and field.endswith("_seconds")),
            key=lambda field: int(field.split("_")[1]),
        )
        affected = 0
        for row in rows:
            samples = np.array([float(row[field]) for field in run_fields])
            assert np.all(np.isfinite(samples)) and np.all(samples > 0)
            keep, cutoff = keep_timings(samples)
            assert keep.sum() >= 3
            old_mean = float(row["seconds"])
            new_mean = float(samples[keep].mean())
            removed_runs = []
            for index in np.flatnonzero(~keep):
                run = int(index) + 1
                removed_runs.append(str(run))
                removed.append({"source": name, "size": row["size"], "f_density": row["f_density"],
                                "loop_order": row["loop_order"], "run": run,
                                "removed_seconds": float(samples[index]), "cutoff_seconds": cutoff,
                                "original_mean_seconds": old_mean, "filtered_mean_seconds": new_mean})
                row[f"run_{run}_seconds"] = ""
            row["seconds"] = new_mean
            row["median_seconds"] = float(np.median(samples[keep]))
            row["retained_runs"] = int(keep.sum())
            row["removed_runs"] = ";".join(removed_runs)
            affected += bool(removed_runs)
        target = OUT / f"{source.stem}_filtered.csv"
        with target.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=[*fields, "retained_runs", "removed_runs"])
            writer.writeheader()
            writer.writerows(rows)
        assert hashlib.sha256(source.read_bytes()).digest() == original_hash
        print(f"{name}: {affected}/{len(rows)} schedules filtered; wrote {target.name}")
    with (OUT / "removed_samples.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source", "size", "f_density", "loop_order", "run",
                                                "removed_seconds", "cutoff_seconds", "original_mean_seconds",
                                                "filtered_mean_seconds"])
        writer.writeheader()
        writer.writerows(removed)
    print(f"Removed {len(removed)} individual samples. Original CSVs preserved.")
