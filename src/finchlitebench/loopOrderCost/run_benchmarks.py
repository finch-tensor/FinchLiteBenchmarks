import importlib
import json
import subprocess
import sys
import traceback
from pathlib import Path

output = Path.cwd()

schedulers = importlib.import_module("finch.autoschedule.default_schedulers")
# Input statistics use Numba too, avoiding the optional Julia initialization lock.
schedulers.NON_RECURSIVE_SCHEDULER = schedulers.COMPILE_NUMBA
schedulers.COMPILE_NUMBA.ctx.cache = True

status = {"microBenchmark2": "queued", "microBenchmark": "queued"}
for module_name, csv_name, plot_script in (
    ("microBenchmark2", "chain_microbenchmark.csv", "plot_microBenchmark2.py"),
    ("microBenchmark", "sddmm_microbenchmark.csv", "plot_microBenchmark.py"),
):
    benchmark = importlib.import_module(f"finchlitebench.loopOrderCost.{module_name}")
    benchmark.OUT = output / csv_name
    status[module_name] = "running"
    (output / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"Running {module_name}: {benchmark.OUT}", flush=True)
    try:
        benchmark.main()
        status[module_name] = "complete"
    except Exception:  # noqa: BLE001 - Run the other benchmark after a failure.
        traceback.print_exc()
        status[module_name] = "failed"
    if status[module_name] == "complete":
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(plot_script)),
                str(benchmark.OUT),
            ],
            check=False,
        )
        if result.returncode:
            status[f"{module_name}_graphs"] = "failed"
    (output / "run_status.json").write_text(json.dumps(status, indent=2))
    print(f"{module_name}: {status[module_name]}", flush=True)
