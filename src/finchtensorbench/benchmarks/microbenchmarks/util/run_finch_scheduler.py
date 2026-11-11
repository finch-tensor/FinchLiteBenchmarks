"""Record scheduler decisions for every Finch benchmark case without timing repeats."""

import argparse
import csv
from pathlib import Path

from finchtensorbench.benchmarks.scripts.run_finch import cases, warm_case


FIELDS = (
    "benchmark", "operation", "density", "shared_k", "num_features",
    "workspace_planning", "logical_optimizer", "logical_search", "loop_search",
    "statistics_model", "pipeline", "selected_loop_orders", "selected_costs",
    "selected_plan", "tensor_formats", "input_statistics", "dataset_metadata",
    "trace_file",
)


def _events(trace, headings):
    return "\n\n".join(
        f"{heading}\n{body}" for heading, body in trace.events if heading in headings
    )


def _stage_body(trace, name):
    return next((body for heading, body in reversed(trace.events)
                 if heading == f"Input to {name}"), "")


def _option(trace, stage, option):
    config = _stage_body(trace, stage).partition("\n\n")[0]
    prefix = option + "="
    return next((line.removeprefix(prefix).strip("'") for line in config.splitlines()
                 if line.startswith(prefix)), "")


def _plan_sections(body):
    # Stage reports put configuration first, followed by plan, formats, and stats.
    content = body.partition("\n\n")[2]
    plan, _, tail = content.partition("\n\nTensor bindings / storage formats:\n")
    formats = tail.partition("\n\nPlanning statistics:\n")[0]
    return plan.partition("\n\nPlanning statistics:\n")[0].strip(), formats.strip()


def decision_row(name, parameter, trace, trace_path):
    """Summarize observed decisions while retaining the full per-case trace."""
    spgemm = name in ("gustavson", "outer")
    features = name.startswith("features_")
    plan, formats = _plan_sections(_stage_body(trace, "LogicCompiler"))
    factorizer = _stage_body(trace, "GalleyLogicFactorizer")
    _, _, statistics = factorizer.partition("\n\nPlanning statistics:\n")
    return {
        "benchmark": name,
        "operation": "spgemm" if spgemm else "matrix_features" if features else name,
        "density": "" if spgemm or features else parameter,
        "shared_k": parameter if spgemm else "",
        "num_features": int(name.rsplit("_", 1)[1]) if features else "",
        "workspace_planning": "enabled" if name == "gustavson" else "disabled" if name == "outer" else "preset",
        "logical_optimizer": _option(trace, "GalleyLogicFactorizer", "optimizer"),
        "logical_search": _events(trace, {"Logical component strategy", "Logical search started"}),
        "loop_search": _events(trace, {"Loop-order search started"}),
        "statistics_model": _option(trace, "LogicExecutor", "stats_factory"),
        "pipeline": " -> ".join(heading.removeprefix("Input to ")
                                 for heading, _ in trace.events if heading.startswith("Input to ")),
        "selected_loop_orders": _events(trace, {"Loop-order search selected"}),
        "selected_costs": _events(trace, {"Logical search selected", "Logical optimization completed",
                                          "Loop-order search selected"}),
        "selected_plan": plan,
        "tensor_formats": formats,
        "input_statistics": statistics.strip(),
        "dataset_metadata": "\n\n".join(
            f"{heading}\n{body}" for heading, body in trace.events
            if "dataset" in heading.lower() or "paper" in heading.lower()
        ),
        "trace_file": str(trace_path),
    }


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("graph/scripts/finch_scheduler_decisions.csv"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_dir = args.output.parent / f"{args.output.stem}_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS)
        writer.writeheader()
        output.flush()
        for count, (name, function, inputs, parameter) in enumerate(cases(), start=1):
            print(f"Scheduling {name}: parameter={parameter}", flush=True)
            _, trace, label = warm_case(name, function, inputs, parameter, capture=True)
            report = trace.report(name, label)
            trace_path = trace_dir / f"{count:02d}_{name}.txt"
            trace_path.write_text(report, encoding="utf-8")
            print(report, end="", flush=True)
            writer.writerow(decision_row(name, parameter, trace, trace_path))
            output.flush()
            print(f"{name}: scheduler decision saved; full trace: {trace_path}", flush=True)
    print(f"Saved {count} scheduler decisions to {args.output}", flush=True)


if __name__ == "__main__":
    run()
