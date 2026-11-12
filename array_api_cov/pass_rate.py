#!/usr/bin/env python3
"""Percentage of array-api-tests that pass for each sparse array library.

    python pass_rate.py                          # every library
    python pass_rate.py finch torch.sparse       # some of them
    python pass_rate.py --max-examples 10 finch -- -k test_sum   # extra pytest args
    FINCH_SCHEDULER=COMPILE_NUMBA python pass_rate.py finch      # default INTERPRET_ASSEMBLY
    envs/finch/bin/python plot.py                # redraw the bar plot only

finch runs on its assembly interpreter; finch-galley runs on the scheduler the
benchmarks use, COMPILE_JULIA_GALLEY (Galley's optimizer, Finch.jl kernels).

Each library runs the array-api-tests checkout in array-api-tests/, inside its
own virtualenv in envs/, with every test input in the library's sparse format
(see sparse_xp.py). The denominator is the whole suite, as collected for numpy:
a test counts as passed only if it ran and passed, so failures, errors, skips
(e.g. a missing linalg extension), timeouts and crashes count against the
library, and so do tests that a library's namespace stops from being collected.

Results go to reports/<library>.json (each test's outcome), are printed as a
table and drawn as a bar plot, reports/pass_rate.png (by plot.py; runs with
extra pytest arguments are partial and leave it alone). This file is also the
pytest plugin (-p pass_rate) that records them.

Setup: array-api-tests/ is a clone of github.com/data-apis/array-api-tests
(with submodules). Each env is a Python 3.12 venv with array-api-tests'
requirements.txt, pytest-xdist and pytest-timeout, plus:

    envs/finch    finch-tensor[julia] (scipy comes with it) and matplotlib. Finch.jl
                  installs itself into julia-depot/ on first use (Julia 1.10-1.12
                  on PATH, or juliapkg downloads one)
    envs/sparse   sparse
    envs/torch    torch, array-api-compat, ninja, and scorch installed editable:
                  pip install --no-build-isolation -e libs/scorch
                  (scorch finds its C++ sources by walking up to its setup.py)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "array-api-tests"
REPORTS = ROOT / "reports"

# Environment for scorch's JIT-compiled kernels, as in the repo's pixi scorch env.
SCORCH_JIT = {
    "CC": "/usr/bin/clang",
    "CXX": "/usr/bin/clang++",
    "SDKROOT": "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk",
}
# Finch.jl and its packages live in this project, not in ~/.julia (the trailing
# separator keeps Julia's bundled depots).
FINCH = {"SPARSE_XP": "finch", "JULIA_DEPOT_PATH": f"{ROOT / 'julia-depot'}{os.pathsep}"}
# name: (virtualenv in envs/, ARRAY_API_TESTS_MODULE, extra environment)
LIBRARIES = {
    "finch": ("finch", "sparse_xp", FINCH),
    "finch-galley": ("finch", "sparse_xp", {**FINCH, "FINCH_SCHEDULER": "COMPILE_JULIA_GALLEY"}),
    "scorch": ("torch", "sparse_xp", {"SPARSE_XP": "scorch", **SCORCH_JIT}),
    "torch.sparse": ("torch", "sparse_xp", {"SPARSE_XP": "torch"}),
    "pydata/sparse": ("sparse", "sparse", {}),
    # scipy.sparse has no Array API namespace; it is tested as one, as is.
    "scipy.sparse": ("finch", "scipy.sparse", {}),
}
REFERENCE = ("sparse", "numpy")  # (env, module) whose collection is the denominator

# Tests that never see an array of rank 1 or more, so never a sparse one: names,
# signatures, constants, dtype and device info, and tests that use only 0-d
# arrays (which stay dense), such as the elementwise special cases.
NO_ARRAYS = {
    "test_has_names.py", "test_signatures.py", "test_constants.py",
    "test_inspection_functions.py", "test_data_type_functions.py::test_finfo",
    "test_data_type_functions.py::test_iinfo", "test_data_type_functions.py::test_isdtype",
    "test_array_object.py::test_scalar_casting", "test_special_cases.py::test_unary",
    "test_special_cases.py::test_binary", "test_special_cases.py::test_iop",
}


def on_arrays(test_id):
    path, rest = test_id.split("::", 1)
    file, func = path.rsplit("/", 1)[-1], rest.split("[")[0]
    return file not in NO_ARRAYS and f"{file}::{func}" not in NO_ARRAYS


def score(outcomes, suite):
    """[(passed, total) over every test, (passed, total) over tests on arrays].

    Test ids can differ between libraries (dtype parameters are named by their
    repr), so this compares counts. Tests that were never collected or never
    reported (a crash) stay in the total.
    """
    cells = []
    for keep in (lambda t: True, on_arrays):
        passed = sum(o == "passed" for t, o in outcomes.items() if keep(t))
        cells.append((passed, max(sum(map(keep, suite)), sum(map(keep, outcomes)))))
    return cells


def report_path(name):
    return REPORTS / (name.replace("/", "-") + ".json")


def environ(env, module, extra, args):
    e = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    e.update(
        ARRAY_API_TESTS_MODULE=module,
        ARRAY_API_TESTS_VERSION=args.api_version,
        PATH=f"{ROOT / 'envs' / env / 'bin'}{os.pathsep}{e.get('PATH', '')}",
        PYTHONPATH=str(ROOT),
        **extra,
    )
    return e


def pytest(env, module, extra, args, *pytest_args, **kwargs):
    cmd = [ROOT / "envs" / env / "bin" / "python", "-m", "pytest", "array_api_tests",
           "-p", "no:cacheprovider", *pytest_args, *args.pytest_args]
    return subprocess.run(cmd, cwd=TESTS, env=environ(env, module, extra, args), **kwargs)


def collect(args):
    env, module = REFERENCE
    out = pytest(env, module, {}, args, "--collect-only", "-q",
                 capture_output=True, text=True).stdout
    return {line for line in out.splitlines() if "::" in line}


def run(name, args):
    env, module, extra = LIBRARIES[name]
    out = report_path(name)
    out.unlink(missing_ok=True)
    log = out.with_suffix(".log")
    plugins = ["-p", "pass_rate"] + (["-p", "sparse_xp"] if module == "sparse_xp" else [])
    with open(log, "w") as f:
        pytest(
            env, module, {**extra, "PASS_RATE_OUT": str(out)}, args,
            *plugins, "-q", "--disable-warnings",
            f"--max-examples={args.max_examples}", "--derandomize", "--disable-deadline",
            # A test that hangs is killed (with its worker, which xdist replaces).
            f"--timeout={args.timeout}", "--timeout-method=thread",
            "-n", str(args.workers), "--max-worker-restart=1000000",
            stdout=f, stderr=subprocess.STDOUT,
        )
    if not out.exists():  # pytest stopped before any test ran (see the log)
        out.write_text("{}")
    return json.loads(out.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("libraries", nargs="*", default=list(LIBRARIES),
                   help=f"any of: {', '.join(LIBRARIES)} (default: all)")
    # Hypothesis tries the simplest inputs (0-d, empty) first, and those stay
    # dense or have nothing to store, so few examples barely test sparse arrays.
    p.add_argument("--max-examples", type=int, default=100,
                   help="hypothesis examples per test, as in array-api-tests "
                   "(default: %(default)s)")
    p.add_argument("--timeout", type=int, default=300,
                   help="seconds before a test is killed (default: %(default)s)")
    p.add_argument("--workers", default="auto", help="pytest-xdist workers")
    p.add_argument("--api-version", default="2024.12", help="(default: %(default)s)")
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    args = p.parse_args(argv[:split])
    args.pytest_args = argv[split + 1:]
    for name in args.libraries:
        if name not in LIBRARIES:
            p.error(f"unknown library {name!r}; choose from {', '.join(LIBRARIES)}")

    REPORTS.mkdir(exist_ok=True)
    suite = collect(args)
    if not suite:
        sys.exit(f"collected no tests with {REFERENCE[1]}; is {TESTS} set up?")
    full = not args.pytest_args  # extra pytest args (-k ...) make a partial run
    if full:
        (REPORTS / "suite.json").write_text(json.dumps(sorted(suite), indent=0))
    n_arrays = sum(on_arrays(t) for t in suite)
    print(f"{len(suite)} tests ({n_arrays} on arrays), Array API {args.api_version}, "
          f"--max-examples={args.max_examples}\n", flush=True)
    print(f"{'library':15} {'passed':>14} {'%':>6}  {'on arrays':>14} {'%':>6}   time",
          flush=True)
    for name in args.libraries:
        start = time.time()
        cells = [f"{passed:6} / {total:<5} {100 * passed / total:5.1f}%"
                 for passed, total in score(run(name, args), suite)]
        print(f"{name:15} {'  '.join(cells)}   {(time.time() - start) / 60:.0f} min",
              flush=True)

    # The plot shows every library with a report, from this run or earlier ones.
    if full:
        subprocess.run([ROOT / "envs" / "finch" / "bin" / "python", ROOT / "plot.py"])


# ------------------------------------------- pytest plugin (-p pass_rate)

_outcomes = {}


def pytest_sessionstart(session):
    # Shrinking a failing example to a minimal one cannot change the outcome,
    # but it runs many more examples, and each can compile new kernels (minutes
    # per test under finch-galley). This runs after array-api-tests' conftest
    # loaded its profile, and before collection applies per-test settings.
    from hypothesis import Phase, settings

    settings.register_profile("pass_rate", settings.default,
                              phases=[Phase.explicit, Phase.reuse, Phase.generate])
    settings.load_profile("pass_rate")


def pytest_runtest_logreport(report):
    # On the xdist controller (or without xdist) every phase of every test
    # arrives here. A test passes if its call passed and no phase failed.
    if report.failed:
        outcome = "failed"
    elif report.skipped:
        outcome = "skipped"
    elif report.when == "call":
        outcome = "passed"
    else:
        return
    if _outcomes.get(report.nodeid) != "failed":
        _outcomes[report.nodeid] = outcome


def pytest_sessionfinish(session):
    if not hasattr(session.config, "workerinput"):  # controller only
        Path(os.environ["PASS_RATE_OUT"]).write_text(json.dumps(_outcomes, indent=0))


if __name__ == "__main__":
    main()
