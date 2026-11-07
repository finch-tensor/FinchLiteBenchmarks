# Loop-order cost benchmarks

## Add your own benchmark

Copy `example_benchmarks.py` and edit the settings at the top of your copy:
`SIZE`, `DENSITY`, `RUNS`, `WARMUP`, `SEED`, `OUT`, and `DRY_RUN`. Define your
expressions in `build_cases()`, yielding a `BenchmarkCase` for each expression
and input configuration. Use `fl.defer` for inputs and return the expression
before calling `fl.compute`. A minimal standalone benchmark is:

```python
import numpy as np
import finch as fl
from finchlitebench.loopOrderCost import BenchmarkCase
from finchlitebench.loopOrderCost.runner import run

SIZE = 64
RUNS = 20
WARMUP = 1
SEED = 42
OUT = "my_results.csv"
DRY_RUN = False

def build_cases():
    rng = np.random.default_rng(SEED)
    A = fl.defer(fl.asarray(rng.random((SIZE, SIZE))))
    B = fl.defer(fl.asarray(rng.random((SIZE, SIZE))))
    yield BenchmarkCase("matmul", A @ B, {"shape": [SIZE, SIZE]})
    yield BenchmarkCase("row_sum", fl.sum(A * B, axis=1))

if __name__ == "__main__":
    run(build_cases(), out=OUT, runs=RUNS, warmup=WARMUP,
        seed=SEED, dry_run=DRY_RUN)
```

The expression can use Finch's frontend operations, including einsum, elementwise
operations, transposes and reductions, or a tuple of lazy results. Shapes, ranks,
formats and sparsity are chosen in the benchmark file; `SIZE` is just a convenience
setting. Names must be unique. Create inputs inside `build_cases` and keep them
unchanged during measurement. For SciPy CSR inputs, use `int64` for both `indices`
and `indptr`, as shown in `example_benchmarks.py`.

From the repository root, with the source paths set as shown below:

```bash
python -u my_benchmarks.py
```

A ready-to-run example is `src/finchlitebench/loopOrderCost/example_benchmarks.py`.
Run it with `python -u src/finchlitebench/loopOrderCost/example_benchmarks.py`.
Set `DRY_RUN = True` in the file to preview schedule counts. Previewing prepares
inputs and statistics, without starting Julia or compiling benchmark kernels;
it does not create a CSV. No command-line settings are needed.

For a four-clique (K4) example, run:

```bash
python -u src/finchlitebench/loopOrderCost/clique4.py
```

`clique4.py` contracts six independent 64 x 64 sparse edge matrices using
`ab,ac,ad,bc,bd,cd->`. Its four indices give 24 loop orders. Defaults are 20%
density, one warmup and 20 timed executions per order, writing `clique4.csv` in
the current directory. All settings are at the top of that file.

The runner discovers indices from Finch's normalized plan, rather than assuming
a particular einsum or number of matrices. It enumerates every index permutation
for each aggregate query, and all combinations when there are multiple queries.
In-place update queries retain the relative order of their output indices, as
required by Finch. Output layout changes use a temporary followed by a copy.
Index labels (`i`, `j`, etc.) are local to each query; `query_layout` in the CSV
maps them to normalized input tensor axes and output axes. Each query's order
list is shuffled with the fixed seed before traversing their combinations.

The Julia pipeline and uniform cost model match the existing benchmarks. The
runner uses Finch's default factorizer; it enumerates loop orders for that plan,
not alternative ways to split/fuse operations. Expressions must be supported by
Finch's normalizer, cost model and Julia backend. Setup, input statistics,
planning, compilation and explicit garbage collection occur outside timed runs.
Each schedule is planned once, executes once during setup, then receives the
requested extra warmups and timed calls of the bound kernel. Time includes the
Python/Julia call and backend buffer management for the entire plan, including
its copies and intermediate queries.

The CSV records metadata, requested and verified orders, estimated cost, raw
mean (`seconds`), median, filtered mean, retained count, and all individual times.
Filtering only excludes high samples above
`max(3 * median, Q3 + 3 * (Q3 - Q1))` from the separate filtered mean; raw values
stay intact. Spearman correlation of cost versus raw mean runtime is printed per
benchmark, using finite pairs and reporting their count. Constant ranks produce
an undefined (`nan`) correlation. Existing chain plotters still expect the old
chain CSV schema; custom CSVs use benchmark names and JSON metadata instead.

For an SCC job, supply `--benchmark-file` first:

```bash
qsub -V -N my_benchmark -l h_rt=48:00:00 -j y -o "$PWD/my_benchmark.log" \
  src/finchlitebench/loopOrderCost/run.sh --benchmark-file my_benchmarks.py
```

## Existing benchmarks

Run from the `FinchLiteBenchmarks` repository root. With a sibling
`finch-tensor-lite` checkout, set the source paths on Linux:

```bash
export PYTHONPATH="$PWD/../finch-tensor-lite/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -u -m finchlitebench.loopOrderCost.microBenchmarks3 --size 128
```

Available matrix sizes are 128, 500, and 1000. Each run writes
`microBenchmark3_confirm_<size>_20runs.csv` in the current working directory.
The other benchmark modules are `microBenchmark` and `microBenchmark2`.

Submit separate SCC jobs using the prepared `~/finch-env` environment:

```bash
for size in 128 500 1000; do
  qsub -V -N "microBenchmark3_${size}_20runs" -l h_rt=48:00:00 \
    -j y -o "$PWD/microBenchmark3_${size}_20runs.log" \
    src/finchlitebench/loopOrderCost/run.sh --size "$size"
done
```

Filter and plot the CSVs in `src/finchlitebench/benchmarks/results_archibe`:

```bash
python -m finchlitebench.loopOrderCost.filter_chain_results
python -m finchlitebench.loopOrderCost.plot_chain_results
```

Both commands also accept explicit CSV paths. Filtering writes to a `filtered`
subfolder beside the input CSVs; plotting writes beside its input CSVs.
Original CSVs are preserved.
