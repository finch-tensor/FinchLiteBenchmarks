# FinchLiteBenchmarks

Benchmarks for [finch-tensor](https://github.com/finch-tensor/finch-tensor):
subgraph-matching (counting) queries on the G-CARE and Sun & Luo graphs, and
microbenchmarks (SpGEMM, SDDMM, chain products, ...) against Galley.jl, Scorch
and SparseAuto.

## Setup

You need `git` and [pixi](https://pixi.sh) (`curl -fsSL https://pixi.sh/install.sh | sh`).
Clone with the submodules; pixi resolves every environment, and the `global`
one installs Scorch from `external/scorch`, so a clone without them can't
install anything:

```bash
git clone --recurse-submodules https://github.com/finch-tensor/FinchLiteBenchmarks.git
cd FinchLiteBenchmarks
pixi install
```

`pixi.lock` pins every package, including the finch-tensor commit, so all
machines get the same environment. Julia is installed into the environment by
juliacall the first time finch runs, with its packages in `.pixi/julia-depot`;
nothing needs to be on your `PATH`.

## Subgraph matching

Run one dblp query to check the setup:

```bash
pixi run python -m finchtensorbench.benchmarks.subgraph.subgraph_matching \
    --dataset dblp --configs galley --queries dense_4/query_dense_4_2 --runs 1
```

The first run takes several minutes: it downloads Julia, precompiles Finch.jl,
and downloads the dataset. It should end with a line like

```
ok      dense_4/query_dense_4_2 exec 0.01027s  plan 2.738s  first 10.22s  count 37913 (gt 37913)
```

Every count is checked against the ground truth (`gt`).

- **Datasets** download on first use into `data/` (set `GCARE_DATA_DIR` to
  put them elsewhere), from pinned commits with checksums. `yeast`, `dblp` and
  `youtube` are the graphs from Galley.jl's experiments, with its "lite" query
  sample. `dblp_ext` and `youtube_ext` add the larger `dense_8` and `sparse_16`
  query families. The G-CARE graphs (`human`, `aids`, ...) come from the
  G-CARE archives on Google Drive.
- **Configs** are presets from `src/finchtensorbench/benchmarks/autoschedule.py`
  (`default`, `galley`, `galley-cost`, `galley-lp`, `galley-sampling`), or
  `logic/loop/formatter/stats`.
- **Results**: a CSV and each query's plan go in `graph/subgraph_matching/`
  (`--output-dir` to change it).
- **Parallel runs**: `--jobs N` runs N workers, each pinned to its own core
  (`--cpus` to choose them) and capped at `--memory-limit-gb`. Queries that
  exceed `--timeout` are recorded as timeouts. Pinning uses
  `os.sched_setaffinity`, so the runner currently works on Linux only.

`run_subgraph.sh` runs several datasets and configs and plots them:

```bash
src/finchtensorbench/benchmarks/subgraph/run_subgraph.sh -d "dblp youtube" -c galley -j 4
```

### Comparing with Galley.jl

`galley_jl_subgraph` runs Galley.jl on the same queries, so its CSV lines up
with finch's. It needs **Julia 1.11** (its pinned Manifest was resolved with
1.11.7), which pixi does not provide. Install it, for example with
[juliaup](https://github.com/JuliaLang/juliaup) (`juliaup add 1.11`), then:

```bash
pixi run python -m finchtensorbench.benchmarks.subgraph.galley_jl_subgraph \
    --dataset dblp --julia "julia +1.11"
pixi run python -m finchtensorbench.benchmarks.subgraph.compare_subgraph \
    graph/subgraph_matching/*.csv --baseline galley.jl
```

The first run installs and precompiles Galley.jl. `run_subgraph.sh -g` runs
it after finch and compares the two; pass the Julia command through `JULIA`,
e.g. `JULIA="julia +1.11"`.

## Microbenchmarks

Each backend has its own pixi environment and task, which writes a CSV to
`graph/scripts/`:

| Backend | Command | Notes |
|---|---|---|
| finch | `pixi run run-finch` | |
| Galley.jl | `pixi run -e galley run-galley` | through galley-jl-python |
| Scorch | `pixi run -e scorch run-scorch` | builds Scorch from `external/scorch`, with the Command Line Tools' clang on macOS and conda-forge's GCC on Linux |
| SparseAuto | `pixi run -e sparseauto build-sparseauto`, then `pixi run -e sparseauto run-sparseauto` | builds TACO from `external/sparseLNR` |

`pixi run run-all [backends...]` runs them all (or the ones named) and plots
the results; `THREADS=N` sets the thread count (default 1).

## Development

- Tests: `pixi run -e test test`.
- To benchmark another finch-tensor revision, change its `branch` in
  `[tool.pixi.pypi-dependencies]` in `pyproject.toml`, then run
  `pixi update finch-tensor` and commit `pixi.lock`. To run a local checkout
  instead of the installed copy, set `PYTHONPATH` inside the environment,
  since pixi's activation resets it to `src`:
  `pixi run env PYTHONPATH=src:/path/to/finch-tensor/src python -m ...`.
  Each subgraph CSV row records the finch-tensor revision it ran.
- The `global` environment combines every backend, for an IDE.
