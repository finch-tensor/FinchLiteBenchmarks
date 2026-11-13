# Software versions behind the results

These are the versions of each system that produced the results and figures in
`graph/`, as of 2026-11-13. They come from `pixi.lock`, the installed packages'
metadata and the run logs. Crow's Nest is `finch-tensor` (this repo's main dependency).

Benchmark harness: this repo, branch `kbd/subgraph-matching-extended`, at
`0062dbb488718f73936a24eca40de5bfad7ad374`.

## Systems

| System | Repository | Commit / version |
|---|---|---|
| Crow's Nest | [finch-tensor/finch-tensor](https://github.com/finch-tensor/finch-tensor), branch `kbd/workspacer-accumulators` | `84add51c4101d5022516524b6c82bab5ad164d0f` (locked in `pixi.lock`) |
| Galley.jl, local checkout (subgraph matching) | [kylebd99/Galley.jl](https://github.com/kylebd99/Galley.jl), branch `sigmod-branch` | `d19e38a80a9003527a66881052cb5cffdb2e9192`, with Finch.jl 0.6.33, on Julia 1.11.7 |
| Galley.jl, registry release (installed by `subgraph/galley_jl/`, and used through galley-jl-python) | Julia General registry | Galley 0.2.2, the same tree as Galley.jl `main` at `8eefbb52c64c9dd1f7cad39e30c9541a9ab9dc56`, on Julia 1.11.7 |
| Galley (microbenchmarks) | [finch-tensor/galley-jl-python](https://github.com/finch-tensor/galley-jl-python) | `55d29268f338af9e3bc7893f232bc152c8ba42b2` (locked in `pixi.lock`), running Galley 0.2.2 |
| Scorch | [bobbyyyan/scorch](https://github.com/bobbyyyan/scorch) (`external/scorch`) | `b3b5d4b77a3bbdd395b19c289bf181cb3cf7df73` |
| SparseAuto | [adhithadias/tensor-schedules](https://github.com/adhithadias/tensor-schedules) (`external/tensor-schedules`) | `d77d045f5d55eb2f1f5f8a62c5a380c4487ccf35` |
| SparseAuto's TACO backend | [adhithadias/SparseLNR](https://github.com/adhithadias/SparseLNR) (`external/sparseLNR`) | `e359ea1516c0466c28bcbf44b502e28d01c730d7` |

## Experiments

### Subgraph matching (`graph/subgraph_matching/`)

Figures: `dblp_youtube_84add51c_lp_galleyjl_{paired,capped}_boxplot.{pdf,png}`.

- **Crow's Nest** used finch-tensor `84add51c` with LP statistics. Result files:
  - `dblp_20261112_153423.csv`
  - `youtube_20261112_154956_merged_retry450.csv`, which merges in the 450 s
    reruns of the timeouts from `youtube_retry450_20261112_232352.csv`
- **Galley.jl** used the local `sigmod-branch` checkout. Result files:
  - `dblp_galleyjl_20261109_183707.csv`
  - `youtube_galleyjl_20261109_202132_merged_retry450.csv`

  One exception: the 450 s rerun of YouTube query `sparse_8/query_sparse_8_84`
  (`youtube_galleyjl_retry450_20261112_232502.csv`) used registry Galley 0.2.2,
  because the runner had been switched to it on 2026-11-10 (commit `211b8c1`).

### Microbenchmarks (`graph/scripts/lp_striped_fixed/`)

Figures: `all_benchmarks`, `sddmm`, `chain`, `spgemm` and `matrix_features`
(`.pdf` and `.png`).

Every system ran single-threaded, each pinned to its own core and under its own
memory cap. All of them computed in float32, except Crow's Nest's row norms,
which use float64.

- **Crow's Nest:** finch-tensor `84add51c`, in this repo's `default` pixi environment.
- **Galley:** galley-jl-python `55d29268` with Galley 0.2.2, in this repo's
  `galley` pixi environment.
- **Scorch and SparseAuto:** built and run from a second checkout of this repo
  (branch `kbd/run-all-repro`, commit
  `5aca6687d317275d275143744b4392ea9712112b`), in its `scorch` and `sparseauto`
  pixi environments, with the runner scripts from this repo. That checkout had
  uncommitted changes:
  - `pyproject.toml`: build Scorch with the system GCC on Linux; add
    `-Wno-narrowing` to the SparseLNR build.
  - `external/sparseLNR/test/tests-workspaces.cpp`: the `clock_t` include fix
    that the `build-sparseauto` task applies.
  - `external/tensor-schedules`: untracked SparseAuto configs and schedules
    (`test_configs/finchbench_{chain,sddmm,spgemm_sa}_config.json` and
    `test_schedules/finchbench_chain_*.json`).

### Estimator accuracy (`graph/stats_comparison/`)

Files: `accuracy.json`, `accuracy.{pdf,png}` and `geomean.{pdf,png}`. All the
estimators come from finch-tensor `84add51c`. Running it again on 2026-11-13
reproduced the 2026-11-11 results exactly.

## Known inconsistencies

1. **Two Galley.jl versions.** The main Galley.jl subgraph results came from
   `sigmod-branch` (`d19e38a8`, Finch.jl 0.6.33). One YouTube rerun and all the
   microbenchmarks used Galley 0.2.2 (`8eefbb52`). To make them consistent,
   either rerun the Galley.jl subgraph results with
   `galley_jl_subgraph.py` (which installs 0.2.2), or pin `subgraph/galley_jl/`
   to `d19e38a8` and rerun `query_sparse_8_84`.
2. **The Scorch/SparseAuto setup isn't committed.** The changes listed under
   Microbenchmarks need to be committed (here or upstream) before those numbers
   can be reproduced from the repository alone.
