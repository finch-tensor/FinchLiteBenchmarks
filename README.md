# FinchLiteBenchmarks

## ErrorLoggerStats

Run the Luxembourg road network from the repository root:

```sh
pixi run python src/finchtensorbench/benchmarks/errorlogger_stats.py --datasets luxembourg_osm
```

The [SuiteSparse Luxembourg road network](https://sparse.tamu.edu/DIMACS10/luxembourg_osm)
has 114,599 nodes and 239,332 nonzeros. The Matrix Market input is stored in the
repository at `data/luxembourg_osm.mtx`, so no download is needed. The plot is saved to
`graph/errorlogger_results/errorlogger_luxembourg_osm_matmul.png`.

Use `--datasets bcsstk39 ca-GrQc luxembourg_osm` to run all three datasets, or
`--help` to list the available options.
