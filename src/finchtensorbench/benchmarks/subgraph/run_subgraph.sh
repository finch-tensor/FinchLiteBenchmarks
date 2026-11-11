#!/usr/bin/env bash
# Run the G-CARE subgraph-matching benchmarks for each dataset, then plot them
# and, with Galley.jl, compare finch against it.
#
#   src/finchtensorbench/benchmarks/subgraph/run_subgraph.sh \
#       -d "human aids" -c galley -j 4 -m 8 -g -- --timeout 300
#
# Options (default in brackets):
#   -d "A B"   datasets to run [human]
#   -c "A B"   finch scheduler configs ["default galley"]
#   -j N       parallel workers per run [1]
#   -m GB      memory cap per worker process, 0 to disable [16]
#   -g         also run Galley.jl
#   -a "ARGS"  extra arguments for galley_jl_subgraph, e.g. "--timeout 300"
#   -o DIR     results directory [graph/subgraph_matching]
#   -- ARGS    passed to subgraph_matching, e.g. -- --timeout 300 --sample 20
#
# Each dataset's CSVs, plans, logs and box plot go in DIR/<dataset>_<time>/.
# To plot or compare existing CSVs, run the Python modules directly:
#
#   pixi run python -m finchtensorbench.benchmarks.subgraph.plot_subgraph_matching A.csv B.csv
#   pixi run python -m finchtensorbench.benchmarks.subgraph.compare_subgraph A.csv B.csv
set -euo pipefail

datasets=human
configs="default galley"
jobs=1
memory_gb=16
galley_jl=0
galley_args=""
out_dir=graph/subgraph_matching

while getopts "d:c:j:m:ga:o:" opt; do
    case $opt in
        d) datasets=$OPTARG ;;
        c) configs=$OPTARG ;;
        j) jobs=$OPTARG ;;
        m) memory_gb=$OPTARG ;;
        g) galley_jl=1 ;;
        a) galley_args=$OPTARG ;;
        o) out_dir=$OPTARG ;;
        *) sed -n '2,22p' "$0"; exit 2 ;;
    esac
done
shift $((OPTIND - 1))

cd "$(dirname "$0")/../../../.."
run() { pixi run python -m "finchtensorbench.benchmarks.subgraph.$1" "${@:2}"; }

for dataset in $datasets; do
    dir="$out_dir/${dataset}_$(date +%Y%m%d_%H%M%S)"
    echo "######## $dataset -> $dir"
    run subgraph_matching --dataset "$dataset" --configs $configs --jobs "$jobs" \
        --memory-limit-gb "$memory_gb" --output-dir "$dir" "$@"
    if [ "$galley_jl" = 1 ]; then
        run galley_jl_subgraph --dataset "$dataset" --jobs "$jobs" \
            --memory-limit-gb "$memory_gb" --out-dir "$dir" $galley_args
        run compare_subgraph "$dir"/*.csv --baseline galley.jl
    fi
    run plot_subgraph_matching "$dir"/*.csv --out "$dir/boxplot.png"
done
