#!/usr/bin/env bash
# Run the G-CARE subgraph-matching benchmarks and plot the results.
#
# For each dataset: runs finch (subgraph_matching) and optionally Galley.jl
# (galley_jl_subgraph), then draws one box plot of all of that dataset's CSVs
# and, when both ran, prints finch's speedup over Galley.jl.
#
#   src/finchtensorbench/benchmarks/subgraph/run_subgraph.sh \
#       -d "human aids" -c "galley" --galley-jl -j 4 -m 8 -- --timeout 300
#
# Options:
#   -d, --datasets "A B"   datasets to run (default: human)
#   -c, --configs "A B"    finch scheduler configs (default: "default galley")
#   -j, --jobs N           parallel workers per run (default: 1)
#   -o, --out-dir DIR      results directory (default: graph/subgraph_matching)
#   -m, --memory-limit-gb G  memory cap per worker process, finch and Galley.jl
#                          (default: 16; 0 disables)
#       --galley-jl        also run Galley.jl
#       --galley-args "…"  extra arguments for galley_jl_subgraph
#       --no-finch         skip the finch run (e.g. Galley.jl only)
#       --plot CSV...      only plot/compare existing CSVs, run nothing (the
#                          last CSV's config is the comparison baseline)
#   --                     everything after is passed to subgraph_matching
set -euo pipefail

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
ROOT="$(cd "$(dirname "$SCRIPT")/../../../.." && pwd)"
cd "$ROOT"
PKG=finchtensorbench.benchmarks.subgraph

datasets="human"
configs="default galley"
jobs=1
memory_gb=16
out_dir="graph/subgraph_matching"
galley_jl=0
galley_args=""
finch=1
plot_only=()
finch_args=()

while (($#)); do
    case "$1" in
        -d|--datasets) datasets="$2"; shift 2 ;;
        -c|--configs) configs="$2"; shift 2 ;;
        -j|--jobs) jobs="$2"; shift 2 ;;
        -o|--out-dir) out_dir="$2"; shift 2 ;;
        -m|--memory-limit-gb) memory_gb="$2"; shift 2 ;;
        --galley-jl) galley_jl=1; shift ;;
        --galley-args) galley_args="$2"; shift 2 ;;
        --no-finch) finch=0; shift ;;
        --plot) shift; while (($#)) && [[ "$1" != -* ]]; do plot_only+=("$1"); shift; done ;;
        --) shift; finch_args=("$@"); break ;;
        -h|--help) sed -n '2,23p' "$SCRIPT" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

py() { pixi run python -m "$PKG.$1" "${@:2}"; }

# Run a benchmark module, echoing its output, and print the CSV it wrote.
run_csv() {
    local log
    log="$(mktemp)"
    py "$@" 2>&1 | tee "$log" >&2
    grep -E '^wrote .*\.csv$' "$log" | tail -1 | cut -d' ' -f2-
    rm -f "$log"
}

plot_and_compare() {  # plot_and_compare FINCH_CSV|"" GALLEY_CSV|"" [OUT_PNG]
    local finch_csv="$1" galley_csv="$2" out="${3:-}"
    local csvs=()
    [[ -n "$finch_csv" ]] && csvs+=("$finch_csv")
    [[ -n "$galley_csv" ]] && csvs+=("$galley_csv")
    ((${#csvs[@]})) || return 0
    echo -e "\n== plot: ${csvs[*]}"
    py plot_subgraph_matching "${csvs[@]}" ${out:+--out "$out"}
    if [[ -n "$finch_csv" && -n "$galley_csv" ]]; then
        echo -e "\n== compare"
        py compare_subgraph "$finch_csv" "$galley_csv"
    fi
}

if ((${#plot_only[@]})); then
    py plot_subgraph_matching "${plot_only[@]}"
    if ((${#plot_only[@]} > 1)); then
        py compare_subgraph "${plot_only[@]}"
    fi
    exit 0
fi

summary=()
for dataset in $datasets; do
    finch_csv="" galley_csv=""
    if ((finch)); then
        echo -e "\n######## finch: $dataset ($configs)"
        # shellcheck disable=SC2086
        finch_csv="$(run_csv subgraph_matching --dataset "$dataset" --configs $configs \
            --jobs "$jobs" --memory-limit-gb "$memory_gb" --output-dir "$out_dir" "${finch_args[@]}")" || true
        [[ -n "$finch_csv" ]] || echo "finch run on $dataset wrote no CSV" >&2
    fi
    if ((galley_jl)); then
        echo -e "\n######## Galley.jl: $dataset"
        # shellcheck disable=SC2086
        galley_csv="$(run_csv galley_jl_subgraph --dataset "$dataset" \
            --jobs "$jobs" --memory-limit-gb "$memory_gb" --out-dir "$out_dir" $galley_args)" || true
        [[ -n "$galley_csv" ]] || echo "Galley.jl run on $dataset wrote no CSV" >&2
    fi
    out=""
    if [[ -n "$finch_csv" && -n "$galley_csv" ]]; then
        out="${finch_csv%.csv}_vs_galleyjl_boxplot.png"
    fi
    plot_and_compare "$finch_csv" "$galley_csv" "$out"
    summary+=("$dataset: ${finch_csv:-} ${galley_csv:-} ${out:-}")
done

echo -e "\n######## done"
printf '  %s\n' "${summary[@]}"
