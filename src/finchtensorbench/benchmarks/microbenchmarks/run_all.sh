#!/usr/bin/env bash
# Run every benchmark backend, then plot the results. Also: pixi run run-all [backends...]
#
#   src/finchtensorbench/benchmarks/microbenchmarks/run_all.sh                     # all backends
#   src/finchtensorbench/benchmarks/microbenchmarks/run_all.sh finch galley        # only the named backends
#   THREADS=4 src/finchtensorbench/benchmarks/microbenchmarks/run_all.sh           # change the thread count (default 1)
#
# Each runner overwrites its CSV in graph/scripts/.
# A failing backend does not stop the others; the script exits non-zero at the end.
set -uo pipefail

# This file lives in src/finchtensorbench/benchmarks/microbenchmarks/; work from the repo root.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
SCRIPTS=src/finchtensorbench/benchmarks/microbenchmarks
RESULTS=graph/scripts

######################### 1. Environment ##########################

# Thread count for the backends that parallelize. Finch and Galley generate
# serial Julia loops (no BLAS), so they need nothing here.
#   THREADS          run_sparseauto.py: TACO thread count and SparseAuto's parallel schedules
#   OMP_NUM_THREADS  Scorch: its OpenMP kernels and PyTorch ops use every core when unset
export THREADS="${THREADS:-1}"
export OMP_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1
# Per-backend variables (PYTHONPATH, Julia depots, compilers, SDKROOT, ...) are
# set by each pixi environment's activation block in pyproject.toml.

command -v pixi >/dev/null || { echo "pixi is not installed" >&2; exit 1; }

ALL=(finch galley scorch sparseauto)
if (($#)); then BACKENDS=("$@"); else BACKENDS=("${ALL[@]}"); fi
for backend in "${BACKENDS[@]}"; do
    [[ " ${ALL[*]} " == *" $backend "* ]] || { echo "unknown backend: $backend (choose from: ${ALL[*]})" >&2; exit 1; }
done

needs() { [[ " ${BACKENDS[*]} " == *" $1 "* ]]; }

# Scorch and SparseAuto live in git submodules.
if needs scorch || needs sparseauto; then
    git submodule update --init external/scorch external/sparseLNR external/tensor-schedules
fi
if needs sparseauto && [[ ! -x external/sparseLNR/build/bin/taco-test ]]; then
    echo "==> Building SparseAuto (TACO fork)"
    pixi run -e sparseauto build-sparseauto || { echo "SparseAuto build failed" >&2; exit 1; }
fi

######################### 2. Experiments ##########################

# Each backend runs in its own pixi environment; Finch uses the default one.
environment() { [[ $1 == finch ]] && echo default || echo "$1"; }
FAILED=()
for backend in "${BACKENDS[@]}"; do
    echo
    echo "==> Running $backend (pixi env: $(environment "$backend"), threads: $THREADS)"
    start=$SECONDS
    if pixi run -e "$(environment "$backend")" python "$SCRIPTS/run_${backend}.py" \
            --output "$RESULTS/${backend}_result.csv"; then
        echo "==> $backend finished in $((SECONDS - start))s"
    else
        echo "==> $backend FAILED after $((SECONDS - start))s" >&2
        FAILED+=("$backend")
    fi
done

######################### 3. Plots ##########################

echo
echo "==> Plotting"
# plot.py picks up every <backend>_result.csv present in graph/scripts/.
pixi run -e default python "$SCRIPTS/plot.py" --output "$RESULTS" || FAILED+=("plot")

if ((${#FAILED[@]})); then
    echo "Failed: ${FAILED[*]}" >&2
    exit 1
fi
echo "All done. Results and plots are in $RESULTS/"
