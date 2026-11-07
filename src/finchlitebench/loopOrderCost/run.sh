#!/usr/bin/env bash
#$ -cwd
#$ -S /bin/bash
#$ -pe omp 1
#$ -binding linear:1
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate "$HOME/finch-env"
cd "${SGE_O_WORKDIR:-$(dirname "${BASH_SOURCE[0]}")/../../..}"

export PYTHONPATH="$PWD/../finch-tensor-lite/src:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="$CONDA_PREFIX/julia_env/pyjuliapkg/install/lib/julia/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export JULIA_NUM_THREADS=1 JULIA_NUM_GC_THREADS=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHON_JULIAPKG_OFFLINE=yes

python -u - <<'PY'
import os

from finch.compile_jl.julia import get_jl

print(f"CPU affinity (logical CPU IDs): {sorted(os.sched_getaffinity(0))}", flush=True)
jl = get_jl()
version = str(jl.seval("Base.pkgversion(Finch)"))
assert version == "1.5.0", f"Expected Finch.jl 1.5.0, got {version}"
assert jl.seval("length(methods(Finch.SparseHashLevel{Int64,true}))") > 0
print(f"Finch.jl {version} ready", flush=True)
PY

if [[ "${1:-}" == "--benchmark-file" ]]; then
    benchmark_file="${2:?Supply a Python file after --benchmark-file}"
    shift 2
    exec python -u "$benchmark_file" "$@"
fi

exec python -u -m finchlitebench.loopOrderCost.microBenchmarks3 "$@"
