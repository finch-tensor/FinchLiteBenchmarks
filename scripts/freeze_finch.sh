#!/bin/bash
# Print a checkout of finch-tensor frozen at a commit, creating it if needed, so a
# benchmark campaign is not affected by later edits to the working tree:
#
#   FINCH=$(scripts/freeze_finch.sh 7a00e273)
#   PYTHONPATH=$FINCH/src:src python -m finchtensorbench.benchmarks.subgraph_matching ...
#
# PYTHONPATH puts the frozen sources ahead of the editable install; workers
# inherit it, and subgraph_matching checks every worker imports the same
# revision.  Args: COMMIT [FINCH_REPO] (default ~/finch-tensor/.claude/worktrees/loop-model).
set -euo pipefail
commit=${1:?usage: freeze_finch.sh COMMIT [FINCH_REPO]}
repo=${2:-$HOME/finch-tensor/.claude/worktrees/loop-model}
sha=$(git -C "$repo" rev-parse --short=10 "$commit^{commit}")
dir=${FINCH_FROZEN_ROOT:-$HOME/finch-frozen}/$sha
if [ ! -d "$dir" ]; then
  mkdir -p "$(dirname "$dir")"
  git -C "$repo" worktree add --detach "$dir" "$sha" >&2
fi
echo "$dir"
