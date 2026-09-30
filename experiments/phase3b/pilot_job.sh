#!/bin/bash
set -euo pipefail
repo=$1
commit=$2
destination=$3
test "$SLURM_JOB_PARTITION" = lem-cpu-normal
test "$(git -C "$repo" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$repo" status --porcelain)"
cd "$repo"
sha256sum --check "$destination/binary.sha256"
srun --cpu-bind=cores python3 -B experiments/phase3b/collect.py \
    --manifest experiments/phase3b/pilot.json \
    --binary target/x86_64-unknown-linux-musl/release/acs2-measure \
    --commit "$commit" --output "$destination/runs" --index "$SLURM_ARRAY_TASK_ID"
