#!/bin/bash
set -euo pipefail
operations=$1
root=$2
assignment=$3
if test "$SLURM_JOB_PARTITION" = lem-cpu-normal && srun --cpu-bind=cores python3 -B "$operations/grid_launch.py" "$root" "$assignment" "$SLURM_ARRAY_TASK_ID"; then
    exit 0
else
    python3 -B "$operations/grid_failure.py" "$root" "$assignment" "$SLURM_ARRAY_TASK_ID"
    exit 1
fi
