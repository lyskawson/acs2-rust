#!/bin/bash
set -euo pipefail
operations=$1
root=$2
assignment=$3
launcher=${4:-launch.py}
if test "$SLURM_JOB_PARTITION" = lem-cpu-normal &&
    srun --cpu-bind=cores python3 -B "$operations/$launcher" "$root" "$assignment" "$SLURM_ARRAY_TASK_ID"; then
    exit 0
fi
python3 -B "$operations/$launcher" "$root" "$assignment" "$SLURM_ARRAY_TASK_ID" --failed-step
exit 1
