#!/bin/bash
set -euo pipefail
operations=$1
assignment=$2
phase=$3
launcher="$operations/grid_run.py"
trap 'python3 -B "$launcher" failed "$assignment"' TERM
test "$SLURM_JOB_PARTITION" = lem-cpu-normal
if test "$phase" = gate; then
    srun --ntasks=1 --cpu-bind=cores python3 -B "$launcher" worker "$assignment" ||
        python3 -B "$launcher" failed "$assignment"
else
    python3 -B "$launcher" worker "$assignment"
fi
