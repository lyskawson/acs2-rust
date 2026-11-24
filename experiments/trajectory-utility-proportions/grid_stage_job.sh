#!/bin/bash
set -euo pipefail
operations=$1
assignment=$2
if test "$SLURM_JOB_PARTITION" = lem-cpu-normal &&
    python3 -B "$operations/grid_stage.py" worker "$assignment"; then
    exit 0
fi
python3 -B "$operations/grid_stage.py" failed "$assignment"
exit 1
