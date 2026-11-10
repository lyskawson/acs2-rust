#!/bin/sh
set -eu
operations=$1
root=$2
assignment=$3
exec python3 -B "$operations/launch.py" "$root" "$assignment" "$SLURM_ARRAY_TASK_ID"
