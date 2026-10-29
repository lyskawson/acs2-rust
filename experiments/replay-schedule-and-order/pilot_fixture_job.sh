#!/usr/bin/env bash
set -euo pipefail
grid_operations=$1
batch_root=$2
assignment=$3
bash "$grid_operations/grid_job.sh" "$grid_operations" "$batch_root" "$assignment"
exec /usr/bin/python3 -B "$grid_operations/../replay-schedule-and-order/verify_pilot_fixture.py" "$batch_root" "$assignment" "$SLURM_ARRAY_TASK_ID"
