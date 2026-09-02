#!/bin/bash
#SBATCH --job-name=mpx-reach
#SBATCH --partition=bem2-cpu-normal
#SBATCH --time=7-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --output=/home/alelys2099/mpx_runs/slurm-%j.wrapper

set -euo pipefail

SIZE="$1"
SEED="$2"
TIME_CAP="${3:-600000}"
shift 3 2>/dev/null || shift $#

REPO="$HOME/acs2-rust-repo"
# Results land OUTSIDE the checkout: the repo also carries committed copies of
# past logs, and writing live output into a tracked directory makes every
# git pull collide with a running job.
RUNS="${MPX_RUNS_DIR:-$HOME/mpx_runs}"
mkdir -p "$RUNS"
OUT="$RUNS/slurm_mpx${SIZE}_s${SEED}${TAG:+_$TAG}.out"

cd "$REPO"
exec "$REPO/target/x86_64-unknown-linux-musl/release/mpx_reach" \
  --sizes "$SIZE" \
  --n-exp 1 \
  --seed "$SEED" \
  --time-cap-secs "$TIME_CAP" \
  --u-max "${U_MAX:-derived}" \
  --alp-gen-variant pyalcs \
  --agent "${AGENT:-acs2}" \
  --encoding "${ENCODING:-flip}" \
  --log-trajectory \
  --eval-interval "${EVAL_INTERVAL:-60000}" \
  "$@" \
  >"$OUT" 2>&1
