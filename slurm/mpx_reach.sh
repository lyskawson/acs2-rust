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

REPO="${MPX_REPO_DIR:-$HOME/acs2-rust-repo}"
# Results land OUTSIDE the checkout: the repo also carries committed copies of
# past logs, and writing live output into a tracked directory makes every
# git pull collide with a running job.
RUNS="${MPX_RUNS_DIR:-$HOME/mpx_runs}"
mkdir -p "$RUNS"
BASE="slurm_mpx${SIZE}_s${SEED}${TAG:+_$TAG}"

# CHECKPOINT=on makes the run resumable across jobs, which is what k=264 needs:
# one seed is 700-4500 CPU-hours against a 504 h queue limit.
#
# The path is derived, never passed in. It keys on size, seed and tag -- the same
# triple the log name keys on -- so two jobs cannot be pointed at one learning
# state by accident, and it lands beside the logs, OUTSIDE the checkout.
#
# Each job writes its own log. A single filename per run would leave only the last
# segment, and the segments are where the trajectory lives.
CHECKPOINT_ARGS=()
if [ "${CHECKPOINT:-off}" = "on" ]; then
  CHECKPOINT_PATH="$RUNS/checkpoints/${BASE}.ckpt"
  mkdir -p "$RUNS/checkpoints"
  CHECKPOINT_ARGS=(--checkpoint-path "$CHECKPOINT_PATH" --checkpoint-every "${CHECKPOINT_EVERY:-0}")
  OUT="$RUNS/${BASE}_seg${SLURM_JOB_ID:-$$}.out"
else
  OUT="$RUNS/${BASE}.out"
fi

cd "$REPO"

# The log must be self-describing: an archived run has to be reproducible from
# the file alone, without the submitting shell or the job name.
{
  echo "run-provenance: commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
job=${SLURM_JOB_ID:-none} tag=${TAG:-none} size=$SIZE seed=$SEED time_cap=${TIME_CAP}s \
partition=${SLURM_JOB_PARTITION:-none} host=$(hostname) started=$(date -Is)"
  echo "run-argv: $* "
  if [ "${CHECKPOINT:-off}" = "on" ]; then
    echo "run-segment: base=$BASE checkpoint=$CHECKPOINT_PATH checkpoint_every=${CHECKPOINT_EVERY:-0} \
resumed=$([ -f "$CHECKPOINT_PATH" ] && echo yes || echo no)"
  fi
} >"$OUT"

exec "${MPX_BINARY:-$REPO/target/x86_64-unknown-linux-musl/release/mpx_reach}" \
  --sizes "$SIZE" \
  --n-exp 1 \
  --seed "$SEED" \
  --time-cap-secs "$TIME_CAP" \
  --u-max "${U_MAX:-derived}" \
  --alp-gen-variant pyalcs \
  --agent "${AGENT:-acs2}" \
  --encoding "${ENCODING:-flip}" \
  --epsilon "${EPSILON:-0.8}" \
  --log-trajectory \
  --log-accuracy \
  --eval-interval "${EVAL_INTERVAL:-60000}" \
  ${CHECKPOINT_ARGS[@]+"${CHECKPOINT_ARGS[@]}"} \
  "$@" \
  >>"$OUT" 2>&1
