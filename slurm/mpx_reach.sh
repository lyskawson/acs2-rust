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

# Extra flags are appended after the ones this script controls, and mpx_reach takes the
# last value of a repeated flag. A trailing --time-cap-secs would sail past the allocation
# check below, and a trailing --checkpoint-path would defeat the derived path that keeps
# two jobs off one learning state.
for argument in "$@"; do
  case "$argument" in
    --time-cap-secs|--checkpoint-path|--checkpoint-every|--sizes|--seed|--n-exp)
      echo "refusing to start: $argument is set by this wrapper and would be overridden by \
the trailing argument, silently, because mpx_reach takes the last value of a repeated flag. \
Pass it through the wrapper's own positional arguments or environment instead." >&2
      exit 2
      ;;
  esac
done

# The internal cap must fire before SLURM's, or the job is killed hard and everything
# since the last periodic checkpoint is re-run. #SBATCH --time above is only a default:
# a k=264 chain overrides it on the sbatch line, and raising TIME_CAP without raising
# --time is the mistake this catches. Silent where SLURM is absent, so local runs and the
# test suite are unaffected, and run BEFORE the log is opened so a refusal cannot truncate
# a previous attempt's output.
# SLURM prints a TimeLimit as `d-hh:mm:ss` or `hh:mm:ss`. Every field is forced to base 10:
# an ordinary `08:00:00` is octal to bash arithmetic and would fail. An unrecognised format
# yields nothing and the check below skips, which is the safe direction.
seconds_of_slurm_time() {
  local text="$1" days=0
  case "$text" in
    *-*) days=$(( 10#${text%%-*} )); text="${text#*-}" ;;
  esac
  case "$text" in
    *:*:*)
      echo $(( days * 86400 \
        + 10#$(echo "$text" | cut -d: -f1) * 3600 \
        + 10#$(echo "$text" | cut -d: -f2) * 60 \
        + 10#$(echo "$text" | cut -d: -f3) ))
      ;;
    *:*)
      [ "$days" -eq 0 ] || { echo ""; return; }
      echo $(( 10#$(echo "$text" | cut -d: -f1) * 60 + 10#$(echo "$text" | cut -d: -f2) ))
      ;;
    *) echo "" ;;
  esac
}

# No scontrol at all means this is not a compute node -- a local run or the test suite --
# and the check is skipped. scontrol present but unable to answer is different: something is
# wrong on a node where the check matters, and continuing would risk the whole allocation.
if [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  allocated_text=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
    | tr ' ' '\n' | sed -n 's/^TimeLimit=//p' | head -1 || true)
  allocated=""
  if [ -n "$allocated_text" ] && [ "$allocated_text" != "UNLIMITED" ]; then
    allocated=$(seconds_of_slurm_time "$allocated_text" 2>/dev/null || true)
  fi
  if [ "${allocated_text:-}" != "UNLIMITED" ] \
     && { [ -z "$allocated" ] || ! [ "$allocated" -gt 0 ] 2>/dev/null; } \
     && [ "${CHECKPOINT_SKIP_TIME_CHECK:-0}" != "1" ]; then
    echo "refusing to start: cannot read this job's TimeLimit from scontrol (got \
'${allocated_text:-nothing}'), so --time-cap-secs ${TIME_CAP}s cannot be checked against the \
allocation. Set CHECKPOINT_SKIP_TIME_CHECK=1 to proceed anyway." >&2
    exit 2
  fi
  if [ -n "$allocated" ] && [ "$allocated" -gt 0 ] 2>/dev/null; then
    {
      if [ "$allocated" -ge 7200 ]; then margin=3600; else margin=60; fi
      if [ "$TIME_CAP" -gt $(( allocated - margin )) ]; then
        echo "refusing to start: --time-cap-secs ${TIME_CAP}s leaves under ${margin}s of the \
${allocated}s SLURM allocation ($allocated_text). SLURM would kill the job before it saves, \
losing every trial since the last periodic checkpoint. Raise --time on the sbatch line or \
lower the cap." >&2
        exit 2
      fi
    }
  fi
fi

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
  # SLURM requeues a job under the SAME id after a node failure or preemption, which is
  # the exact case checkpointing exists for. Without the attempt in the name the second
  # run truncates the first's log: the run continues from its checkpoint, and the
  # trajectory of everything before it is gone.
  OUT="$RUNS/${BASE}_seg${SLURM_JOB_ID:-$$}${SLURM_RESTART_COUNT:+.r$SLURM_RESTART_COUNT}.out"
else
  OUT="$RUNS/${BASE}.out"
fi

cd "$REPO"

# The log must be self-describing: an archived run has to be reproducible from
# the file alone, without the submitting shell or the job name.
{
  echo "run-provenance: commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown) \
job=${SLURM_JOB_ID:-none} attempt=${SLURM_RESTART_COUNT:-0} tag=${TAG:-none} size=$SIZE \
seed=$SEED time_cap=${TIME_CAP}s \
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
