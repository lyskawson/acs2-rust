#!/bin/bash
# Compact status of the MPX runs on WCSS Bem2. Run locally: ./slurm/mpx_status.sh

set -euo pipefail

ssh -i ~/.ssh/id_rsa_wcss -o ConnectTimeout=20 alelys2099@ui.wcss.pl 'bash -s' <<'REMOTE'
echo "=== QUEUE ==="
if queue=$(squeue -u alelys2099 -o "%.10i %.14j %.18P %.2t %.11M %.12L" 2>&1); then
  echo "$queue"
  [ "$(echo "$queue" | wc -l)" -le 1 ] && echo "(no jobs queued or running)"
else
  echo "!! squeue FAILED -- job state below is unknown, not empty:"
  echo "$queue"
fi

echo
echo "=== PROGRESS ==="
printf "%-34s %-14s %10s %8s %9s %5s %6s %7s\n" log state trials wall knowledge rel spec pop

# Prefer the git clone over the legacy rsync tree when both hold the same log,
# and skip archived runs (.cancelled./.partial.) which are history, not status.
declare -A seen
for f in ~/mpx_runs/slurm_mpx*.out ~/acs2-rust-repo/reports/slurm_mpx*.out ~/acs2-rust/reports/slurm_mpx*.out; do
  [ -f "$f" ] || continue
  name=$(basename "$f" .out)
  case "$name" in *.cancelled|*.partial) continue;; esac
  [ -n "${seen[$name]:-}" ] && continue
  seen[$name]="$f"
done

for name in $(printf '%s\n' "${!seen[@]}" | sort); do
  f="${seen[$name]}"
  verdict=$(grep -oE 'repeat 0: [A-Z-]+' "$f" 2>/dev/null | tail -1 | awk '{print $3}')
  state=${verdict:-running}

  # a finished log ends with the verdict-agreement footer, so read the last
  # trajectory point rather than the last line
  grep 'traj:' "$f" 2>/dev/null | tail -1 | awk -v name="${name#slurm_}" -v st="$state" '
    /traj:/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        v[kv[1]] = kv[2]
      }
      gsub(/s$/, "", v["wall"])
      printf "%-34s %-14s %10d %8d %9.4f %5d %6.2f %7d\n",
             name, st, v["trials"], v["wall"], v["knowledge"], v["reliable"], v["spec"], v["pop"]
      done = 1
    }
    END { if (!done) printf "%-34s %-14s %10s\n", name, st, "(starting)" }
  '
done
REMOTE
