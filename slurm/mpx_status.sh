#!/bin/bash
# Compact status of the MPX runs on WCSS Bem2. Run locally: ./slurm/mpx_status.sh

set -euo pipefail

ssh -i ~/.ssh/id_rsa_wcss -o ConnectTimeout=20 alelys2099@ui.wcss.pl 'bash -s' <<'REMOTE'
echo "=== QUEUE ==="
squeue -u alelys2099 -o "%.10i %.14j %.18P %.2t %.12M %.12L %.14R"

echo
echo "=== PROGRESS ==="
printf "%-24s %-14s %10s %8s %9s %5s %6s %7s\n" log state trials wall knowledge rel spec pop
for f in ~/acs2-rust/reports/slurm_mpx*.out ~/acs2-rust-repo/reports/slurm_mpx*.out; do
  [ -f "$f" ] || continue
  # only canonical live logs; archived runs carry an extra .<tag> before .out
  [[ "$(basename "$f")" =~ ^slurm_mpx[0-9]+_s[0-9]+\.out$ ]] || continue

  verdict=$(grep -oE 'repeat 0: [A-Z-]+' "$f" 2>/dev/null | tail -1 | awk '{print $3}')
  state=${verdict:-running}

  # a finished log ends with the verdict-agreement footer, so read the last
  # trajectory point rather than the last line
  grep 'traj:' "$f" | tail -1 | awk -v name="$(basename "$f" .out | sed 's/^slurm_//')" -v st="$state" '
    /traj:/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, "=")
        v[kv[1]] = kv[2]
      }
      gsub(/s$/, "", v["wall"])
      printf "%-24s %-14s %10d %8d %9.4f %5d %6.2f %7d\n",
             name, st, v["trials"], v["wall"], v["knowledge"], v["reliable"], v["spec"], v["pop"]
      done = 1
    }
    END { if (!done) printf "%-24s %-14s %10s\n", name, st, "(starting)" }
  '
done
REMOTE
