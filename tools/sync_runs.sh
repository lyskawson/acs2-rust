#!/bin/bash
# Pull every cluster run into the repo and rebuild the archive from it.
#
# Nothing does this on its own. A SLURM job writes only to ~/mpx_runs/ on the
# cluster, outside any checkout, so until this script runs the cluster home is
# the SINGLE copy of a result that cost days of compute. Run it after every
# batch finishes, and before the grant expires.
#
#   ./tools/sync_runs.sh            pull, rebuild, report
#   ./tools/sync_runs.sh --commit   the same, then commit the result

set -euo pipefail

REMOTE="alelys2099@ui.wcss.pl"
KEY="$HOME/.ssh/id_rsa_wcss"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cd "$REPO"

commit=false
local_only=false
for arg in "$@"; do
  case "$arg" in
    --commit) commit=true ;;
    --local) local_only=true ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

new=0
updated=0
if ! $local_only; then
  echo "==> pulling ~/mpx_runs from the cluster"
  # Checkpoints live under ~/mpx_runs/checkpoints/ and are resumable state, not results:
  # a k=264 one is ~1.5 KB per classifier and a chained run keeps several. Pulling them
  # would move hundreds of megabytes on every session start and then discard them --
  # the result of a run is its log.
  rsync -az --exclude 'checkpoints/' -e "ssh -i $KEY -o ConnectTimeout=30" \
    "$REMOTE:~/mpx_runs/" "$STAGE/"
  for file in "$STAGE"/*.out "$STAGE"/*.cancelled; do
    [ -f "$file" ] || continue
    name="$(basename "$file")"
    if [ ! -f "reports/$name" ]; then
      new=$((new + 1))
      echo "    new     $name"
    elif ! cmp -s "$file" "reports/$name"; then
      updated=$((updated + 1))
      echo "    updated $name"
    fi
    cp "$file" "reports/$name"
  done
  echo "==> $new new, $updated updated"
fi

echo "==> rebuilding the CSV archive"
python3 tools/parse_mpx_logs.py reports/slurm_*.out reports/*.cancelled \
  reports/mpx_m2b_reach*.log reports/mpx_m3_e1_traj70_*.log

echo "==> rebuilding the readable tables"
python3 tools/rebuild_tables.py

echo "==> runs that reached knowledge 1.0"
python3 tools/list_solved.py

if $commit; then
  if [ -z "$(git status --porcelain --untracked-files=all -- reports/)" ]; then
    echo "==> nothing changed, no commit"
  else
    git add reports/
    git commit -q -m "data: sync cluster runs into the archive

$new new logs, $updated updated. CSVs and per-size tables rebuilt by
tools/sync_runs.sh."
    echo "==> committed"
  fi
fi
