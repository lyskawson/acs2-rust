"""Regenerate the readable per-size table for every size present in the archive.

Sizes are read from the archive rather than hard-coded, so a first run at a new
multiplexer size gets its page without anyone remembering to add it.
"""

import csv
import subprocess
import sys
from pathlib import Path

VERDICTS = Path("reports/mpx_verdicts.csv")
TRAJECTORY = Path("reports/mpx_trajectory.csv")


def main():
    if not VERDICTS.exists():
        raise SystemExit(f"missing {VERDICTS}; run tools/parse_mpx_logs.py first")
    sizes = set()
    for path in (VERDICTS, TRAJECTORY):
        with path.open(newline="") as handle:
            sizes.update(int(row["size"]) for row in csv.DictReader(handle))
    for size in sorted(sizes):
        subprocess.run(
            [sys.executable, "tools/summarize_mpx.py", "--size", str(size)], check=True
        )


if __name__ == "__main__":
    main()
