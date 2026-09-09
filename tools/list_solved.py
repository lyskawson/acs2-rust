"""Print every run in the archive that reached knowledge 1.0.

The point of the line is to make a lost result obvious: if a run you remember
solving is not here after a sync, its log never made it off the cluster.
"""

import csv
from pathlib import Path

VERDICTS = Path("reports/mpx_verdicts.csv")


def main():
    with VERDICTS.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["verdict"] == "SUCCESS"]
    for row in sorted(rows, key=lambda r: (int(r["size"]), int(r["seed"]))):
        trials = f"{int(row['trials']):,}".replace(",", " ")
        hours = float(row["wall_s"]) / 3600
        print(
            f"    k={row['size']:>3}  seed {row['seed']}  {row['encoding']:<7} "
            f"eps={row['epsilon']:<3} u_max={row['u_max']:<2} "
            f"{trials:>13} trials  {hours:6.1f} h"
        )
    print(f"    {len(rows)} solved runs in the archive")


if __name__ == "__main__":
    main()
