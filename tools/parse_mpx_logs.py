"""Parse `mpx_reach` stdout logs into two tidy CSVs for plotting and review.

Figures must be reproducible from committed data, so the raw logs (which live on
the cluster and on the laptop in several vintages) are reduced here to a stable
schema that lands in `reports/`. Everything downstream reads the CSVs, never the
logs.

Two record types come out of a run:

  trajectory  one row per `--log-trajectory` evaluation point (the S-curve)
  verdict     one row per repeat, the terminal SUCCESS / TIME-LIMITED / ... line

Three details of the log format that the state machine exists to handle:

  * Repeat r runs with `seed = base_seed + r` (mpx_reach.rs), so an `n_exp=3`
    log at seed 42 actually holds seeds 42, 43 and 44 -- they must not be
    collapsed into one series.
  * Trajectory lines carry no repeat index. They are attributed to the repeat
    that the next verdict line closes; a trajectory tail with no verdict line
    (a run still in flight, or one killed by scancel) is emitted as the next,
    unfinished repeat.
  * Older logs predate `u_max` and `alp_gen_variant`, and a single file may hold
    several runs separated by banner lines. A header line resets the context.
"""

import argparse
import csv
import re
from pathlib import Path

HEADER = re.compile(r"acs2-bench mpx-reach:\s*(?P<fields>.*)")
CONFIG = re.compile(r"^\s*mpx-(?P<size>\d+) trials_cap=(?P<trials_cap>\d+).*?(?:u_max=(?P<u_max>\d+))?\s*$")
TRAJECTORY = re.compile(r"^\s*mpx-(?P<size>\d+) traj:\s*(?P<fields>.*)")
VERDICT = re.compile(
    r"^\s*mpx-(?P<size>\d+) repeat (?P<repeat>\d+):\s*(?P<verdict>[A-Z-]+)\s*(?P<fields>.*)"
)

TRAJECTORY_COLUMNS = [
    "source", "size", "seed", "variant", "u_max", "repeat",
    "trials", "wall_s", "knowledge", "reliable", "spec", "pop",
]
VERDICT_COLUMNS = [
    "source", "size", "seed", "variant", "u_max", "repeat", "verdict",
    "trials", "knowledge", "reliable", "spec", "n_bits",
    "peak_macro", "peak_rss_gb", "wall_s",
]


def parse_fields(text):
    """Split `k=v k=v` into a dict, stripping the unit suffixes the logs carry."""
    fields = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        fields[key] = value.rstrip("s") if key in ("wall", "time_cap") else value
    return fields


def parse_spec(value):
    """`spec=6.03/38` on verdict lines, bare `spec=7.04` on trajectory lines."""
    head, _, tail = value.partition("/")
    return float(head), (int(tail) if tail else None)


class RunContext:
    """Header-scoped state: what every record in this run block inherits."""

    def __init__(self, fields):
        self.base_seed = int(fields.get("seed", 0))
        self.variant = fields.get("alp_gen_variant", "")
        self.u_max = {}
        self.pending = {}
        self.next_repeat = {}

    def seed_for(self, repeat):
        return self.base_seed + repeat


def parse_log(path):
    """Return (trajectory_rows, verdict_rows) for one log file."""
    source = path.name
    trajectory_rows, verdict_rows = [], []
    context = RunContext({})

    for line in path.read_text(errors="replace").splitlines():
        header = HEADER.search(line)
        if header:
            context = RunContext(parse_fields(header.group("fields")))
            continue

        config = CONFIG.match(line)
        if config and config.group("u_max"):
            context.u_max[config.group("size")] = config.group("u_max")
            continue

        trajectory = TRAJECTORY.match(line)
        if trajectory:
            size = trajectory.group("size")
            fields = parse_fields(trajectory.group("fields"))
            spec, _ = parse_spec(fields.get("spec", "0"))
            context.pending.setdefault(size, []).append({
                "trials": int(fields["trials"]),
                "wall_s": float(fields.get("wall", 0)),
                "knowledge": float(fields["knowledge"]),
                "reliable": int(fields["reliable"]),
                "spec": spec,
                "pop": int(fields["pop"]),
            })
            continue

        verdict = VERDICT.match(line)
        if verdict:
            size = verdict.group("size")
            repeat = int(verdict.group("repeat"))
            fields = parse_fields(verdict.group("fields"))
            spec, n_bits = parse_spec(fields.get("spec", "0"))
            shared = {
                "source": source,
                "size": int(size),
                "seed": context.seed_for(repeat),
                "variant": context.variant,
                "u_max": context.u_max.get(size, ""),
                "repeat": repeat,
            }
            verdict_rows.append({
                **shared,
                "verdict": verdict.group("verdict"),
                "trials": int(fields["trials"]),
                "knowledge": float(fields["knowledge"]),
                "reliable": int(fields["reliable"]),
                "spec": spec,
                "n_bits": n_bits if n_bits is not None else "",
                "peak_macro": int(fields.get("peak_macro", 0)),
                "peak_rss_gb": float(fields.get("peak_rss", "0").rstrip("GB")),
                "wall_s": float(fields.get("wall", 0)),
            })
            for point in context.pending.pop(size, []):
                trajectory_rows.append({**shared, **point})
            context.next_repeat[size] = repeat + 1
            continue

    # A run still in flight (or scancelled) leaves trajectory points unclosed.
    for size, points in context.pending.items():
        repeat = context.next_repeat.get(size, 0)
        shared = {
            "source": source,
            "size": int(size),
            "seed": context.seed_for(repeat),
            "variant": context.variant,
            "u_max": context.u_max.get(size, ""),
            "repeat": repeat,
        }
        for point in points:
            trajectory_rows.append({**shared, **point})

    return trajectory_rows, verdict_rows


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{path}: {len(rows)} rows")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="mpx_reach log files (.log/.out/.txt)")
    parser.add_argument("--trajectory-csv", type=Path, default=Path("reports/mpx_trajectory.csv"))
    parser.add_argument("--verdict-csv", type=Path, default=Path("reports/mpx_verdicts.csv"))
    args = parser.parse_args()

    trajectory_rows, verdict_rows = [], []
    for log in args.logs:
        if not log.is_file():
            raise SystemExit(f"not a file: {log}")
        trajectory, verdicts = parse_log(log)
        trajectory_rows.extend(trajectory)
        verdict_rows.extend(verdicts)

    sort_key = lambda row: (row["size"], row["variant"], row["seed"], row["trials"])
    write_csv(args.trajectory_csv, TRAJECTORY_COLUMNS, sorted(trajectory_rows, key=sort_key))
    write_csv(args.verdict_csv, VERDICT_COLUMNS, sorted(verdict_rows, key=sort_key))


if __name__ == "__main__":
    main()
