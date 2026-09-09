"""Parse `mpx_reach` stdout logs into tidy CSVs for plotting and review.

Figures must be reproducible from committed data, so the raw logs (which live on
the cluster and on the laptop in several vintages) are reduced here to a stable
schema that lands in `reports/`. Everything downstream reads the CSVs, never the
logs.

Three record types come out of a run:

  trajectory  one row per `--log-trajectory` evaluation point (the S-curve)
  diagnostic  one row per `--log-diagnostics` point (population structure)
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

`encoding` and `epsilon` decide the outcome as strongly as `u_max` does, so a run
without them recorded is not reproducible. The header carries them only from the
commit that added them, so for earlier logs the value is reconstructed -- and
`encoding_source` says which of three ways, because a reconstructed value is not
a measured one:

  header            the log states it; trustworthy
  filename          the log name contains `outcome`/`enc`; the tag convention
                    held for every such run, but the log itself does not say so
  wrapper-default   neither; `slurm/mpx_reach.sh` defaults ENCODING to flip, and
                    reaching `outcome` required a TAG that says so. Reliable, but
                    it is an inference about how the job was submitted, not a
                    record of what ran.

  submission-record the log name says nothing and the wrapper default is wrong;
                    the value comes from KNOWN_ENCODINGS below, which records what
                    the job was actually submitted with

Treat anything but `header` as evidence to re-check before it goes in a paper.

`epsilon` needs no such column. The `--epsilon` flag and the header field landed in
the same commit (`0bc6bd0`), so a log that does not state it had no way to run at
anything but the 0.8 default -- the value is filled in and is not an inference.

Accuracy (`acc:`) and coverage (`cover:`) points are keyed by trial count and
merged into the trajectory row for that evaluation point, since that is what they
are -- the same point measured along another axis.
"""

import argparse
import csv
import re
from pathlib import Path

HEADER = re.compile(r"acs2-bench mpx-reach:\s*(?P<fields>.*)")
CONFIG = re.compile(r"^\s*mpx-(?P<size>\d+) trials_cap=(?P<trials_cap>\d+).*?(?:u_max=(?P<u_max>\d+))?\s*$")
TRAJECTORY = re.compile(r"^\s*mpx-(?P<size>\d+) traj:\s*(?P<fields>.*)")
DIAGNOSTIC = re.compile(r"^\s*mpx-(?P<size>\d+) diag:\s*(?P<fields>.*)")
VERDICT = re.compile(
    r"^\s*mpx-(?P<size>\d+) repeat (?P<repeat>\d+):\s*(?P<verdict>[A-Z-]+)\s*(?P<fields>.*)"
)
ACCURACY = re.compile(r"^\s*mpx-(?P<size>\d+) acc:\s*(?P<fields>.*)")
COVERAGE = re.compile(r"^\s*mpx-(?P<size>\d+) cover:\s*(?P<fields>.*)")
QUADRANT_DETAIL = re.compile(r"^\s*mpx-(?P<size>\d+) qdetail:\s*(?P<fields>.*)")
PROVENANCE = re.compile(r"^run-provenance:\s*(?P<fields>.*)")

REPLAY_COLUMNS = ["er_buffer_size", "er_min_samples", "er_samples_number"]
PROVENANCE_COLUMNS = [
    "encoding", "encoding_source", "epsilon", "agent", "eval_interval", "commit", "tag",
    "do_ga", *REPLAY_COLUMNS,
]
IDENTITY_COLUMNS = ["source", "block", "size", "seed", "variant", "u_max", "repeat"] + PROVENANCE_COLUMNS
COVERAGE_COLUMNS = [
    "a0_nochange", "a0_change", "a1_nochange", "a1_change", "matched_but_wrong",
]
QUADRANT_COLUMNS = [
    f"{cell}_{metric}"
    for cell in ("a0nc", "a0c", "a1nc", "a1c")
    for metric in ("any", "q")
]
TRAJECTORY_COLUMNS = IDENTITY_COLUMNS + [
    "trials", "wall_s", "knowledge", "reliable", "spec", "pop", "accuracy",
] + COVERAGE_COLUMNS + QUADRANT_COLUMNS
DIAGNOSTIC_COLUMNS = IDENTITY_COLUMNS + [
    "trials",
    "micro", "pop_spec", "spec_max", "q_mean", "q_max", "q_above_half",
    "marked", "mark_density", "exp_mean",
    "addr_spec", "addr_random", "addr_full", "correct",
]
VERDICT_COLUMNS = IDENTITY_COLUMNS + [
    "verdict", "trials", "knowledge", "reliable", "spec", "n_bits",
    "peak_macro", "peak_rss_gb", "wall_s", "trials_per_s",
]

# ru_maxrss was read as bytes on Linux until f93b71e, so cluster logs written
# before that print 0.00GB. Zero is not a measurement; it is a missing value.
RSS_FIX_COMMIT_NOTE = "peak_rss_gb=0 on a cluster log means unmeasured, not 0 GB"

# EXPLORE_EPSILON in mpx_reach.rs, and the only reachable value before 0bc6bd0.
DEFAULT_EPSILON = "0.8"


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


# Runs whose encoding is known from the submission record but appears nowhere in
# the log: submitted with ENCODING=outcome under a tag that does not say so, and
# before the header carried the field. Without this they read as `flip`, which is
# how the wrapper defaults -- the exact silent mislabelling `encoding_source`
# exists to expose.
KNOWN_ENCODINGS = {
    "slurm_mpx264_s42_probe264.out": "outcome",
}


def encoding_from_name(source):
    """Reconstruct the encoding of a log written before the header carried it.

    Returns (encoding, how) -- see the module docstring for what `how` means.
    """
    if source in KNOWN_ENCODINGS:
        return KNOWN_ENCODINGS[source], "submission-record"
    lowered = source.lower()
    if "outcome" in lowered or "_enc" in lowered:
        return "outcome", "filename"
    return "flip", "wrapper-default"


class RunContext:
    """Header-scoped state: what every record in this run block inherits."""

    def __init__(self, fields, source="", provenance=None, block=0):
        self.block = block
        self.learning_config = {key: fields.get(key, "") for key in ("do_ga", *REPLAY_COLUMNS)}
        self.base_seed = int(fields.get("seed", 0))
        self.variant = fields.get("alp_gen_variant", "")
        if fields.get("encoding"):
            self.encoding, self.encoding_source = fields["encoding"], "header"
        else:
            self.encoding, self.encoding_source = encoding_from_name(source)
        self.epsilon = fields.get("epsilon", DEFAULT_EPSILON)
        self.agent = fields.get("agent", "")
        self.eval_interval = fields.get("eval_interval", "")
        provenance = provenance or {}
        self.commit = provenance.get("commit", "")
        self.tag = provenance.get("tag", "")
        self.u_max = {}
        self.pending = {}
        self.pending_diagnostics = {}
        self.next_repeat = {}

    def provenance(self):
        return {
            **self.learning_config,
            "encoding": self.encoding,
            "encoding_source": self.encoding_source,
            "epsilon": self.epsilon,
            "agent": self.agent,
            "eval_interval": self.eval_interval,
            "commit": self.commit,
            "tag": self.tag,
        }

    def seed_for(self, repeat):
        return self.base_seed + repeat


def parse_log(path):
    """Return (trajectory_rows, diagnostic_rows, verdict_rows) for one log file."""
    source = path.name
    trajectory_rows, diagnostic_rows, verdict_rows = [], [], []
    provenance = {}
    context = RunContext({}, source)
    block = 0

    def point_for(size, trials):
        """The trajectory point at this trial count, created if new.

        `traj:`, `acc:` and `cover:` are three views of one evaluation point and
        the logs do not emit them in a fixed order, so they are merged by trial
        count rather than by position.
        """
        points = context.pending.setdefault(size, {})
        return points.setdefault(trials, {"trials": trials})

    for line in path.read_text(errors="replace").splitlines():
        marker = PROVENANCE.match(line)
        if marker:
            provenance = parse_fields(marker.group("fields"))
            continue

        header = HEADER.search(line)
        if header:
            block += 1
            context = RunContext(parse_fields(header.group("fields")), source, provenance, block)
            continue

        config = CONFIG.match(line)
        if config and config.group("u_max"):
            context.u_max[config.group("size")] = config.group("u_max")
            continue

        trajectory = TRAJECTORY.match(line)
        if trajectory:
            fields = parse_fields(trajectory.group("fields"))
            spec, _ = parse_spec(fields.get("spec", "0"))
            point_for(trajectory.group("size"), int(fields["trials"])).update({
                "wall_s": float(fields.get("wall", 0)),
                "knowledge": float(fields["knowledge"]),
                "reliable": int(fields["reliable"]),
                "spec": spec,
                "pop": int(fields["pop"]),
            })
            continue

        accuracy = ACCURACY.match(line)
        if accuracy:
            fields = parse_fields(accuracy.group("fields"))
            point_for(accuracy.group("size"), int(fields["trials"]))["accuracy"] = float(
                fields["accuracy"]
            )
            continue

        coverage = COVERAGE.match(line)
        if coverage:
            fields = parse_fields(coverage.group("fields"))
            point = point_for(coverage.group("size"), int(fields["trials"]))
            for key in COVERAGE_COLUMNS:
                if key in fields:
                    point[key] = fields[key]
            continue

        detail = QUADRANT_DETAIL.match(line)
        if detail:
            fields = parse_fields(detail.group("fields"))
            point = point_for(detail.group("size"), int(fields["trials"]))
            point.update({key: fields[key] for key in QUADRANT_COLUMNS if key in fields})
            continue

        diagnostic = DIAGNOSTIC.match(line)
        if diagnostic:
            size = diagnostic.group("size")
            fields = parse_fields(diagnostic.group("fields"))
            context.pending_diagnostics.setdefault(size, []).append({
                key: fields.get(key, "")
                for key in DIAGNOSTIC_COLUMNS
                if key not in IDENTITY_COLUMNS
            })
            continue

        verdict = VERDICT.match(line)
        if verdict:
            size = verdict.group("size")
            repeat = int(verdict.group("repeat"))
            fields = parse_fields(verdict.group("fields"))
            spec, n_bits = parse_spec(fields.get("spec", "0"))
            shared = identity(context, source, size, repeat)
            wall = float(fields.get("wall", 0))
            trials = int(fields["trials"])
            verdict_rows.append({
                **shared,
                "verdict": verdict.group("verdict"),
                "trials": trials,
                "knowledge": float(fields["knowledge"]),
                "reliable": int(fields["reliable"]),
                "spec": spec,
                "n_bits": n_bits if n_bits is not None else "",
                "peak_macro": int(fields.get("peak_macro", 0)),
                "peak_rss_gb": float(fields.get("peak_rss", "0").rstrip("GB")),
                "wall_s": wall,
                "trials_per_s": round(trials / wall, 2) if wall else "",
            })
            for point in context.pending.pop(size, {}).values():
                trajectory_rows.append({**shared, **point})
            for point in context.pending_diagnostics.pop(size, []):
                diagnostic_rows.append({**shared, **point})
            context.next_repeat[size] = repeat + 1
            continue

    # A run still in flight (or scancelled) leaves points unclosed. They are the
    # live state of the cluster and must survive into the archive, so they are
    # attributed to the repeat the next verdict line would have closed.
    for size, points in context.pending_diagnostics.items():
        shared = identity(context, source, size, context.next_repeat.get(size, 0))
        for point in points:
            diagnostic_rows.append({**shared, **point})

    for size, points in context.pending.items():
        shared = identity(context, source, size, context.next_repeat.get(size, 0))
        for point in points.values():
            trajectory_rows.append({**shared, **point})

    return trajectory_rows, diagnostic_rows, verdict_rows


def identity(context, source, size, repeat):
    """The columns every record in a run block carries, provenance included."""
    return {
        "source": source,
        "block": context.block,
        "size": int(size),
        "seed": context.seed_for(repeat),
        "variant": context.variant,
        "u_max": context.u_max.get(size, ""),
        "repeat": repeat,
        **context.provenance(),
    }


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
    parser.add_argument("--diagnostic-csv", type=Path, default=Path("reports/mpx_diagnostics.csv"))
    parser.add_argument("--verdict-csv", type=Path, default=Path("reports/mpx_verdicts.csv"))
    args = parser.parse_args()

    trajectory_rows, diagnostic_rows, verdict_rows = [], [], []
    for log in args.logs:
        if not log.is_file():
            raise SystemExit(f"not a file: {log}")
        trajectory, diagnostics, verdicts = parse_log(log)
        trajectory_rows.extend(trajectory)
        diagnostic_rows.extend(diagnostics)
        verdict_rows.extend(verdicts)

    sort_key = lambda row: (
        row["size"], row["encoding"], row["variant"], row["seed"],
        row["source"], row["trials"],
    )
    write_csv(args.trajectory_csv, TRAJECTORY_COLUMNS, sorted(trajectory_rows, key=sort_key))
    write_csv(args.diagnostic_csv, DIAGNOSTIC_COLUMNS, sorted(diagnostic_rows, key=sort_key))
    write_csv(args.verdict_csv, VERDICT_COLUMNS, sorted(verdict_rows, key=sort_key))


if __name__ == "__main__":
    main()
