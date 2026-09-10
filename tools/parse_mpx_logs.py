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

  * A checkpointed run spans several jobs, each writing its own log. `source` is then
    the run and `segment` the job that recorded the row; a resumed job re-runs whatever
    the previous one did after its last checkpoint, so the later segment supersedes the
    earlier one above the trial it resumed at, and only the closing segment's verdict is
    the run's. Without this a chain reads as several independent runs sharing a seed --
    exactly what plot_mpx.py refuses.
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
from datetime import datetime
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
SEGMENT = re.compile(r"^run-segment:\s*(?P<fields>.*)")
RESUMED = re.compile(r"^\s*mpx-(?P<size>\d+) resumed:\s*(?P<fields>.*)")

REPLAY_COLUMNS = ["er_buffer_size", "er_min_samples", "er_samples_number"]
PROVENANCE_COLUMNS = [
    "encoding", "encoding_source", "epsilon", "agent", "eval_interval", "commit", "tag",
    "do_ga", *REPLAY_COLUMNS, "strict_resource_limits", "rss_scope",
]
IDENTITY_COLUMNS = [
    "source", "segment", "block", "size", "seed", "variant", "u_max", "repeat",
] + PROVENANCE_COLUMNS
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
    "peak_macro", "peak_rss_gb", "wall_s", "trials_per_s", "knowledge_trials", "knowledge_status",
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
        self.learning_config = {key: fields.get(key, "") for key in ("do_ga", *REPLAY_COLUMNS, "strict_resource_limits", "rss_scope")}
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
        # A checkpointed run spans several jobs, each with its own log. `run_name` is the
        # run; `segment_file` is the job. Empty for every log written before checkpointing.
        self.run_name = ""
        self.segment_file = ""
        self.resume_at = {}
        self.sizes_seen = set()
        self.job = (provenance or {}).get("job", "")
        self.attempt = (provenance or {}).get("attempt", "0")
        self.started = (provenance or {}).get("started", "")

    def segment_of(self, base, source):
        self.run_name = base
        self.segment_file = source

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
    """Return (trajectory_rows, diagnostic_rows, verdict_rows, segments) for one log file.

    `segments` is emitted whether or not the job recorded anything. A job that resumed
    and stopped before its next evaluation has no rows at all, and its resume point is
    still what invalidates the measurements the job before it took after its last
    checkpoint.
    """
    source = path.name
    trajectory_rows, diagnostic_rows, verdict_rows = [], [], []
    provenance = {}
    segment = {}
    sizes_seen = set()
    context = RunContext({}, source)
    context.sizes_seen = sizes_seen
    block = 0

    def flush_pending():
        for size, points in context.pending_diagnostics.items():
            shared = identity(context, source, size, context.next_repeat.get(size, 0))
            for point in points:
                diagnostic_rows.append({**shared, **point})

        for size, points in context.pending.items():
            shared = identity(context, source, size, context.next_repeat.get(size, 0))
            for point in points.values():
                trajectory_rows.append({**shared, **point})
        context.pending.clear()
        context.pending_diagnostics.clear()

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

        chained = SEGMENT.match(line)
        if chained:
            # Flush first: rows pending from an earlier block belong to the run they were
            # measured under, not to the one this marker names.
            flush_pending()
            if segment:
                # One job, one log. Boundaries are built once per file from the context
                # that survives to EOF, so a second segment in the same file would emit
                # its own boundaries under the last one's name and silently drop the
                # first's rows. Rejecting is right rather than clever: sync_runs.sh
                # copies each job's log as its own file and nothing produces this.
                raise ValueError(
                    f"{source}: two run-segment markers in one log; a checkpointed job "
                    "writes its own file"
                )
            segment.update(parse_fields(chained.group("fields")))
            context.segment_of(segment.get("base", ""), source)
            continue

        resumed = RESUMED.match(line)
        if resumed:
            fields = parse_fields(resumed.group("fields"))
            context.resume_at[resumed.group("size")] = int(fields.get("trials", 0))
            continue

        header = HEADER.search(line)
        if header:
            flush_pending()
            block += 1
            context = RunContext(parse_fields(header.group("fields")), source, provenance, block)
            context.sizes_seen = sizes_seen
            if segment.get("base"):
                context.segment_of(segment["base"], source)
            continue

        config = CONFIG.match(line)
        if config:
            sizes_seen.add(config.group("size"))
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
            if repeat < context.next_repeat.get(size, 0):
                raise ValueError(f"duplicate/out-of-order repeat {repeat} in {source}, block {context.block}")
            fields = parse_fields(verdict.group("fields"))
            spec, n_bits = parse_spec(fields.get("spec", "0"))
            shared = identity(context, source, size, repeat)
            wall = float(fields.get("wall", 0))
            trials = int(fields["trials"])
            knowledge = fields["knowledge"]
            knowledge_trials = ""
            if knowledge == "unmeasured":
                knowledge = ""
                knowledge_status = "unmeasured"
            else:
                knowledge = float(knowledge)
                if fields.get("knowledge_trials", "").isdigit():
                    knowledge_trials = int(fields["knowledge_trials"])
                elif verdict.group("verdict") == "SUCCESS":
                    knowledge_trials = trials
                else:
                    measured = [point["trials"] for point in context.pending.get(size, {}).values()
                                if "knowledge" in point and point["trials"] <= trials]
                    if measured:
                        knowledge_trials = max(measured)
                knowledge_status = ("legacy-unverified" if knowledge_trials == "" else
                                    "at-verdict" if knowledge_trials == trials else "stale")
            verdict_rows.append({
                **shared,
                "verdict": verdict.group("verdict"),
                "trials": trials,
                "knowledge": knowledge,
                "knowledge_trials": knowledge_trials,
                "knowledge_status": knowledge_status,
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

    flush_pending()

    segments = []
    if context.run_name:
        if block > 1:
            raise ValueError(
                f"{source}: a checkpointed log carries {block} header blocks; a "
                "checkpointed job runs one size once and writes its own file"
            )
        for size in sorted(sizes_seen):
            # `resumed=yes` from the wrapper and the binary's `resumed:` line must agree,
            # per size. A segment that resumed but whose resume point went unrecorded
            # reads as a restart from zero, and a restart from zero discards the whole
            # run's history -- silently, which is the one thing this archive must not do.
            if segment.get("resumed") == "yes" and size not in context.resume_at:
                raise ValueError(
                    f"{source}: the wrapper recorded resumed=yes but the log carries no "
                    f"`resumed:` line for size {size}, so the trial it resumed at is unknown"
                )
            segments.append({
                "source": context.run_name,
                "size": int(size),
                "segment": source,
                "resume_at": context.resume_at.get(size, 0),
                "started": context.started,
                "order": segment_order(context.job, context.attempt, source),
            })

    return trajectory_rows, diagnostic_rows, verdict_rows, segments


def started_instant(started):
    """The moment a job began, as an absolute instant, or None if it cannot be read.

    `date -Is` writes local time with an offset, so comparing the strings is wrong across a
    daylight-saving change -- `...T02:30:00+02:00` sorts after `...T02:00:00+01:00` although
    the second is half an hour later. Poland changes clocks on 2026-10-25 and the grant runs
    into 2027, so a k=264 chain reaches it.
    """
    if not started:
        return None
    try:
        return datetime.fromisoformat(started).timestamp()
    except ValueError:
        return None


def segment_order(job, attempt, source):
    """Tiebreak within one instant: attempts of a requeued job share its id and can share a
    second. Job ids are not a chronology on their own -- SLURM wraps them from MaxJobId back
    to FirstJobId -- so they only ever separate segments the clock cannot."""
    return (
        int(job) if job.isdigit() else -1,
        int(attempt) if attempt.isdigit() else -1,
        source,
    )


def identity(context, source, size, repeat):
    """The columns every record in a run block carries, provenance included.

    For a checkpointed run `source` is the run's stable name and `segment` is the job
    that wrote the record; for every other log `source` is the file and `segment` empty.
    Downstream tools key a run on `source`, so a chain has to collapse to one value there
    or it reads as several independent runs that happen to share a seed.
    """
    return {
        "source": context.run_name or source,
        "segment": context.segment_file,
        "_resume_at": context.resume_at.get(size, 0),
        "block": context.block,
        "size": int(size),
        "seed": context.seed_for(repeat),
        "variant": context.variant,
        "u_max": context.u_max.get(size, ""),
        "repeat": repeat,
        **context.provenance(),
    }


def run_key(row):
    """A run, across the files its jobs wrote.

    `block` is deliberately absent: it counts header blocks within one file, so keeping it
    would stop a run's segments joining each other.
    """
    return (row["source"], row["size"], row["seed"], row["repeat"])


def boundaries_by_run(segments):
    """A run's segments, in the order its jobs ran.

    Ordering here is destructive -- a segment discards every measurement above the trial it
    resumed at -- so it is settled by an actual clock or not at all. Every segment the
    wrapper writes carries `started`; a run whose segments do not all carry a readable one
    has no reliable order, and guessing has twice produced exactly the silent loss this
    exists to prevent: job ids wrap, and a missing timestamp sorts ahead of every real one.
    Refusing hands the operator a decision instead of making a destructive one for them.
    """
    grouped = {}
    for segment in segments:
        grouped.setdefault((segment["source"], segment["size"]), []).append(segment)

    for (source, size), group in grouped.items():
        instants = {segment["segment"]: started_instant(segment.get("started")) for segment in group}
        unreadable = sorted(name for name, instant in instants.items() if instant is None)
        if unreadable:
            raise ValueError(
                f"{source} (size {size}): cannot order the run's jobs -- no readable "
                f"`started` on {', '.join(unreadable)}. Ordering decides which segment's "
                "measurements survive, so it is not guessed."
            )
        group.sort(key=lambda segment: (instants[segment["segment"]], segment["order"]))
    return grouped


def stitch_segments(rows, key_field, segments):
    """Collapse a checkpointed run's per-job records into the one run they describe.

    A resumed job re-runs whatever the previous one did after its last checkpoint, so the
    same trial can be recorded twice and the earlier record is of work that was discarded.
    Walking the run's jobs in chronological order, each discards every record above the
    trial it resumed at before its own records go in.

    The walk is over the *segments*, not over the rows: a job that resumed and stopped
    before recording anything still invalidates what the job before it measured after its
    last checkpoint.
    """
    plain = [row for row in rows if not row["segment"]]
    chained = [row for row in rows if row["segment"]]
    if not chained:
        return rows

    by_run = {}
    for row in chained:
        by_run.setdefault(run_key(row), {}).setdefault(row["segment"], []).append(row)

    stitched = []
    for key, per_segment in by_run.items():
        source, size = key[0], key[1]
        kept = {}
        for segment in boundaries_by_run(segments).get((source, size), []):
            resume_at = segment["resume_at"]
            for trials in [trials for trials in kept if trials > resume_at]:
                del kept[trials]
            for row in per_segment.get(segment["segment"], []):
                # Diagnostic rows carry `trials` as text and trajectory rows as an int;
                # the key has to be one type or the comparison above raises.
                kept[int(row[key_field])] = row
        stitched.extend(kept.values())

    return plain + stitched


def close_chained_runs(rows, segments, progress=()):
    """One verdict per run, not one per job.

    Intermediate jobs stop on their own wall clock and record TIME-LIMITED. That says a
    job stopped, not that the run did. The run's verdict is the last one its jobs recorded
    in chronological order -- not the largest trial count, because a job killed after its
    final checkpoint reports more trials than the job that legitimately superseded it.
    """
    plain = [row for row in rows if not row["segment"]]
    chained = [row for row in rows if row["segment"]]
    if not chained:
        return rows

    by_run = {}
    for row in chained:
        by_run.setdefault(run_key(row), {})[row["segment"]] = row

    furthest = {}
    for row in progress:
        if not row["segment"]:
            continue
        key = run_key(row)
        furthest[key] = max(furthest.get(key, -1), int(row["trials"]))

    grouped = boundaries_by_run(segments)
    closing = []
    for key, per_segment in by_run.items():
        held = None
        for boundary in grouped.get((key[0], key[1]), []):
            # A verdict describes the run as it stood when a job stopped. A later job that
            # resumed below that trial -- a restart from zero after a deleted checkpoint,
            # say -- superseded the work the verdict described, so it must not survive
            # merely because the job that replaced it recorded no verdict of its own.
            if held is not None and held["trials"] > boundary["resume_at"]:
                held = None
            held = per_segment.get(boundary["segment"], held)
        if held is not None and held["trials"] < furthest.get(key, -1):
            # A later job carried the run past the trial this verdict describes. The
            # verdict says a *job* stopped; the run did not, and rendering it as the run's
            # final state would report a mid-chain reading as where the run ended.
            held = None
        if held is not None:
            closing.append(held)
    return plain + closing


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
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

    trajectory_rows, diagnostic_rows, verdict_rows, segments = [], [], [], []
    if len({path.name for path in args.logs}) != len(args.logs):
        raise SystemExit("duplicate log basenames would collide; archive each source once under a unique name")
    for log in args.logs:
        if not log.is_file():
            raise SystemExit(f"not a file: {log}")
        trajectory, diagnostics, verdicts, found = parse_log(log)
        trajectory_rows.extend(trajectory)
        diagnostic_rows.extend(diagnostics)
        verdict_rows.extend(verdicts)
        segments.extend(found)

    trajectory_rows = stitch_segments(trajectory_rows, "trials", segments)
    diagnostic_rows = stitch_segments(diagnostic_rows, "trials", segments)
    verdict_rows = close_chained_runs(verdict_rows, segments, trajectory_rows)

    sort_key = lambda row: (
        row["size"], row["encoding"], row["variant"], row["seed"],
        row["source"], row["trials"],
    )
    write_csv(args.trajectory_csv, TRAJECTORY_COLUMNS, sorted(trajectory_rows, key=sort_key))
    write_csv(args.diagnostic_csv, DIAGNOSTIC_COLUMNS, sorted(diagnostic_rows, key=sort_key))
    write_csv(args.verdict_csv, VERDICT_COLUMNS, sorted(verdict_rows, key=sort_key))


if __name__ == "__main__":
    main()
