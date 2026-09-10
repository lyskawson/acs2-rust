import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from parse_mpx_logs import (
    close_chained_runs, parse_log, stitch_segments, TRAJECTORY_COLUMNS, VERDICT_COLUMNS,
)
from summarize_mpx import collect, render


class ArchiveTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.out"
            path.write_text(content)
            return parse_log(path)

    def segment(self, base, job, resumed_at, points, verdict_trials):
        """One job of a checkpointed run, as slurm/mpx_reach.sh writes it."""
        lines = [
            f"run-provenance: commit=abc job={job} tag=k264 size=20 seed=42",
            f"run-segment: base={base} checkpoint=/runs/{base}.ckpt checkpoint_every=4000 "
            f"resumed={'yes' if resumed_at else 'no'}",
            "acs2-bench mpx-reach: seed=42 alp_gen_variant=pyalcs do_ga=true encoding=flip",
            "mpx-20 trials_cap=1000000 u_max=6",
        ]
        if resumed_at:
            lines.append(f"  mpx-20 resumed: trials={resumed_at} time={resumed_at} since_eval=0")
        for trials, knowledge in points:
            lines.append(
                f"  mpx-20 traj: trials={trials} knowledge={knowledge} reliable=1 spec=6 pop=9"
            )
        lines.append(
            f"  mpx-20 repeat 0: TIME-LIMITED trials={verdict_trials} knowledge=unmeasured "
            "knowledge_trials=unmeasured reliable=1 spec=6/21 wall=10s"
        )
        return "\n".join(lines) + "\n"

    def parse_named(self, name, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / name
            path.write_text(content)
            return parse_log(path)

    def chain(self, segments):
        trajectory, diagnostics, verdicts = [], [], []
        for name, content in segments:
            one, two, three = self.parse_named(name, content)
            trajectory.extend(one)
            diagnostics.extend(two)
            verdicts.extend(three)
        return (
            stitch_segments(trajectory, "trials"),
            close_chained_runs(verdicts),
        )

    def test_a_chained_run_is_one_run_not_one_per_job(self):
        base = "slurm_mpx20_s42_k264"
        trajectory, verdicts = self.chain([
            (f"{base}_seg1.out", self.segment(base, 1, 0, [(2000, "0.1"), (4000, "0.2")], 4000)),
            (f"{base}_seg2.out", self.segment(base, 2, 4000, [(6000, "0.3")], 6000)),
            (f"{base}_seg3.out", self.segment(base, 3, 6000, [(8000, "0.4")], 8000)),
        ])

        self.assertEqual(len(verdicts), 1, "three jobs, one run, one verdict")
        self.assertEqual(verdicts[0]["trials"], 8000)
        self.assertEqual(verdicts[0]["segment"], f"{base}_seg3.out")
        self.assertEqual({row["source"] for row in trajectory}, {base})
        self.assertEqual(
            sorted(row["trials"] for row in trajectory), [2000, 4000, 6000, 8000]
        )

    def test_work_a_killed_job_did_after_its_last_checkpoint_is_superseded(self):
        # The killed job logged 6000 and 8000, but its checkpoint holds 4000, so the
        # resumed job re-ran those trials. The earlier records are of discarded work.
        base = "slurm_mpx20_s42_k264"
        trajectory, verdicts = self.chain([
            (
                f"{base}_seg1.out",
                self.segment(base, 1, 0, [(4000, "0.2"), (6000, "0.9"), (8000, "0.9")], 8000),
            ),
            (
                f"{base}_seg2.out",
                self.segment(base, 2, 4000, [(6000, "0.3"), (8000, "0.4")], 8000),
            ),
        ])

        measured = {row["trials"]: row["knowledge"] for row in trajectory}
        self.assertEqual(sorted(measured), [4000, 6000, 8000])
        self.assertEqual(measured[4000], 0.2)
        self.assertEqual(measured[6000], 0.3, "the discarded 0.9 must not survive")
        self.assertEqual(measured[8000], 0.4, "the discarded 0.9 must not survive")
        self.assertEqual(len(verdicts), 1)
        self.assertEqual(
            verdicts[0]["segment"],
            f"{base}_seg2.out",
            "the run's verdict is the last segment's by resume point, not by trial count",
        )

    def test_an_unchained_log_is_untouched_by_stitching(self):
        trajectory, _, verdicts = self.parse(
            "acs2-bench mpx-reach: seed=42 alp_gen_variant=pyalcs do_ga=true\n"
            "mpx-20 traj: trials=500 knowledge=0.5 reliable=2 pop=3\n"
            "mpx-20 repeat 0: TIME-LIMITED trials=500 knowledge=0.5 reliable=2\n"
        )
        self.assertEqual([row["segment"] for row in trajectory], [""])
        self.assertEqual(stitch_segments(trajectory, "trials"), trajectory)
        self.assertEqual(close_chained_runs(verdicts), verdicts)

    def test_identical_headers_and_variants_have_distinct_runs(self):
        blocks = []
        for variant in ("pyalcs", "butz", "butz"):
            blocks.append(
                f"acs2-bench mpx-reach: seed=42 alp_gen_variant={variant} do_ga=true\n"
                "mpx-37 trials_cap=1000 u_max=8\n"
                "mpx-37 repeat 0: SUCCESS trials=1000 knowledge=1 reliable=2 spec=6/38 wall=1s\n"
            )
        trajectory, _, verdicts = self.parse("".join(blocks))
        records = collect(trajectory, verdicts, 37)
        self.assertEqual(len(records), 3)
        self.assertEqual([row["block"] for row in records], [1, 2, 3])
        self.assertIn("| seed | variant |", render(records, 37))

    def test_replay_and_quadrant_detail_survive_in_every_record_type(self):
        trajectory, diagnostics, verdicts = self.parse(
            "acs2-bench mpx-reach: seed=42 agent=acs2er do_ga=true "
            "er_buffer_size=10000 er_min_samples=1000 er_samples_number=1\n"
            "mpx-70 qdetail: trials=500 a0nc_any=0.9457 a0nc_q=0.823\n"
            "mpx-70 traj: trials=500 knowledge=0.5 reliable=2 pop=3\n"
            "mpx-70 diag: trials=500 micro=3\n"
            "mpx-70 repeat 0: TIME-LIMITED trials=500 knowledge=0.5 reliable=2\n"
        )
        self.assertEqual(len(trajectory), 1)
        self.assertEqual(trajectory[0]["a0nc_q"], "0.823")
        for row in (trajectory[0], diagnostics[0], verdicts[0]):
            self.assertEqual(row["do_ga"], "true")
            self.assertEqual(row["er_buffer_size"], "10000")
            self.assertEqual(row["er_min_samples"], "1000")
            self.assertEqual(row["er_samples_number"], "1")

    def test_missing_header_values_are_not_invented(self):
        _, _, rows = self.parse(
            "acs2-bench mpx-reach: seed=42\n"
            "mpx-37 repeat 0: SUCCESS trials=500 knowledge=1 reliable=1\n"
        )
        self.assertEqual(rows[0]["do_ga"], "")
        self.assertEqual(rows[0]["er_samples_number"], "")
        self.assertEqual(rows[0]["encoding_source"], "wrapper-default")

    def test_header_flushes_unfinished_trajectory_and_diagnostics(self):
        trajectory, diagnostics, verdicts = self.parse(
            "acs2-bench mpx-reach: seed=42 alp_gen_variant=pyalcs\n"
            "mpx-37 trials_cap=1000 u_max=7\n"
            "mpx-37 traj: trials=500 knowledge=0.5 reliable=2 pop=3\n"
            "mpx-37 diag: trials=500 micro=3\n"
            "acs2-bench mpx-reach: seed=90 alp_gen_variant=butz\n"
            "mpx-37 trials_cap=1000 u_max=8\n"
            "mpx-37 traj: trials=500 knowledge=0.2 reliable=1 pop=2\n"
        )
        self.assertEqual([(r["seed"], r["block"], r["u_max"]) for r in trajectory],
                         [(42, 1, "7"), (90, 2, "8")])
        self.assertEqual(diagnostics[0]["seed"], 42)
        self.assertEqual(verdicts, [])

    def test_verdict_snapshot_provenance_and_missing_measurement(self):
        trajectory, _, verdicts = self.parse(
            "acs2-bench mpx-reach: seed=42\n"
            "mpx-37 traj: trials=500 knowledge=0.5 reliable=2 pop=3\n"
            "mpx-37 acc: trials=500 accuracy=0.9\n"
            "mpx-37 cover: trials=500 a0_nochange=0.7\n"
            "mpx-37 repeat 0: TIME-LIMITED trials=1000 knowledge=0.5 reliable=4\n"
            "mpx-37 repeat 1: TIME-LIMITED trials=500 knowledge=unmeasured "
            "knowledge_trials=unmeasured reliable=3\n"
            "mpx-37 repeat 2: SUCCESS trials=1500 knowledge=1 reliable=3\n"
        )
        self.assertEqual(verdicts[0]["knowledge_trials"], 500)
        self.assertEqual(verdicts[0]["knowledge_status"], "stale")
        self.assertEqual(verdicts[1]["knowledge"], "")
        self.assertEqual(verdicts[1]["knowledge_status"], "unmeasured")
        self.assertEqual(verdicts[2]["knowledge_trials"], 1500)
        self.assertEqual(verdicts[2]["knowledge_status"], "at-verdict")
        row = collect(trajectory, verdicts, 37)[0]
        self.assertNotIn("accuracy", row)
        self.assertNotIn("a0_nochange", row)

    def test_selection_separates_replay_arms(self):
        from mpx_selection import arm_of, selected
        base = {"source": "run.out", "agent": "acs2er", "encoding": "flip", "epsilon": "1.0",
                "u_max": "8", "do_ga": "true", "er_buffer_size": "10000",
                "er_min_samples": "1000", "er_samples_number": "1"}
        for key, value in (("agent", "acs2"), ("do_ga", "false"),
                           ("er_buffer_size", "100"), ("er_min_samples", "10"),
                           ("er_samples_number", "3")):
            changed = {**base, key: value}
            self.assertNotEqual(arm_of(base), arm_of(changed))
            self.assertFalse(selected(changed, {key: base[key]}))
        self.assertTrue(selected(base, {"epsilon": "1"}, ["run.out"]))
        self.assertFalse(selected(base, sources=["other.out"]))

    def test_all_archived_verdicts_are_rendered(self):
        repo = Path(__file__).resolve().parents[1]
        with (repo / "reports/mpx_verdicts.csv").open() as handle:
            verdicts = list(csv.DictReader(handle))
        with (repo / "reports/mpx_trajectory.csv").open() as handle:
            trajectory = list(csv.DictReader(handle))
        for size in {int(row["size"]) for row in verdicts}:
            rows = collect(trajectory, verdicts, size)
            expected = sum(int(row["size"]) == size for row in verdicts)
            self.assertEqual(sum(row["final"] for row in rows), expected)
        solved37 = [row for row in collect(trajectory, verdicts, 37)
                    if row["state"] == "SUCCESS"]
        self.assertEqual(len(solved37), sum(row["size"] == "37" and row["verdict"] == "SUCCESS"
                                           for row in verdicts))

    def test_new_size_gets_a_table_before_its_first_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "reports").mkdir()
            for filename in ("rebuild_tables.py", "summarize_mpx.py", "mpx_selection.py"):
                shutil.copy(Path(__file__).with_name(filename), root / "tools")
            for filename, columns, rows in (
                ("mpx_verdicts.csv", VERDICT_COLUMNS, []),
                ("mpx_trajectory.csv", TRAJECTORY_COLUMNS, [{
                    "source": "first.out", "block": 1, "size": 521, "seed": 42, "repeat": 0,
                    "variant": "pyalcs", "trials": 500, "knowledge": 0.2, "pop": 5,
                }]),
            ):
                with (root / "reports" / filename).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(rows)
            subprocess.run([sys.executable, "tools/rebuild_tables.py"], cwd=root,
                           capture_output=True, text=True, check=True)
            rendered = (root / "reports/MPX521_runs.md").read_text()
            self.assertIn("| running |", rendered)
            self.assertIn("0.2000", rendered)

    def test_sync_commits_an_untracked_header_only_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tools").mkdir()
            (root / "reports").mkdir()
            (root / "bin").mkdir()
            shutil.copy(Path(__file__).with_name("sync_runs.sh"), root / "tools")
            fake_python = root / "bin/python3"
            fake_python.write_text("#!/bin/sh\nexit 0\n")
            fake_python.chmod(0o755)
            env = {**os.environ, "PATH": f"{root / 'bin'}:{os.environ['PATH']}"}
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=root, text=True,
                                               stderr=subprocess.STDOUT)
            git("init", "-q")
            git("config", "user.name", "Archive Test")
            git("config", "user.email", "archive-test@example.invalid")
            git("add", "tools")
            git("commit", "-qm", "fixture")
            (root / "reports/slurm_header.out").write_text("acs2-bench mpx-reach: seed=42\n")
            result = subprocess.run(["bash", "tools/sync_runs.sh", "--local", "--commit"],
                                    cwd=root, env=env, text=True, capture_output=True, check=True)
            self.assertIn("==> committed", result.stdout)
            self.assertEqual(git("status", "--porcelain"), "?? bin/\n")
            self.assertIn("acs2-bench", git("show", "HEAD:reports/slurm_header.out"))


if __name__ == "__main__":
    unittest.main()
