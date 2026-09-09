import csv
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from parse_mpx_logs import parse_log
from summarize_mpx import collect, render


class ArchiveTests(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.out"
            path.write_text(content)
            return parse_log(path)

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
        self.assertEqual(len(solved37), 18)

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
