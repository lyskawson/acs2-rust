import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collect import complete_rows, interval_cost, outside_repository, run, validate_row


class CollectionTests(unittest.TestCase):
    def test_incomplete_tail_is_reported_without_changing_raw_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            raw = b'{"step": 1}\n{"step": '
            path.write_bytes(raw)
            self.assertEqual(complete_rows(path), ([{"step": 1}], True))
            self.assertEqual(path.read_bytes(), raw)
            path.write_bytes(b'{"step": 1}\ninvalid\n')
            with self.assertRaises(json.JSONDecodeError):
                complete_rows(path)

    def test_valid_but_unterminated_tail_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"step": 1}')
            self.assertEqual(complete_rows(path), ([], True))

    def test_repository_guard_catches_symlinks_and_worktrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "repo"
            repo.mkdir()
            (repo / ".git").write_text("gitdir: /elsewhere")
            alias = root / "alias"
            alias.symlink_to(repo, target_is_directory=True)
            with self.assertRaises(ValueError):
                outside_repository(alias / "new" / "output")
            self.assertEqual(outside_repository(root / "safe"), (root / "safe").resolve())

    def test_cpu_interpolation_separates_sleep_from_compute(self):
        samples = [(0, 0), (2, 2), (4, 2), (6, 4)]
        cost = interval_cost(samples, 6, 4, 2)
        self.assertEqual(cost["train_cpu_seconds_estimate"], 2)
        self.assertEqual(cost["eval_cpu_seconds_estimate"], 2)

    def test_provenance_pool_seed_and_budget_are_enforced(self):
        config = {"task": "maze4", "cap": 5, "row_pool": "full", "encoding": "coordinates"}
        row = {"schema": 2, "commit": "a" * 40, "source_state": "clean", "task": "maze4",
               "cap": 5, "agent": "acs2er", "seed": 42, "nominal_step": 1000,
               "starts": None, "goal_pool": "full", "goal_encoding": "coordinates",
               "evaluated_policy": "greedy_change_anticipating_population", "host": "node",
               "cpu_model": "cpu", "actual_steps": 1004, "online_updates": 0,
               "replay_updates": 3009, "preset": {"replay_updates_per_step": 3}}
        validate_row(row, config, "acs2er", 42, "a" * 40, 1000)
        for key, value in (("commit", "b" * 40), ("source_state", "dirty"), ("seed", 43),
                           ("goal_pool", "restricted"), ("cpu_model", "unknown"),
                           ("nominal_step", 999), ("actual_steps", 1005), ("replay_updates", 3012)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_row(dict(row, **{key: value}), config, "acs2er", 42, "a" * 40, 1000)

    def test_exited_writer_is_drained_and_missing_row_process_is_stopped(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {"id": "test", "task": "maze4", "cap": 5, "pool": "full",
                      "row_pool": "full", "encoding": "coordinates", "targets": [1]}
            manifest = {"configurations": [config], "agents": ["acs2", "acs2er"], "pilot_seed": 42,
                        "first_row_timeout_seconds": 3, "process_timeout_seconds": 5}
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            binary = root / "writer"
            binary.write_text("#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
                              "Path(sys.argv[-1]).write_text('{}\\n')\n")
            binary.chmod(0o755)
            args = argparse.Namespace(manifest=path, binary=binary, index=0, commit="a" * 40,
                                      output=root / "done", sample_interval=0.001)
            row = {"wall_seconds_train": 0.001, "wall_seconds_eval": 0.001, "nominal_step": 1}
            with patch.dict("os.environ", {"SLURM_JOB_PARTITION": "lem-cpu-normal"}), \
                    patch("collect.process_sample", return_value=(0.0, 0)), \
                    patch("collect.validate_row"), patch("collect.json.loads", side_effect=[manifest, row]), \
                    contextlib.redirect_stdout(io.StringIO()):
                run(args)
            completion = json.loads(next(args.output.glob("*/completion.json")).read_text())
            self.assertEqual(completion["rows_seen"], 1)
            self.assertEqual(completion["exit_code"], 0)
            manifest["first_row_timeout_seconds"] = 0.2
            path.write_text(json.dumps(manifest))
            binary.write_text("#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n")
            args.output = root / "timeout"
            with patch.dict("os.environ", {"SLURM_JOB_PARTITION": "lem-cpu-normal"}), \
                    patch("collect.process_sample", return_value=(0.0, 0)), self.assertRaises(SystemExit):
                run(args)
            completion = json.loads(next(args.output.glob("*/completion.json")).read_text())
            self.assertIn("first row deadline", completion["failure"])
            self.assertLess(completion["elapsed_seconds"], 2)
            self.assertNotEqual(completion["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
