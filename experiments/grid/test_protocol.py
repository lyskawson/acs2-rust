import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collect import validate_row
from grid_launch import allocation_fields
from grid_attempts import verify_artifacts
from grid_protocol import accounting, load_plan, require_budget
from row_schema import evaluation_blocks, validate_blocks
from submit import prepare, require_authorization, shared_reservation, submission_command

ROOT = Path(__file__).resolve().parents[2]


def split_row():
    manifest = json.loads((ROOT / 'experiments/maze-baselines/pilot.json').read_text())
    config = manifest['configurations'][4]
    oracle = json.loads((ROOT / 'fixtures/goal_measurement_regression.json').read_text())
    old = oracle['configurations'][0]['rows'][0]
    row = dict(old, schema=4, commit='a' * 40, source_state='clean', host='node', cpu_model=manifest['cpu_model'],
               task=config['task'], cap=config['cap'], agent='acs2', seed=42, nominal_step=1000,
               actual_steps=1000, online_updates=1000, replay_updates=0, goal_pool=config['row_pool'])
    row['evaluation_blocks'] = {}
    for name, reference in config['registration']['blocks'].items():
        row['evaluation_blocks'][name] = dict(evaluation_blocks(dict(old, schema=2))['full_pool'],
            evaluation_pairs=reference['evaluation_pairs'], random_floor=reference['random_floor'],
            reachable_within_cap=reference['reachable_within_cap'], reachable_after_cap=0, unreachable_or_ambiguous=0,
            success=0.05)
    for key in list(evaluation_blocks(dict(old, schema=2))['full_pool']):
        row.pop(key)
    row['goal_protocol'] = {'id': config['protocol'], 'configuration': config['registration']}
    return config, row


class SharedGridTests(unittest.TestCase):
    def test_old_schemas_remain_readable_without_relabeling_their_distribution(self):
        old = json.loads((ROOT / 'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]
        for schema in (2, 3, 4):
            self.assertEqual(evaluation_blocks(dict(old, schema=schema)), {'full_pool': {key: old[key] for key in evaluation_blocks(dict(old, schema=2))['full_pool']}})

    def test_blocks_and_references_are_validated_separately(self):
        config, row = split_row()
        validate_row(row, config, 'acs2', 42, 'a' * 40, 1000)
        for key in ['random_floor', 'evaluation_pairs']:
            wrong = copy.deepcopy(row)
            wrong['evaluation_blocks']['test'][key] = wrong['evaluation_blocks']['training'][key]
            with self.assertRaises(ValueError):
                validate_blocks(wrong, config)
        wrong = copy.deepcopy(row)
        wrong['evaluation_blocks']['test'] = copy.deepcopy(wrong['evaluation_blocks']['training'])
        with self.assertRaises(ValueError):
            validate_blocks(wrong, config)
        wrong = dict(row, success=0.5)
        with self.assertRaises(ValueError):
            evaluation_blocks(wrong)

    def test_exact_allocation_identity_prevents_another_task_changing_the_watchdog(self):
        data = 'JobId=123 ArrayJobId=120 ArrayTaskId=2 TimeLimit=01:00:00 RunTime=00:00:01'
        self.assertEqual(allocation_fields(data, '123', '120', '2')['TimeLimit'], '01:00:00')
        with self.assertRaises(ValueError):
            allocation_fields(data + '\n' + data, '123', '120', '2')
        with self.assertRaises(ValueError):
            allocation_fields(data, '123', '120', '3')

    def test_budget_counts_pilot_used_reserved_and_new_allocations_once(self):
        ledger = [{'array_id': 12, 'runs': [{'limit_seconds': 100}, {'limit_seconds': 200}]}]
        records = [{'JobID': '12_0', 'CPUTimeRAW': '50', 'State': 'COMPLETED'},
                   {'JobID': '12_1', 'CPUTimeRAW': '20', 'State': 'RUNNING'}]
        state = accounting(ledger, records)
        self.assertEqual(state, {'spent_seconds': 70, 'reserved_seconds': 180})
        self.assertEqual(require_budget(10, state, 40), 300)
        with self.assertRaises(ValueError):
            require_budget(10, state, 40 * 3600)

    def test_submission_has_node_guards_and_no_lease_or_host_pinning(self):
        command = submission_command(Path('/tmp/run'), Path('/tmp/a.json'), 60, 40, 20, 512, [123])
        self.assertIn('--partition=lem-cpu-normal', command)
        self.assertIn('--array=0-39%20', command)
        self.assertIn('--cpus-per-task=1', command)
        self.assertIn('--dependency=afterok:123', command)
        self.assertFalse(any('nodelist' in part or 'lease' in part for part in command))
        self.assertEqual(shared_reservation('1|job|R|1-00:00:00|1\n2|job|PD|00:10:00|2'), 87600)

    def test_plan_bytes_and_explicit_grid_approval_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = {'stage': 'grid_pending_approval', 'seeds': list(range(42, 62))}
            path = root / 'plan.json'
            path.write_text(json.dumps(plan))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(load_plan(path, digest), plan)
            with self.assertRaises(ValueError):
                load_plan(path, 'wrong')
            with self.assertRaises(FileNotFoundError):
                require_authorization(root, plan, digest)
            (root / 'approval.json').write_text(json.dumps({'approved': True, 'plan_sha256': digest}))
            require_authorization(root, plan, digest)
            with self.assertRaises(ValueError):
                require_authorization(root, plan, 'wrong')

    def test_missing_approval_leaves_no_result_root(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            binary = repo / 'target/x86_64-unknown-linux-musl/release/acs2-measure'
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b'binary')
            (repo / 'fixtures').mkdir()
            (repo / 'fixtures/goal_measurement_regression.json').write_text(json.dumps({'configurations': [{'rows': [{'preset': {}}]}]}))
            manifest = repo / 'manifest.json'
            manifest.write_text(json.dumps({'stage': 'grid_pending_approval', 'seeds': list(range(42, 62))}))
            root = repo / 'results'
            with patch('submit.execute', side_effect=['a' * 40, '']):
                with self.assertRaises(FileNotFoundError):
                    prepare(root, manifest, repo)
            self.assertFalse(root.exists())

    def test_attempt_checksums_detect_changed_raw_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            names = ['rows.jsonl', 'resources.jsonl', 'cpu_samples.csv', 'launch.json']
            for name in names:
                (path / name).write_bytes(b'original\n')
            completion = {'sha256': {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in names}}
            verify_artifacts(path, completion)
            (path / 'rows.jsonl').write_bytes(b'changed\n')
            with self.assertRaises(ValueError):
                verify_artifacts(path, completion)


if __name__ == '__main__':
    unittest.main()
