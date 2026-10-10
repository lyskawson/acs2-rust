import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from grid_attempts import select_attempts
from grid_control import chain_plan, monitor, submission_command
from grid_launch import allocation_fields, main as launch
from grid_protocol import ORDER, accounting, atomic_json, stop_phase


class RecoveryTests(unittest.TestCase):
    def test_attempt_selection_preserves_empty_partial_and_equal_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(id='fixture', targets=[1000])
            ledger = []
            for index, status in enumerate(['no_rows', 'incomplete', 'complete', 'complete']):
                path = root / str(index)
                path.mkdir()
                run = dict(agent='acs2', seed=42, relative_directory=str(index), attempt_id=f'a{index}', cause=status)
                ledger.append(dict(configuration_id='fixture', array_id=index + 1, runs=[run]))
                if status == 'no_rows':
                    continue
                row = dict(nominal_step=1000, success=0.5, host=f'host{index}', wall_seconds_total=index)
                (path / 'rows.jsonl').write_text(json.dumps(row) + '\n' + ('{"tail":' if status == 'incomplete' else ''))
                atomic_json(path / 'launch.json', dict(started_unix=10 - index))
                atomic_json(path / 'completion.json', dict(exit_code=0, failure=None, incomplete_final_line=False, rows_seen=1))
                (path / 'resources.jsonl').write_text('{"nominal_step":1000}\n')
            selected, inventory = select_attempts(root, ledger, config)
            self.assertEqual(selected['acs2', 42]['attempt_id'], 'a3')
            self.assertEqual([item['status'] for item in inventory], ['no_rows', 'incomplete', 'complete', 'complete'])
            self.assertEqual(inventory[2]['duplicate_of'], 'a3')
            self.assertEqual((root / '1/rows.jsonl').read_bytes()[-8:], b'{"tail":')
            row['success'] = 0.6
            (root / '2/rows.jsonl').write_text(json.dumps(row) + '\n')
            with self.assertRaisesRegex(ValueError, 'disagree'):
                select_attempts(root, ledger, config)

    def test_accounted_failures_do_not_block_but_new_failures_do(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [dict(JobID='12_0', CPUTimeRAW='4', State='FAILED'), dict(JobID='12_1', CPUTimeRAW='0', State='CANCELLED by 7')]
            runs = [dict(agent='acs2', seed=42 + i, limit_seconds=100, relative_directory=str(i), attempt_id=str(i),
                         accounted_failure=dict(state=row['State'].split()[0], cpu_seconds=float(row['CPUTimeRAW']), cause='audited')) for i, row in enumerate(records)]
            ledger = [dict(array_id=12, configuration_id='fixture', runs=runs)]
            plan = dict(configurations=[dict(id='fixture', targets=[1])])
            balance = accounting(ledger, records)
            self.assertEqual(balance, dict(spent_seconds=4, reserved_seconds=0))
            with patch('grid_control.snapshot', return_value=(balance, records, '')):
                result = monitor(root, plan, ledger)
                self.assertEqual(len(result['accounted_historical_failures']), 2)
                self.assertEqual(result['failures'], [])
                del runs[0]['accounted_failure']
                result = monitor(root, plan, ledger)
                self.assertEqual(len(result['failures']), 1)
                self.assertFalse((root / 'STOP.json').exists())

    def test_chain_has_four_lanes_no_pinning_and_all_previous_dependencies(self):
        plan = dict(agents=['acs2', 'acs2er'], seeds=list(range(42, 62)),
                    configurations=[dict(id=name, task='maze', targets=[1], limits_minutes=dict(acs2=10, acs2er=60)) for name in ORDER])
        selected_bit = {(agent, seed): {} for agent in plan['agents'] for seed in plan['seeds']}
        selected_he = {('acs2', seed): {} for seed in plan['seeds'] if seed % 2}
        selected_he.update({('acs2er', seed): {} for seed in [43, 45, 47, 49]})
        with patch('grid_control.select_attempts', side_effect=lambda root, ledger, config: (selected_bit if config['id'] == ORDER[0] else selected_he if config['id'] == ORDER[1] else {}, [])):
            batches = chain_plan(Path('/tmp/grid'), plan, [])
        self.assertEqual([batch['configuration_id'] for batch in batches], ORDER[1:])
        self.assertEqual(sum(len(group['runs']) for batch in batches for group in batch['groups']), 346)
        self.assertEqual([len(group['runs']) for group in batches[0]['groups']], [5, 5, 8, 8])
        for batch in batches:
            self.assertEqual(len(batch['groups']), 4)
            for group in batch['groups']:
                command = submission_command(Path('/ops'), Path('/root'), Path('/assignment'), Path('/logs'), plan['configurations'][1], group, [11, 12, 13, 14])
                self.assertIn('--dependency=afterok:11:12:13:14', command)
                self.assertIn('--hold', command)
                self.assertIn(f"--array=0-{len(group['runs']) - 1}%1", command)
                self.assertFalse(any('nodelist' in arg for arg in command))

    def test_launch_does_not_read_supervision_and_stop_blocks_without_new_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / 'supervision.json', {'unix': 0})
            assignment = root / 'assignment.json'
            atomic_json(assignment, dict(runs=[dict(configuration_id='fixture', agent='acs2', seed=42,
                        attempt_id='a2', relative_directory='attempts/a2/measurement/fixture_acs2_s42')],
                        measurement_repository='/repo', operations_commit='ops', plan_sha256='plan', binary='/binary'))
            (root / 'attempts/a2').mkdir(parents=True)
            atomic_json(root / 'references.json', {'fixture': {'preset': {}}})
            atomic_json(root / 'ledger.json', [])
            plan = dict(measurement_commit='commit', configurations=[dict(id='fixture')], cpu_model='cpu', binary_sha256='sha')
            with patch('grid_launch.sys.argv', ['launch', str(root), str(assignment), '0']), patch('grid_launch.load_plan', return_value=plan), patch.dict('os.environ', {'SLURM_JOB_ID': '1', 'SLURM_ARRAY_JOB_ID': '1', 'SLURM_ARRAY_TASK_ID': '0'}), patch('grid_launch.subprocess.check_output', side_effect=['commit', '', 'JobId=1 ArrayJobId=1 ArrayTaskId=0 TimeLimit=00:10:00 RunTime=00:00:01']) as query, patch('grid_launch.subprocess.run') as run, patch('grid_launch.select_attempts'):
                run.return_value.returncode = 0
                launch()
                manifest = json.loads(next((root / 'attempts/a2/resolved-manifests').glob('*.json')).read_text())
                self.assertEqual(query.call_args.args[0][-1], '1_0')
                self.assertEqual(json.loads((root / 'attempts/a2/allocation-start.json').read_text())['selector'], '1_0')
                self.assertIsNone(manifest['expected_host'])
                self.assertEqual(manifest['process_timeout_seconds'], 569)
                stop_phase(root / 'STOP.json', {'cause': 'original'})
                before = (root / 'STOP.json').read_bytes()
                with self.assertRaises(SystemExit):
                    launch()
                self.assertEqual((root / 'STOP.json').read_bytes(), before)
                self.assertEqual(len(list(root.glob('STOP-event-*'))), 1)
                self.assertEqual(run.call_count, 1)

    def test_array_master_response_cannot_replace_this_tasks_runtime(self):
        current = 'JobId=6015551 ArrayJobId=6015551 ArrayTaskId=9 TimeLimit=00:50:00 RunTime=00:00:00'
        earlier = 'JobId=6016450 ArrayJobId=6015551 ArrayTaskId=8 TimeLimit=00:50:00 RunTime=00:25:58'
        fields = allocation_fields(current, '6015551', '6015551', '9')
        self.assertEqual(fields['RunTime'], '00:00:00')
        for response in [current + '\n' + earlier, earlier, current.replace('ArrayTaskId=9', 'ArrayTaskId=8'), '']:
            with self.assertRaises(ValueError):
                allocation_fields(response, '6015551', '6015551', '9')

    def test_accounted_partial_attempt_requires_unchanged_raw_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / 'attempt'
            path.mkdir()
            raw = b'{"nominal_step":1,"commit":"commit","source_state":"clean","cpu_model":"cpu"}\n'
            (path / 'rows.jsonl').write_bytes(raw)
            run = dict(agent='acs2', seed=42, relative_directory='attempt', attempt_id='attempt', limit_seconds=3000,
                       accounted_failure=dict(state='FAILED', cpu_seconds=1412, rows_sha256=hashlib.sha256(raw).hexdigest(), cause='audited watchdog defect'))
            ledger = [dict(array_id=12, configuration_id='fixture', runs=[run])]
            records = [dict(JobID='12_0', CPUTimeRAW='1412', State='FAILED')]
            plan = dict(configurations=[dict(id='fixture', targets=[1, 2])], measurement_commit='commit', cpu_model='cpu')
            with patch('grid_control.snapshot', return_value=(accounting(ledger, records), records, '')):
                self.assertEqual(monitor(root, plan, ledger)['failures'], [])
                (path / 'rows.jsonl').write_bytes(raw + b'{"tail":')
                self.assertEqual(len(monitor(root, plan, ledger)['failures']), 1)
                (path / 'rows.jsonl').unlink()
                self.assertEqual(len(monitor(root, plan, ledger)['failures']), 1)

    def test_stop_preserves_first_failure_and_all_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'STOP.json'
            stop_phase(path, {'cause': 'first'})
            stop_phase(path, {'cause': 'second'})
            self.assertEqual(json.loads(path.read_text())['cause'], 'first')
            self.assertEqual(len(list(path.parent.glob('STOP-event-*'))), 2)


if __name__ == '__main__':
    unittest.main()
