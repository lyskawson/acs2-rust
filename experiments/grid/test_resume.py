import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from grid_protocol import atomic_json
from grid_resume import resume
from grid_scope import SACCT_FIELDS
from submit import submission_command
from test_authorization import changed_manifests, tuning_fixture


class ResumeTests(unittest.TestCase):
    def test_resumption_outside_the_scope_refuses_before_executor_or_file_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = self.fixture(base)
            sibling = base / 'sibling'
            root.rename(sibling)
            before = {str(path.relative_to(base)): path.read_bytes() for path in base.rglob('*') if path.is_file()}
            execute, commands = self.execute()
            with self.assertRaisesRegex(ValueError, 'batch root.*accounting scope'):
                resume(sibling, execute, submission_command)
            self.assertEqual(commands, [])
            self.assertEqual(before, {str(path.relative_to(base)): path.read_bytes() for path in base.rglob('*') if path.is_file()})

    def fixture(self,base):
        root=base/'tuning'
        root.mkdir()
        plan=tuning_fixture(base)
        (root/'approval.json').write_bytes((base/'approval.json').read_bytes())
        atomic_json(root/'plan.json',plan)
        (root/'plan.sha256').write_text(hashlib.sha256((root/'plan.json').read_bytes()).hexdigest())
        from submit import run_batches
        ledger=[]
        for index,batch in enumerate(run_batches(plan)[:-1]):
            limit=60*max(config['limits_minutes'][agent] for config,agent,_ in batch)
            runs=[dict(configuration_id=config['id'],agent=agent,seed=seed,limit_seconds=limit) for config,agent,seed in batch]
            ledger.append(dict(array_id=10+index,runs=runs))
        atomic_json(root/'ledger.json',ledger)
        runs=ledger[0]['runs']
        atomic_json(root/'assignment-00.json',dict(measurement_repository='/clone',binary='/frozen',operations_commit='original',plan_sha256=(root/'plan.sha256').read_text(),runs=runs))
        return root

    def execute(self,fail_test=False):
        commands=[]
        def run(command):
            commands.append(command)
            if command[0]=='git': return 'a592a5bad8bbc13d62553cc09a72748182809879\n' if command[-1]=='HEAD' else ''
            if command[0]=='sacct':
                header=SACCT_FIELDS.replace(',','|')+'|\n'
                return header+''.join(f'{array}_{i}|{array}_{i}|1|00:00:01|1|1|0|COMPLETED|0:0|node|lem-cpu-normal|a|b|\n' for array in range(10,15) for i in range(20))
            if command[0]=='sshare': return '0\n'
            if command[0]=='squeue': return ''
            if '--test-only' in command:
                if fail_test: raise subprocess.CalledProcessError(1,command,output='QOSMaxSubmitJobsPerUserLimit')
                return 'test accepted'
            if command[0]=='sbatch': return '20\n'
            raise AssertionError(command)
        return run,commands

    def test_resumption_submits_only_the_registered_unsubmitted_suffix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary))
            execute,commands=self.execute()
            result=resume(root,execute,submission_command)
            self.assertEqual(result,dict(complete=True,submitted_arrays=6,runs=120))
            self.assertEqual(sum(command[0]=='sbatch' and '--test-only' not in command for command in commands),1)
            ledger=json.loads((root/'ledger.json').read_text())
            self.assertEqual([entry['array_id'] for entry in ledger],[10,11,12,13,14,20])
            self.assertEqual(ledger[-1]['runs'][0]['configuration_id'],'mazema_c9_pairs_p1of3')

    def test_test_only_rejection_records_zero_cost_and_leaves_no_submission_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            root=self.fixture(base)
            execute,commands=self.execute(True)
            result=resume(root,execute,submission_command)
            self.assertFalse(result['complete'])
            self.assertEqual(len(json.loads((root/'ledger.json').read_text())),5)
            self.assertFalse((base/'STOP.json').exists())
            event=json.loads((base/'submission-events.jsonl').read_text())
            self.assertEqual(event['allocated_cpu_seconds'],0)
            self.assertEqual(event['error_output'],'QOSMaxSubmitJobsPerUserLimit')
            self.assertFalse(any(command[0]=='sbatch' and '--test-only' not in command for command in commands))

    def test_existing_measurement_artifacts_prevent_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary))
            (root/'attempts/mazema_c9_pairs_p1of3_her_values_s42_a01/measurement/mazema_c9_pairs_p1of3_her_values_s42').mkdir(parents=True)
            execute,_=self.execute()
            with self.assertRaises(ValueError): resume(root,execute,submission_command)

    def test_resumption_rederives_runs_even_when_the_child_digest_and_ledger_prefix_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary))
            original=json.loads((root/'plan.json').read_text())
            for path,changed in changed_manifests(original):
                with self.subTest(field=path):
                    atomic_json(root/'plan.json',changed)
                    (root/'plan.sha256').write_text(hashlib.sha256((root/'plan.json').read_bytes()).hexdigest())
                    execute,commands=self.execute()
                    with self.assertRaises(ValueError): resume(root,execute,submission_command)
                    self.assertFalse(any(command[0]=='sbatch' for command in commands))

    def test_resumption_rejects_changed_allocation_limits_in_an_existing_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary))
            ledger=json.loads((root/'ledger.json').read_text())
            ledger[0]['runs'][0]['limit_seconds']+=60
            atomic_json(root/'ledger.json',ledger)
            execute,commands=self.execute()
            with self.assertRaises(ValueError): resume(root,execute,submission_command)
            self.assertFalse(any(command[0]=='sbatch' for command in commands))


if __name__ == '__main__': unittest.main()
