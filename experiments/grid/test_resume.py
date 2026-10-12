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


class ResumeTests(unittest.TestCase):
    def fixture(self,base):
        root=base/'grid'
        root.mkdir()
        approval=dict(approved=True,plan_sha256='approved')
        atomic_json(base/'approval.json',approval)
        atomic_json(root/'approval.json',approval)
        scope=base/'scope.json'
        atomic_json(scope,dict(batch_roots=[str(root)],pilot_allocation_cpu_seconds=5445,
                              authorization_sha256='approved',accounting_since='2026-10-12'))
        plan=dict(stage='approved_grid',configurations=[dict(id=name,limits_minutes={'her_values':2}) for name in ['first','next']],
            agents=['her_values'],seeds=list(range(42,62)),batch_size=20,memory_mib=512,
            measurement_commit='frozen',authorization_plan=str(base/'registered.json'),authorization_sha256='approved',
            budget_scope=str(scope),budget_scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest(),
            pilot_allocation_cpu_seconds=5445,stop_path=str(base/'STOP.json'))
        atomic_json(root/'plan.json',plan)
        (root/'plan.sha256').write_text(hashlib.sha256((root/'plan.json').read_bytes()).hexdigest())
        runs=[dict(configuration_id='first',agent='her_values',seed=seed,limit_seconds=120) for seed in range(42,62)]
        atomic_json(root/'ledger.json',[dict(array_id=10,runs=runs)])
        atomic_json(root/'assignment-00.json',dict(measurement_repository='/clone',binary='/frozen',operations_commit='original',plan_sha256=(root/'plan.sha256').read_text(),runs=runs))
        return root

    def execute(self,fail_test=False):
        commands=[]
        def run(command):
            commands.append(command)
            if command[0]=='git': return 'frozen\n' if command[-1]=='HEAD' else ''
            if command[0]=='sacct':
                header=SACCT_FIELDS.replace(',','|')+'|\n'
                return header+''.join(f'10_{i}|10_{i}|1|00:00:01|1|1|0|COMPLETED|0:0|node|lem-cpu-normal|a|b|\n' for i in range(20))
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
            self.assertEqual(result,dict(complete=True,submitted_arrays=2,runs=40))
            self.assertEqual(sum(command[0]=='sbatch' and '--test-only' not in command for command in commands),1)
            ledger=json.loads((root/'ledger.json').read_text())
            self.assertEqual([entry['array_id'] for entry in ledger],[10,20])
            self.assertEqual(ledger[1]['runs'][0]['configuration_id'],'next')

    def test_test_only_rejection_records_zero_cost_and_leaves_no_submission_intent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            root=self.fixture(base)
            execute,commands=self.execute(True)
            result=resume(root,execute,submission_command)
            self.assertFalse(result['complete'])
            self.assertEqual(len(json.loads((root/'ledger.json').read_text())),1)
            self.assertFalse((base/'STOP.json').exists())
            event=json.loads((base/'submission-events.jsonl').read_text())
            self.assertEqual(event['allocated_cpu_seconds'],0)
            self.assertEqual(event['error_output'],'QOSMaxSubmitJobsPerUserLimit')
            self.assertFalse(any(command[0]=='sbatch' and '--test-only' not in command for command in commands))

    def test_existing_measurement_artifacts_prevent_duplicate_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.fixture(Path(temporary))
            (root/'attempts/next_her_values_s42_a01/measurement/next_her_values_s42').mkdir(parents=True)
            execute,_=self.execute()
            with self.assertRaises(ValueError): resume(root,execute,submission_command)


if __name__ == '__main__': unittest.main()
