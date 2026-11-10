import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import HERE,new_cells,configurations,recorded
from register_study import build
from execution import dispatch,registered_runs,require_authorization
from grid_protocol import atomic_json
from prepare_plan import forecast


class ExecutionTests(unittest.TestCase):
    def plan(self):
        from execution import checksum
        return dict(registration=build(),registration_sha256=checksum(HERE/'registration.json'),
            phase_cpu_hour_limit=15,concurrency=20,partition='lem-cpu-normal',
            configurations=[recorded(c,42) for c in configurations()],measurement_repository='/tmp/repo',binary='/tmp/binary',measurement_commit='pinned')

    def test_registration_contains_only_138_new_seed42_cells_and_fits_reservations(self):
        p=self.plan();r=registered_runs(p)
        self.assertEqual(len(r),138);self.assertTrue(all(x['seed']==42 and x['per_pair'] for x in r))
        self.assertEqual(len({x['attempt_id'] for x in r}),138)
        self.assertLessEqual(sum(x['limit_seconds'] for x in r),15*3600)
        for key,value in [('phase_cpu_hour_limit',16),('concurrency',40)]:
            bad=copy.deepcopy(p);bad[key]=value
            with self.assertRaises(ValueError):require_authorization(bad)
        bad=copy.deepcopy(p);bad['registration']['pilot_runs'][0]['seed']=43
        with self.assertRaises(ValueError):require_authorization(bad)

    def test_dispatch_checks_grant_and_scheduler_before_submission_and_records_every_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'logs').mkdir();p=self.plan();p['operations_manifest']=str(root/'operations/manifest.json');p['operations_manifest_sha256']='fake';p['stop_path']=str(root/'STOP.json');p['result_root']=str(root)
            atomic_json(root/'plan.json',p)
            from execution import checksum
            (root/'plan.sha256').write_text(checksum(root/'plan.json')+'\n');atomic_json(root/'ledger.json',[])
            calls=[]
            def run(command):
                calls.append(command)
                if command[0]=='sshare':return '23000000\n'
                if command[0]=='squeue':return ''
                if command[0]=='sbatch':return 'test accepted' if '--test-only' in command else '12345\n'
                raise AssertionError(command)
            with patch('execution.verify_operations'):
                result=dispatch(root,run)
            self.assertTrue(result['submitted']);self.assertEqual(result['runs'],6)
            self.assertEqual([c[0] for c in calls],['sshare','squeue','sbatch','sbatch'])
            self.assertIn('--test-only',calls[-2]);self.assertNotIn('--test-only',calls[-1])
            ledger=json.loads((root/'ledger.json').read_text());self.assertEqual(ledger[0]['array_id'],12345)
            self.assertEqual(sum(r['limit_seconds'] for r in ledger[0]['runs']),7200)

    def test_forecast_retains_per_maze_arm_costs_and_rejects_missing_or_duplicate_cells(self):
        runs=[dict(configuration_id=c['id'],task=c['task'],agent=a,cost=dict(allocation_cpu_seconds=10+i,body_wall_seconds=9+i,peak_rss_kib=1024*(10+i))) for i,(c,a) in enumerate(new_cells())]
        result=forecast(runs,100)
        self.assertEqual(result['future_runs'],2622)
        self.assertEqual(result['future_expected_cpu_hours'],19*sum(10+i for i in range(138))/3600)
        self.assertEqual(len(result['cells']),138)
        self.assertGreater(result['requested_cpu_hour_limit'],result['expected_total_cpu_hours'])
        for bad in [runs[:-1],runs[:-1]+[runs[0]]]:
            with self.assertRaises(ValueError):forecast(bad,100)

    def test_stop_budget_shared_grant_and_two_stream_guards_prevent_submission(self):
        for cause in ('stop','budget','grant','streams'):
            with self.subTest(cause=cause),tempfile.TemporaryDirectory() as d:
                root=Path(d);p=self.plan();p['stop_path']=str(root/'STOP.json');p['result_root']=str(root);p['operations_manifest']=str(root/'operations/manifest.json')
                atomic_json(root/'plan.json',p)
                from execution import checksum
                (root/'plan.sha256').write_text(checksum(root/'plan.json')+'\n');atomic_json(root/'ledger.json',[])
                if cause=='stop':(root/'STOP.json').write_text('{}')
                state=dict(spent_seconds=14.9*3600 if cause=='budget' else 0,reserved_seconds=0)
                records=[]
                if cause=='streams':
                    runs=registered_runs(self.plan())[:2]
                    atomic_json(root/'ledger.json',[dict(array_id=i+1,runs=[r]) for i,r in enumerate(runs)])
                    records=[dict(JobID=f'{i+1}_0',State='RUNNING') for i in range(2)]
                calls=[]
                def run(command):
                    calls.append(command)
                    if command[0]=='sshare':return str(15000*3600-1)
                    if command[0]=='squeue':return ''
                    raise AssertionError('submission must be prevented')
                with patch('execution.verify_operations'),patch('execution.balance',return_value=(state,records)):
                    if cause=='streams':self.assertFalse(dispatch(root,run)['submitted'])
                    else:
                        with self.assertRaises(ValueError):dispatch(root,run)
                self.assertFalse(any(c[0]=='sbatch' for c in calls))

    def test_preparation_driver_freezes_clean_identity_and_operations_and_rejects_modified_snapshot(self):
        import shutil
        from execution import prepare,checksum
        from protocol import ROOT
        with tempfile.TemporaryDirectory() as d:
            home=Path(d);repo=home/'repo';root=home/'tu_runs/per-goal-reward-ablation/pilot'
            files=['fixtures/goal_measurement_regression.json','experiments/per-goal-reward-ablation/registration.json']
            for name in files:
                target=repo/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
            binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure';binary.parent.mkdir(parents=True);binary.write_bytes(b'pinned-static-binary')
            def execute(command):
                if command[-2:]==['rev-parse','HEAD']:return 'pinned\n'
                if command[-2:]==['status','--porcelain']:return ''
                if 'ls-files' in command:return '\n'.join(files)
                raise AssertionError(command)
            with patch('execution.Path.home',return_value=home),patch('execution.execute',side_effect=execute):
                result=prepare(root,repo)
                with self.assertRaises(ValueError):prepare(root,repo)
            self.assertEqual(result['commit'],'pinned');self.assertEqual(result['binary_sha256'],checksum(binary))
            plan=json.loads((root/'plan.json').read_text());require_authorization(plan)
            copied=root/'operations'/files[0];copied.write_text('changed')
            with self.assertRaises(ValueError):require_authorization(plan)

    def test_final_plan_driver_keeps_registration_parameters_and_requires_frozen_references(self):
        from prepare_plan import prepare
        runs=[dict(configuration_id=c['id'],task=c['task'],agent=a,cost=dict(allocation_cpu_seconds=10+i,body_wall_seconds=9+i,peak_rss_kib=1024*(10+i))) for i,(c,a) in enumerate(new_cells())]
        summary=dict(runs=runs,allocation_cpu_seconds=100,measurement_commit='pinned',binary_sha256='b'*64)
        with self.assertRaises(ValueError):prepare(summary,None)
        plan=prepare(summary,'a'*64)
        self.assertEqual(plan['stage'],'grid_pending_approval');self.assertEqual(plan['new_runs'],2760)
        self.assertEqual(plan['archive_reference_index_sha256'],'a'*64)
        self.assertEqual(plan['group_counts'],dict(no_aliasing=9,aliasing=19))
        self.assertFalse(plan['preset']['do_ga']);self.assertEqual(plan['preset']['epsilon'],0.8)
        self.assertEqual(plan['remaining_seeds'],list(range(43,62)))
