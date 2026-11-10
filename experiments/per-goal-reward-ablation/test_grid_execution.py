import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import grid_execution as grid
import launch_grid
from protocol import recorded
from grid_protocol import atomic_json


class GridExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registered=grid.registered_plan()
        cls.runs=grid.registered_runs(cls.registered)

    def completion(self,root,run,registered=None):
        registered=registered or self.registered
        config=next(c for c in registered['configurations'] if c['id']==run['configuration_id'])
        path=root/run['relative_directory'];path.mkdir(parents=True,exist_ok=True)
        atomic_json(path/'completion.json',dict(exit_code=0,failure=None,incomplete_final_line=False,
            rows_seen=len(config['targets']),rows_expected=len(config['targets'])))

    def wrapper(self,root):
        operations=root/'operations';operations.mkdir(exist_ok=True)
        atomic_json(operations/'manifest.json',dict(commit='operations',files={}))
        return dict(result_root=str(root),registered_plan=str(root/'registered-plan.json'),authorization=grid.authorization(),
            measurement_commit=grid.MEASUREMENT_COMMIT,binary_sha256=grid.BINARY_SHA256,
            phase_cpu_hour_limit=3583,pilot_allocation_cpu_seconds=12965,memory_mib=2048,partition='lem-cpu-normal',concurrency=20,
            configurations=self.registered['configurations'],preset=self.registered['preset'],pilots=[],
            measurement_repository='/measurement',binary='/binary',operations_commit='operations',operations_manifest=str(operations/'manifest.json'),
            operations_manifest_sha256=grid.checksum(operations/'manifest.json'),
            cpu_model='AMD EPYC 9554 64-Core Processor')

    def scouts(self,root,registered=None):
        registered=registered or self.registered
        runs=[r for r in grid.registered_runs(registered) if r['phase']=='scout']
        ledger=[dict(array_id=100+i,runs=[r]) for i,r in enumerate(runs)]
        records=[]
        for entry in ledger:
            run=entry['runs'][0];self.completion(root,run,registered)
            records.append(dict(JobID=f"{entry['array_id']}_0",JobIDRaw=str(entry['array_id']),State='COMPLETED',
                CPUTimeRAW=str(int(run['forecast_high_cpu_seconds'])),AllocCPUS='1',ExitCode='0:0'))
        return ledger,records

    def test_full_run_set_preserves_all_seeds_budgets_and_per_pair_schedule(self):
        self.assertEqual(len(self.runs),2760)
        self.assertEqual(len({r['attempt_id'] for r in self.runs}),2760)
        self.assertEqual({phase:sum(r['phase']==phase for r in self.runs) for phase in ['ordinary','scout','conditional']},
            dict(ordinary=2520,scout=12,conditional=228))
        for run in self.runs:
            config=next(c for c in self.registered['configurations'] if c['id']==run['configuration_id'])
            self.assertEqual(config['targets'][-1],4000000 if run['task']=='woods14' else 2000000)
            self.assertEqual(recorded(config,run['seed']).get('per_start_diagnostics',False),run['seed']<=46)
            if run['task'] in grid.OLD_TASKS:self.assertEqual(run['agent'],grid.A1)
        reserved=12965+sum(r['limit_seconds'] for r in self.runs)
        self.assertLessEqual(reserved,3583*3600)
        self.assertGreater(reserved,3582*3600)

    def test_approved_plan_bytes_and_wrapper_scope_cannot_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=self.wrapper(root)
            (root/'registered-plan.json').write_bytes((grid.HERE/'plan.json').read_bytes())
            with patch.object(grid,'verify_operations'):
                self.assertEqual(grid.require_authorization(plan),self.registered)
                for field,value in [('phase_cpu_hour_limit',3584),('memory_mib',4096),('concurrency',40),
                        ('measurement_commit','changed'),('binary_sha256','changed')]:
                    changed=copy.deepcopy(plan);changed[field]=value
                    with self.assertRaises(ValueError):grid.require_authorization(changed)
                changed=copy.deepcopy(plan);changed['configurations'][0]['targets'][-1]=30000
                with self.assertRaises(ValueError):grid.require_authorization(changed)
                with (root/'registered-plan.json').open('a') as stream:stream.write(' ')
                with self.assertRaises(ValueError):grid.require_authorization(plan)

    def test_frozen_pilots_authenticate_all_156_attempts_against_archive_manifests(self):
        with tempfile.TemporaryDirectory() as directory:
            parent=Path(directory);registered=copy.deepcopy(self.registered)
            for name,key,count,account in [('stage1','prior_archive_manifest_sha256',18,'accounting.psv'),
                    ('pilot-30000','fresh_archive_manifest_sha256',138,'accounting-all.psv')]:
                folder=parent/name;folder.mkdir();atomic_json(folder/'plan.json',{})
                atomic_json(folder/'ledger.json',[dict(array_id=10,runs=[{}]*count)])
                (folder/account).write_text('accounting')
                atomic_json(folder/'archive-checksums.json',{n:grid.checksum(folder/n) for n in ['plan.json','ledger.json',account]})
                registered[key]=grid.checksum(folder/'archive-checksums.json')
            self.assertEqual(len(grid.frozen_pilots(parent,registered)),2)
            atomic_json(parent/'stage1/ledger.json',[])
            with self.assertRaises(ValueError):grid.frozen_pilots(parent,registered)

    def test_prepare_keeps_approved_plan_bytes_and_pinned_binary_in_a_separate_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            home=Path(directory);root=home/'tu_runs/per-goal-reward-ablation/grid'
            root.parent.mkdir(parents=True);repo=home/'acs2-tu';operations=home/'operations';operations.mkdir()
            atomic_json(operations/'manifest.json',dict(commit='operations',files={}))
            binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure';binary.parent.mkdir(parents=True);binary.write_bytes(b'binary')
            actual_checksum=grid.checksum
            def checksum(path):return grid.BINARY_SHA256 if Path(path).resolve()==binary.resolve() else actual_checksum(path)
            def execute(command):return grid.MEASUREMENT_COMMIT if 'rev-parse' in command else ''
            with patch.object(grid.Path,'home',return_value=home),patch.object(grid,'checksum',side_effect=checksum),patch.object(grid,'execute',side_effect=execute),patch.object(grid,'frozen_pilots',return_value=[]):
                result=grid.prepare(root,repo,operations)
                self.assertEqual(result['runs'],2760)
                self.assertEqual(grid.checksum(root/'registered-plan.json'),grid.APPROVED_SHA256)
                wrapper=json.loads((root/'plan.json').read_text())
                self.assertEqual(grid.require_authorization(wrapper),self.registered)
                self.assertEqual(wrapper['measurement_commit'],grid.MEASUREMENT_COMMIT)
                self.assertEqual(wrapper['binary_sha256'],grid.BINARY_SHA256)
                self.assertFalse(wrapper['authorization']['prefix_comparison'])
                with self.assertRaises(ValueError):grid.prepare(root,repo,operations)

    def test_balance_deduplicates_job_id_raw_and_retains_pilot_spend_and_pending_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prior=root/'prior';prior.mkdir()
            atomic_json(prior/'ledger.json',[dict(array_id=10,runs=[dict(limit_seconds=14000)])])
            ledger=[dict(array_id=20,runs=[dict(limit_seconds=180),dict(limit_seconds=180)])]
            header='JobID|JobIDRaw|CPUTimeRAW|State|\n'
            rows='10_0|100|12965|COMPLETED|\n20_0|200|15|COMPLETED|\n20_1|201|0|PENDING|\n'
            state,records=grid.balance(root,dict(pilots=[dict(root=str(prior))],accounting_since='2026-11-09'),ledger,
                lambda command:header+rows+rows)
            self.assertEqual(len(records),3)
            self.assertEqual(state['spent_seconds'],12980)
            self.assertEqual(state['reserved_seconds'],180)
            self.assertEqual(state['pilot_cpu_seconds'],12965)
            self.assertEqual(state['grid_cpu_seconds'],15)
            with self.assertRaises(ValueError):
                grid.balance(root,dict(pilots=[dict(root=str(prior))],accounting_since='2026-11-09'),ledger,
                    lambda command:header+rows.replace('12965','12964'))

    def test_scout_gate_requires_all_twelve_completions_and_each_own_upper_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger,records=self.scouts(root)
            self.assertTrue(grid.scout_gate(root,self.registered,ledger,records)['passed'])
            self.assertFalse(grid.scout_gate(root,self.registered,ledger[:-1],records)['passed'])
            changed=copy.deepcopy(records);changed[-1]['State']='RUNNING'
            self.assertFalse(grid.scout_gate(root,self.registered,ledger,changed)['passed'])
            changed=copy.deepcopy(records);changed[0]['CPUTimeRAW']=str(int(ledger[0]['runs'][0]['forecast_high_cpu_seconds'])+1)
            gate=grid.scout_gate(root,self.registered,ledger,changed)
            self.assertTrue(gate['complete']);self.assertTrue(gate['failed_cost']);self.assertFalse(gate['passed'])
            path=root/ledger[0]['runs'][0]['relative_directory']/'completion.json'
            value=json.loads(path.read_text());value['rows_seen']-=1;atomic_json(path,value)
            with self.assertRaises(ValueError):grid.scout_gate(root,self.registered,ledger,records)

    def test_scout_gate_boundary_uses_cpu_allocation_not_process_cpu_or_cell_average(self):
        registered=copy.deepcopy(self.registered)
        for cell in registered['forecast']['cells']:cell['forecast_registered_high_cpu_seconds']=100
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ledger,records=self.scouts(root,registered)
            for record in records:record['CPUTimeRAW']='100'
            self.assertTrue(grid.scout_gate(root,registered,ledger,records)['passed'])
            records[0]['CPUTimeRAW']='101';records[1]['CPUTimeRAW']='1'
            self.assertFalse(grid.scout_gate(root,registered,ledger,records)['passed'])

    def dispatch(self,root,ledger=None,state=None,queue='',grant='0'):
        plan=self.wrapper(root);atomic_json(root/'plan.json',plan)
        (root/'plan.sha256').write_text(grid.checksum(root/'plan.json'))
        atomic_json(root/'ledger.json',ledger or [])
        calls=[]
        def executor(command):
            calls.append(command)
            if command[0]=='sshare':return grant
            if command[0]=='squeue':return queue
            if command[0]=='sbatch':return 'accepted' if '--test-only' in command else '12345;cluster'
            raise AssertionError(command)
        state=state or dict(spent_seconds=12965,reserved_seconds=0)
        with patch.object(grid,'require_authorization',return_value=self.registered),patch.object(grid,'balance',return_value=(state,[])):
            result=grid.dispatch(root,executor)
        return result,calls

    def test_dispatch_checks_grant_and_test_only_and_reserves_only_equal_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result,calls=self.dispatch(root)
            self.assertTrue(result['submitted']);self.assertEqual(result['phase'],'ordinary')
            self.assertEqual([c[0] for c in calls],['sshare','squeue','sbatch','sbatch'])
            self.assertIn('--test-only',calls[2]);self.assertNotIn('--test-only',calls[3])
            self.assertIn('--mem=2048M',calls[3]);self.assertIn('--array=0-19%20',calls[3])
            assigned=json.loads((root/'assignment-000.json').read_text())
            self.assertEqual(len({r['limit_seconds'] for r in assigned['runs']}),1)
            self.assertNotIn('prefix',json.dumps(assigned))
            entry=json.loads((root/'ledger.json').read_text())[0]
            self.assertEqual(entry['bound_seconds'],12965+sum(r['limit_seconds'] for r in entry['runs']))

    def test_dispatch_rejects_phase_budget_and_shared_grant_before_real_submission(self):
        for kind in ['budget','grant']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                with self.assertRaises(ValueError):
                    self.dispatch(root,state=dict(spent_seconds=3583*3600-1,reserved_seconds=0) if kind=='budget' else None,
                        grant=str(15000*3600) if kind=='grant' else '0')
                self.assertEqual(json.loads((root/'ledger.json').read_text()),[])

    def test_dispatch_never_exceeds_two_active_streams(self):
        ledger=[dict(array_id=100+i,runs=[r]) for i,r in enumerate(self.runs[:2])]
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.dispatch(Path(directory),ledger=ledger)
            self.assertFalse(result['submitted']);self.assertEqual(result['status']['active_streams'],2)
            self.assertEqual(calls,[])

    def test_dispatch_waits_for_scouts_before_the_remaining_228_runs(self):
        ledger=[dict(array_id=100,runs=[r for r in self.runs if r['phase']!='conditional'])]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plan=self.wrapper(root);atomic_json(root/'plan.json',plan)
            (root/'plan.sha256').write_text(grid.checksum(root/'plan.json'));atomic_json(root/'ledger.json',ledger)
            status=dict(active_streams=0,scout_gate=dict(passed=False,failed_cost=False))
            with patch.object(grid,'require_authorization',return_value=self.registered),patch.object(grid,'balance',return_value=(dict(spent_seconds=12965,reserved_seconds=0),[])),patch.object(grid,'inspect',return_value=status):
                result=grid.dispatch(root,lambda command:self.fail('must not contact scheduler before scout gate'))
            self.assertTrue(result['waiting_for_scout_gate'])

    def test_terminal_timeout_sets_stop_and_cannot_be_resubmitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=next(r for r in self.runs if r['phase']=='ordinary')
            status=grid.inspect(root,{},self.registered,[dict(array_id=10,runs=[run])],
                [dict(JobID='10_0',State='TIMEOUT',CPUTimeRAW='100')])
            self.assertEqual(status['terminal_failures'][0]['state'],'TIMEOUT')
            self.assertTrue((root/'STOP.json').exists())
            result,calls=self.dispatch(root)
            self.assertTrue(result['stopped']);self.assertEqual(calls,[])

    def test_node_runs_only_the_collector_with_seed_specific_full_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=next(r for r in self.runs if r['phase']=='ordinary' and r['seed']==47)
            plan=self.wrapper(root);assigned=dict(plan_sha256='scope',runs=[run],measurement_repository='/measurement',binary='/binary',operations_commit='operations')
            atomic_json(root/'assignment.json',assigned);(root/run['relative_directory']).parent.parent.mkdir(parents=True)
            commands=[]
            def output(command,**kwargs):
                if 'rev-parse' in command:return grid.MEASUREMENT_COMMIT
                if 'status' in command:return ''
                return f"JobId=1000 ArrayJobId=100 ArrayTaskId=0 TimeLimit={run['limit_seconds']//60}:00 RunTime=00:00:00 NumCPUs=1 Partition=lem-cpu-normal MinMemoryNode=2G"
            def collect(command,**kwargs):
                commands.append(command);self.completion(root,run)
            with patch.object(launch_grid,'load_plan',return_value=plan),patch.object(launch_grid,'require_authorization',return_value=self.registered),patch.object(launch_grid,'checksum',return_value=grid.BINARY_SHA256),patch.object(launch_grid.subprocess,'check_output',side_effect=output),patch.object(launch_grid.subprocess,'run',side_effect=collect),patch.dict(os.environ,SLURM_ARRAY_JOB_ID='100',SLURM_ARRAY_TASK_ID='0',SLURM_JOB_ID='1000'):
                launch_grid.launch(root,root/'assignment.json',0)
            self.assertEqual(len(commands),1);self.assertTrue(commands[0][2].endswith('/collector.py'))
            manifest=json.loads((root/run['relative_directory']).parent.parent.joinpath('manifest.json').read_text())
            self.assertEqual(manifest['seed'],47);self.assertNotIn('per_start_diagnostics',manifest['configuration'])
            self.assertEqual(manifest['configuration']['targets'][-1],2000000)
            self.assertFalse((root/run['relative_directory']/'rows.jsonl').exists())

    def test_node_rejects_later_large_maze_seed_before_measurement_without_the_scout_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=next(r for r in self.runs if r['phase']=='conditional')
            plan=self.wrapper(root);atomic_json(root/'assignment.json',dict(plan_sha256='scope',runs=[run]))
            atomic_json(root/'scout-cost-gate.json',dict(passed=False))
            with patch.object(launch_grid,'load_plan',return_value=plan),patch.object(launch_grid,'require_authorization',return_value=self.registered),patch.object(launch_grid.subprocess,'run') as collector:
                with self.assertRaises(ValueError):launch_grid.launch(root,root/'assignment.json',0)
                collector.assert_not_called()
            self.assertTrue((root/'STOP.json').exists())


if __name__=='__main__':unittest.main()
