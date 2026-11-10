import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import HERE,configurations
from cost_pilot_registration import build,configuration,TARGETS,PRIOR_CPU_SECONDS,MEASUREMENT_COMMIT,BINARY_SHA256
from cost_pilot import require_authorization,registered_runs,dispatch,balance
from execution import checksum
from grid_protocol import atomic_json
from prefix_identity import compare


class CostPilotTests(unittest.TestCase):
    def plan(self,root):
        return dict(result_root=str(root),registration=build(),registration_sha256=checksum(HERE/'cost-pilot.json'),
            prior=dict(root='/tmp/prior',files={}),phase_cpu_hour_limit=15,concurrency=20,partition='lem-cpu-normal',
            measurement_commit=MEASUREMENT_COMMIT,binary_sha256=BINARY_SHA256,configurations=[configuration(c) for c in configurations()],
            stop_path=str(root/'STOP.json'),measurement_repository='/tmp/repo',binary='/tmp/binary',
            operations_commit='operations',operations_manifest=str(root/'operations/manifest.json'))

    def test_short_pilot_is_exact_registered_prefix_in_all_cells_and_charges_prior_budget(self):
        value=build();self.assertEqual(len(value['pilot_runs']),138)
        self.assertEqual(value['maximum_combined_cpu_hours'],14.725)
        self.assertEqual(value['prior_cpu_seconds'],9810)
        for original in configurations():
            short=configuration(original)
            self.assertEqual(short['targets'],TARGETS);self.assertNotEqual(short['targets'],original['targets'])
            self.assertEqual(short['registration'],original['registration']);self.assertEqual(short['cap'],original['cap'])
            self.assertTrue(short['per_start_diagnostics'])
        with patch('cost_pilot.verify_prior'),patch('cost_pilot.verify_operations'):
            p=self.plan(Path('/tmp/test'));runs=registered_runs(p)
            self.assertEqual(len(runs),138);self.assertTrue(all(r['seed']==42 for r in runs))
            bad=copy.deepcopy(p);bad['configurations'][0]['targets']=[1000,3000]
            with self.assertRaises(ValueError):require_authorization(bad)
            bad=copy.deepcopy(p);bad['registration']['prior_cpu_seconds']=0
            with self.assertRaises(ValueError):require_authorization(bad)

    def test_dispatch_includes_prior_cpu_before_grant_and_test_only_and_keeps_all_attempts(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.plan(root);atomic_json(root/'plan.json',p);(root/'plan.sha256').write_text(checksum(root/'plan.json'))
            atomic_json(root/'ledger.json',[]);calls=[]
            def execute(command):
                calls.append(command)
                if command[0]=='sshare':return '23000000'
                if command[0]=='squeue':return ''
                if command[0]=='sbatch':return 'accepted' if '--test-only' in command else '12345'
                raise AssertionError(command)
            state=dict(spent_seconds=9810,reserved_seconds=0,prior_cpu_seconds=9810,new_cpu_seconds=0)
            with patch('cost_pilot.verify_prior'),patch('cost_pilot.verify_operations'),patch('cost_pilot.balance',return_value=(state,[])):
                result=dispatch(root,execute)
            self.assertEqual(result['runs'],6);self.assertEqual(result['bound_cpu_hours'],(9810+3600)/3600)
            self.assertEqual([c[0] for c in calls],['sshare','squeue','sbatch','sbatch'])
            self.assertIn('--test-only',calls[-2]);self.assertEqual(json.loads((root/'ledger.json').read_text())[0]['array_id'],12345)

    def test_prior_spend_or_reservations_cannot_be_reset_and_two_streams_are_the_limit(self):
        for case in ['spent','reserved','streams','stop']:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
                root=Path(d);p=self.plan(root);atomic_json(root/'plan.json',p);(root/'plan.sha256').write_text(checksum(root/'plan.json'))
                ledger=[];records=[]
                state=dict(spent_seconds=9810,reserved_seconds=0)
                if case=='spent':state['spent_seconds']=14.9*3600
                if case=='reserved':state['reserved_seconds']=12*3600
                if case=='stop':(root/'STOP.json').write_text('{}')
                if case=='streams':
                    run=dict(configuration_id='x',agent='x',limit_seconds=300)
                    ledger=[dict(array_id=1,runs=[run]),dict(array_id=2,runs=[run])]
                    records=[dict(JobID='1_0',State='RUNNING'),dict(JobID='2_0',State='PENDING')]
                atomic_json(root/'ledger.json',ledger)
                with patch('cost_pilot.verify_prior'),patch('cost_pilot.verify_operations'),patch('cost_pilot.balance',return_value=(state,records)):
                    def execute(command):raise AssertionError('no submission allowed')
                    if case=='streams':self.assertFalse(dispatch(root,execute)['submitted'])
                    else:
                        with self.assertRaises(ValueError):dispatch(root,execute)

    def test_prefix_comparison_covers_all_values_counters_and_pair_profiles_in_float_bits(self):
        old=[dict(nominal_step=t,seed=42,task='maze',agent='acs2',host='old',cpu_model='cpu',wall_seconds_train=1.0,
                  population_classifiers=i+3,replay_diagnostics=dict(per_goal_rewards=dict(updated_entries=i+7)),
                  evaluation_blocks=dict(test=dict(starts=[dict(first_action=i,first_decision=dict(r=0.1+i))]))) for i,t in enumerate(TARGETS)]
        fresh=copy.deepcopy(old)
        for r in fresh:r['host']='fresh';r['wall_seconds_train']=999.0
        self.assertTrue(compare(old+[dict(nominal_step=100000)],fresh)['learning_and_per_pair_fields_bit_identical'])
        for mutate in [lambda r:r.update(seed=43),lambda r:r.update(population_classifiers=100),
            lambda r:r['replay_diagnostics']['per_goal_rewards'].update(updated_entries=999),
            lambda r:r['evaluation_blocks']['test']['starts'][0]['first_decision'].update(r=0.10000000000000002),
            lambda r:r['evaluation_blocks']['test']['starts'][0].update(first_action=7)]:
            changed=copy.deepcopy(fresh);mutate(changed[0])
            with self.assertRaises(ValueError):compare(old,changed)
        with self.assertRaises(ValueError):compare(old,fresh[:-1])

    def test_combined_accounting_deduplicates_raw_ids_and_rejects_a_missing_prior_attempt(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);prior=root/'prior';prior.mkdir()
            first=[dict(limit_seconds=1200) for _ in range(6)]
            second=[dict(limit_seconds=600) for _ in range(12)]
            atomic_json(prior/'ledger.json',[dict(array_id=6296853,runs=first),dict(array_id=6296855,runs=second)])
            plan=dict(prior=dict(root=str(prior)))
            header='JobID|JobIDRaw|CPUTimeRAW|State|\n'
            lines=[]
            amounts=[573]*6+[571]*4+[508]*2+[512]*6
            for i,n in enumerate(amounts):
                job=f'6296853_{i}' if i<6 else f'6296855_{i-6}'
                lines.append(f'{job}|{100+i}|{n}|FAILED|\n')
            lines.append('500_0|999|15|RUNNING|\n')
            ledger=[dict(array_id=500,runs=[dict(limit_seconds=300)])]
            state,records=balance(root,plan,ledger,lambda _:header+''.join(lines)+lines[0])
            self.assertEqual(state['spent_seconds'],9825);self.assertEqual(state['reserved_seconds'],285)
            self.assertEqual(len(records),19);self.assertEqual(state['prior_cpu_seconds'],9810)
            with self.assertRaises(ValueError):balance(root,plan,ledger,lambda _:header+''.join(lines[1:]))

    def test_preparation_driver_keeps_measurement_clone_pinned_and_freezes_prior_and_operations(self):
        import shutil
        import cost_pilot
        from protocol import ROOT
        with tempfile.TemporaryDirectory() as d:
            home=Path(d).resolve();repo=home/'repo';ops=home/'ops';ops.mkdir()
            root=home/'tu_runs/per-goal-reward-ablation/cost-pilot'
            fixture=repo/'fixtures/goal_measurement_regression.json';fixture.parent.mkdir(parents=True);shutil.copyfile(ROOT/'fixtures/goal_measurement_regression.json',fixture)
            binary=repo/'target/x86_64-unknown-linux-musl/release/acs2-measure';binary.parent.mkdir(parents=True);binary.write_bytes(b'binary')
            atomic_json(ops/'manifest.json',dict(commit='operations',files={}))
            prior=dict(root=str(home/'prior'),files={})
            actual_checksum=cost_pilot.checksum
            def checksum(path):return BINARY_SHA256 if Path(path)==binary else actual_checksum(path)
            def execute(command):
                if command[-2:]==['rev-parse','HEAD']:return MEASUREMENT_COMMIT
                if command[-2:]==['status','--porcelain']:return ''
                raise AssertionError(command)
            with patch('cost_pilot.Path.home',return_value=home),patch('cost_pilot.execute',side_effect=execute),patch('cost_pilot.verify_prior',return_value=prior),patch('cost_pilot.checksum',side_effect=checksum):
                value=cost_pilot.prepare(root,repo,ops,home/'prior')
            self.assertEqual(value['measurement_commit'],MEASUREMENT_COMMIT)
            self.assertEqual(value['operations_commit'],'operations')
            plan=json.loads((root/'plan.json').read_text());self.assertEqual(plan['prior'],prior)
            self.assertEqual(plan['configurations'][0]['targets'],TARGETS)

    def test_prefix_file_driver_requires_authentic_old_rows_and_a_completed_fresh_run(self):
        from prefix_identity import verify_cell
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);old=root/'old';fresh=root/'fresh';old.mkdir();fresh.mkdir()
            run=dict(configuration_id='cell',agent='acs2',relative_directory='data')
            path=old/'data';path.mkdir();atomic_json(old/'ledger.json',[dict(runs=[run])])
            rows=[dict(nominal_step=t,seed=42,agent='acs2',test=dict(first_action=i,value=float(i))) for i,t in enumerate(TARGETS)]
            content=''.join(json.dumps(r)+'\n' for r in rows)
            (path/'rows.jsonl').write_text(content);(fresh/'rows.jsonl').write_text(content)
            atomic_json(old/'archive-checksums.json',{'data/rows.jsonl':checksum(path/'rows.jsonl')})
            completion=dict(exit_code=0,failure=None,sha256={'rows.jsonl':checksum(fresh/'rows.jsonl')})
            atomic_json(fresh/'completion.json',completion)
            self.assertTrue(verify_cell(old,run,fresh/'rows.jsonl')['learning_and_per_pair_fields_bit_identical'])
            completion['failure']='failed';atomic_json(fresh/'completion.json',completion)
            with self.assertRaises(ValueError):verify_cell(old,run,fresh/'rows.jsonl')
            completion['failure']=None;atomic_json(fresh/'completion.json',completion)
            (path/'rows.jsonl').write_text(content+'\n')
            with self.assertRaises(ValueError):verify_cell(old,run,fresh/'rows.jsonl')
