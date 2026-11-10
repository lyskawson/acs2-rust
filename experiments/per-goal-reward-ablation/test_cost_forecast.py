import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from protocol import ARMS,A,A1,E,S,E1,S1,configurations,new_cells
from cost_forecast import growth,project,forecast
from cost_evidence import profile,export_checkpoints
from execution import checksum


def curve(task,arm,kind,rate=0.0002,post_rate=0.0003,stop=2000000,seed=42,identifier='cell'):
    checkpoints=[];previous=0;total=0
    for i,t in enumerate([1000,3000,10000,30000,100000,300000,1000000,2000000,4000000]):
        if t>stop:break
        delta=(t-previous)*(rate if t<=30000 else post_rate);total+=delta
        checkpoints.append(dict(step=t,actual_steps=t,train_cpu_seconds_delta_estimate=delta,
            eval_cpu_seconds_delta_estimate=1.0 if t<=30000 else 2.0,
            train_cpu_seconds_cumulative_estimate=total,eval_cpu_seconds_cumulative_estimate=4.0+2.0*max(i-3,0) if i>=3 else i+1.0))
        previous=t
    cpu=total+sum(p['eval_cpu_seconds_delta_estimate'] for p in checkpoints)
    return dict(task=task,agent=arm,seed=seed,kind=kind,configuration_id=identifier,checkpoints=checkpoints,
        complete=kind!='stopped_long',process_cpu_seconds_wait4=cpu,allocation_cpu_seconds=math.ceil(cpu)+3,
        peak_rss_kib=10000,source_sha256='a'*64)


class CostForecastTests(unittest.TestCase):
    def data(self):
        configs=configurations();pilots=[];longer=[];archives=[]
        ids={c['task']:c['id'] for c in configs}
        for i,(c,arm) in enumerate(new_cells()):
            pilots.append(curve(c['task'],arm,'pilot_30000',rate=0.00003*(i+1),stop=30000,identifier=c['id']))
        for task in ['mazee1_alcs','mazee2_alcs','mazee3_alcs']:
            for ai,arm in enumerate(ARMS):
                value=curve(task,arm,'stopped_long',rate=0.0001*(ai+1),post_rate=0.0002*(ai+1),stop=100000,identifier=ids[task])
                value['source_sha256']=format(len(longer)+1,'064x');longer.append(value)
        for c in configs[:6]:
            for arm in [A,E,S,E1,S1]:
                for seed in range(42,62):
                    archives.append(curve(c['task'],arm,'archive_full',rate=0.0001*(seed-30),post_rate=0.00015*(seed-30),stop=c['targets'][-1],seed=seed,identifier=c['id']))
        identity=dict(verified=True,cells=18,checkpoint_comparisons=72,
            checks=[dict(configuration_id=p['configuration_id'],agent=p['agent'],prior_sha256=p['source_sha256']) for p in longer])
        return pilots,archives,longer,identity

    def test_full_and_stopped_growth_use_measured_training_rates_and_explicit_tail_models(self):
        full=curve('x',A,'archive_full',rate=2.0,post_rate=3.0)
        r=growth(full,2000000);self.assertEqual(r['training_low'],1.5);self.assertEqual(r['training_high'],1.5)
        self.assertEqual(r['evaluation_low'],2);self.assertEqual(r['evaluation_high'],2)
        stopped=curve('x',A,'stopped_long',rate=2.0,post_rate=4.0,stop=100000)
        r=growth(stopped,2000000)
        self.assertAlmostEqual(r['training_low'],2.0)
        self.assertGreater(r['training_high'],2.0);self.assertGreater(r['rate_exponent'],0)
        missing=curve('x',A,'stopped_long',stop=30000)
        self.assertEqual(growth(missing,2000000)['post_30k_points'],0)
        self.assertEqual(growth(missing,2000000)['training_high'],1)

    def test_projection_keeps_cpu_components_checkpoint_counts_and_observed_censored_cost(self):
        pilot=curve('x',A,'pilot_30000',rate=0.001,stop=30000)
        bounds=dict(training_low=1,training_high=2,evaluation_low=1,evaluation_high=3)
        target=[1000,3000,10000,30000,100000,300000,1000000,2000000]
        result=project(pilot,target,bounds,[])
        self.assertEqual(result['low_cpu_seconds'],2000+8+3)
        self.assertEqual(result['high_cpu_seconds'],30+2*1970+4+4*3+3*2)
        long=curve('x',A,'stopped_long',stop=100000);long['allocation_cpu_seconds']=9999
        self.assertGreaterEqual(project(pilot,target,bounds,[long])['low_cpu_seconds'],9999)

    def test_forecast_covers_138_cells_all_20_full_seeds_and_unchanged_4m_horizon(self):
        pilots,archives,longer,identity=self.data();r=forecast(pilots,archives,longer,10000,identity)
        self.assertEqual(len(r['cells']),138);self.assertEqual(r['new_full_runs'],2760)
        self.assertTrue(all(c['full_runs_remaining']==20 for c in r['cells']))
        c=next(c for c in r['cells'] if c['task']=='woods14' and c['agent']==A1)
        self.assertEqual(c['registered_budget_steps'],4000000)
        self.assertGreater(c['forecast_registered_high_cpu_seconds'],c['forecast_2m_high_cpu_seconds'])
        self.assertGreaterEqual(r['requested_cpu_hour_limit'],r['hard_reservations_with_pilot_cpu_hours'])
        self.assertGreater(r['future_full_grid_high_cpu_hours'],r['future_full_grid_low_cpu_hours'])
        self.assertEqual(r['cells'][0]['forecast_registered_high_cpu_seconds'],max(c['forecast_registered_high_cpu_seconds'] for c in r['cells']))
        self.assertFalse(r['success_summary']);self.assertFalse(r['budgets_changed'])

    def test_forecast_rejects_incomplete_or_unauthenticated_evidence_and_never_uses_success(self):
        pilots,archives,longer,identity=self.data();expected=forecast(pilots,archives,longer,10000,identity)
        changed=copy.deepcopy(pilots)
        for p in changed:
            p['success']=0.999
            for c in p['checkpoints']:c['success']=0.001
        self.assertEqual(forecast(changed,archives,longer,10000,identity),expected)
        with self.assertRaises(ValueError):forecast(pilots[:-1],archives,longer,10000,identity)
        bad=copy.deepcopy(identity);bad['verified']=False
        with self.assertRaises(ValueError):forecast(pilots,archives,longer,10000,bad)
        bad=copy.deepcopy(longer);bad[0]['source_sha256']='b'*64
        with self.assertRaises(ValueError):forecast(pilots,archives,bad,10000,identity)

    def test_cost_extraction_and_csv_drop_success_and_keep_every_cost_population_entry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=[];resources=[]
            for i,t in enumerate([1000,3000,10000,30000]):
                rows.append(dict(task='x',agent=A1,seed=42,nominal_step=t,actual_steps=t+1,population_classifiers=10+i,
                    population_numerosity=20+i,population_logical_bytes=10000+i,population_mark_entries=30+i,
                    replay_diagnostics=dict(per_goal_rewards=dict(existing_pairs=100+i,updated_entries=40+i,reward_updates=dict(original=1000+i,relabeled=0))),
                    evaluation_blocks=dict(test=dict(success=0.999,starts=[]))))
                resources.append(dict(nominal_step=t,train_cpu_seconds_estimate=i+1.0,eval_cpu_seconds_estimate=0.2,
                    process_cpu_seconds_at_observation=10+i,observed_elapsed_seconds=20+i,peak_rss_kib=30000+i,
                    maximum_sample_gap_seconds=0.03,row_bytes=1000+i))
            for name,values in [('rows.jsonl',rows),('resources.jsonl',resources)]:
                (root/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
            completion=dict(exit_code=0,failure=None,user_cpu_seconds=10.0,system_cpu_seconds=1.0,elapsed_seconds=12.0,peak_rss_kib=40000,
                sha256={name:checksum(root/name) for name in ['rows.jsonl','resources.jsonl']})
            (root/'completion.json').write_text(json.dumps(completion))
            meta=dict(task='x',agent=A1,seed=42,kind='pilot_30000');account=dict(CPUTimeRAW='15',JobIDRaw='123')
            result=profile(root/'rows.jsonl',meta,account)
            self.assertEqual(result['checkpoints'][-1]['train_cpu_seconds_cumulative_estimate'],10)
            self.assertEqual(result['process_cpu_seconds_wait4'],11)
            self.assertEqual(result['checkpoints'][-1]['per_goal_entries']['existing_pairs'],103)
            self.assertNotIn('success',json.dumps(result))
            export_checkpoints([result],root/'costs.csv');text=(root/'costs.csv').read_text()
            self.assertNotIn('success',text);self.assertIn('process_cpu_seconds_at_observation',text);self.assertIn('103',text)

    def test_final_cost_plan_preserves_full_protocols_and_requires_future_approval(self):
        from cost_plan import prepare
        pilots,archives,longer,identity=self.data();estimated=forecast(pilots,archives,longer,12000,identity)
        audit=dict(verified=True,prior_cpu_seconds=9810,fresh_cpu_seconds=2190,all_attempts_cpu_seconds=12000,
            prefix_identity=dict(checks=[dict(configuration_id='cell',agent=A,seed=42,checkpoints=[1000,3000,10000,30000],learning_and_per_pair_fields_bit_identical=True,prior_sha256='d'*64,fresh_sha256='e'*64)]),
            copies=[dict(manifest_sha256='f'*64),dict(manifest_sha256='a'*64)],operations_commit='operations',operations_manifest_sha256='b'*64)
        plan=prepare(audit,estimated,'b'*64,'c'*64)
        self.assertEqual(plan['stage'],'grid_pending_approval');self.assertEqual(plan['remaining_seeds'],list(range(42,62)))
        self.assertEqual(plan['pilot_allocation_cpu_seconds'],12000)
        self.assertEqual(plan['configurations'],configurations())
        self.assertEqual(plan['pilot']['horizon'],30000)
        self.assertFalse(plan['pilot']['success_summary']);self.assertEqual(plan['new_full_runs'],2760)
        self.assertFalse(plan['preset']['do_ga']);self.assertEqual(plan['preset']['epsilon'],0.8)
        with self.assertRaises(ValueError):prepare(dict(audit,verified=False),estimated,'b'*64,'c'*64)

    def test_archive_copy_verification_driver_rejects_modified_cost_evidence(self):
        from cost_plan import verify_copy
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'costs.json').write_text('{"cpu":12}')
            (root/'archive-checksums.json').write_text(json.dumps({'costs.json':checksum(root/'costs.json')}))
            self.assertEqual(verify_copy(root)['files'],1)
            (root/'costs.json').write_text('{"cpu":11}')
            with self.assertRaises(ValueError):verify_copy(root)

    def test_growth_envelope_retains_joint_seed_level_variation_against_seed42(self):
        from cost_forecast import envelope
        archives=[curve('m'+str(maze),A,'archive_full',rate=float(seed-41),post_rate=1.5*(seed-41),seed=seed) for maze in range(6) for seed in range(42,62)]
        longer=[curve('long'+str(i),A,'stopped_long',rate=1.0,post_rate=1.0,stop=100000) for i in range(3)]
        value=envelope(A,2000000,archives,longer)
        self.assertEqual(value['archived_seed_training_level_max'],20)
        self.assertEqual(value['training_high'],30)
        self.assertEqual(value['training_low'],1)
