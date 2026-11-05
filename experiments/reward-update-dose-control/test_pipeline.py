import copy
import unittest
from unittest.mock import patch

from protocol import ARM, G, S, ARMS, BLOCKS, configurations
from analyze import calculate


class PipelineTests(unittest.TestCase):
    def test_real_pipeline_uses_test_success_final_pairs_equal_block_mass_and_winner_ties(self):
        configs=[dict(config,targets=[11,37]) for config in configurations()]
        inputs=dict(new={},success={},pairs={})
        storage,truth,expected={},{},{}
        for m,config in enumerate(configs):
            maze=config['task']
            truth[maze]={}
            for b,n in enumerate([2,3,5]):
                for i in range(n):
                    truth[maze][(0,0),(b,i)]=dict(cap=1,shortest_distance=1,value=1000.,action_values=[1000.,0.,0.,0.,0.,0.,0.,0.],optimal_actions=[0])
            for a,arm in enumerate(ARMS):
                for seed in range(42,62):
                    s=seed-42
                    rows=[]
                    for p,point in enumerate(config['targets']):
                        blocks={}
                        credits=[]
                        errors=[]
                        eligible=[]
                        for b,(name,n) in enumerate(zip(BLOCKS,[2,3,5])):
                            records=[]
                            for i in range(n):
                                action=(a+s+m+b+i+p)%2
                                fallback=(2*a+s+m+b+i+p)%7==0
                                goals=(a+2*s+m+b+i+p)%4
                                slope=.7+.2*a+.001*s+.01*m+.02*p
                                q=1000. if action==0 else 0.
                                estimate=slope*q+10+3*a+s
                                credit=0 if fallback else goals/3
                                credits.append(credit/(3*n))
                                errors.append((estimate-q)/(3*n))
                                if not fallback: eligible.append((1/(3*n),action,estimate>1000.))
                                records.append(dict(start='(0, 0)',goal=f'Goal {{ symbols: [Token({b}), Token({i})] }}',weight=1/n,
                                    first_action=action,shortest_distance=1,estimated_first_action_value=estimate,
                                    success=action==0,discounted_return=q,
                                    first_decision=dict(random_fallback=fallback,winners=[] if fallback else [dict(specified_goal_attributes=int(j<goals)) for j in range(3)])))
                            blocks[name]=dict(success=.01*(a+1)**2*(1+.01*s+.02*m+.03*b+.1*p),starts=records)
                        route=lambda skip: dict(action_set_rules=100,skipped_without_goal=skip,skipped_with_goal=0)
                        rows.append(dict(nominal_step=point,evaluation_blocks=blocks,
                            replay_diagnostics=dict(reward_updates=dict(original=route(0),relabeled=route(50+a)))))
                        if p==1:
                            mass=sum(weight for weight,_,_ in eligible)
                            f=sum(weight for weight,action,_ in eligible if action==0)/mass
                            expected[maze,arm,seed]=dict(winner=sum(credits),error=sum(errors),slope=slope,
                                F=f,F_mod=float(f>=.5),over=sum(weight for weight,_,over in eligible if over)/mass,
                                primary=.01*(a+1)**2*(1+.01*s+.02*m+.06+.05),final=.01*(a+1)**2*(1+.01*s+.02*m+.06+.1))
                    key=config['id'],arm,seed
                    compact_path=('success',*key)
                    pair_path=('pairs',*key)
                    if arm==ARM:
                        inputs['new'][key]=pair_path
                    else:
                        inputs['success'][key]=compact_path
                        inputs['pairs'][key]=pair_path
                    storage[pair_path]=rows
                    compact=copy.deepcopy(rows)
                    for row in compact:
                        for block in row['evaluation_blocks'].values(): block['starts']=None
                    storage[compact_path]=compact
        with patch('analyze.read_rows',side_effect=lambda path:iter(storage[path])):
            result=calculate(configs,inputs,truth)
            self.assertEqual(len(result['registered']),96)
            for row in result['registered']:
                key={'winner_goal_mass':'winner','estimate_minus_q':'error','within_start_slope':'slope','estimate_above_v':'over'}.get(row['metric'],row['metric'])
                seeds=range(42,62) if row['group']=='success' else range(42,47)
                expected_values=[expected[row['maze'],row['left'],seed][key]-expected[row['maze'],row['right'],seed][key] for seed in seeds]
                for actual,wanted in zip(row['seed_differences'],expected_values): self.assertAlmostEqual(actual,wanted,places=10)
            key=configs[0]['id'],G,42
            storage[inputs['success'][key]]=storage[inputs['success'][key]][:-1]
            with self.assertRaises(ValueError): calculate(configs,inputs,truth)
