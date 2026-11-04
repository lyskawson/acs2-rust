import json
import math
from pathlib import Path
import tempfile
import unittest

from protocol import ARM, ARMS, REFERENCES
from analysis import truth_index
from analyze import calculate


class AnalysisPipelineTests(unittest.TestCase):
    def test_full_driver_distinguishes_checkpoint_means_final_points_pair_weights_and_references(self):
        configs=[dict(id=maze,task=maze,cap=3,targets=[10,20]) for maze in ['maze4','maze5','maze6','maze7','woods14','mazema']]
        model=truth_index(dict(gamma=.95,reward=1000,actions=8,cap=3,pairs=[dict(start=[1,1],goal=[2,i+1],
            shortest_distance=2,value=950.0,action_values=[950.0 if action==i%3 else 902.5 for action in range(8)],
            optimal_actions=[i%3]) for i in range(7)]))
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            inputs=dict(new={},success={},pairs={})
            for config in configs:
                for ai,arm in enumerate(ARMS):
                    for seed in range(42,62):
                        rows=[]
                        for point in (10,20):
                            blocks={}
                            for bi,indices in enumerate(([0],[1,2],[3,4,5,6])):
                                records=[]
                                for pair in indices:
                                    action=(seed+ai+bi+point//10)%3
                                    records.append(dict(start='(1, 1)',goal='Goal { symbols: [Token(2), Token(%d)] }'%(pair+1),
                                        shortest_distance=2,weight=1/len(indices),first_action=action,success=action==pair%3,
                                        estimated_first_action_value=800+ai*80+(seed-42)*2+point+pair*3,
                                        first_decision=dict(random_fallback=False,winners=[dict(specified_goal_attributes=ai%2)])))
                                blocks[('training','validation','test')[bi]]=dict(starts=records,
                                    success=.1+point*.01+ai*.005*(seed-41))
                            rows.append(dict(nominal_step=point,evaluation_blocks=blocks))
                        path=root/(config['id']+'_'+arm+str(seed)+'.jsonl')
                        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                        key=config['id'],arm,seed
                        inputs['new' if arm==ARM else 'success'][key]=path
                        if arm!=ARM and seed<47:
                            inputs['pairs'][key]=path
            result=calculate(configs,inputs,{config['task']:model for config in configs})
            self.assertEqual(len(result['registered']),84)
            self.assertEqual(len(result['success']),360)
            self.assertEqual(len(result['curves']),180)
            self.assertEqual(len(result['supplementary']),450)
            success=next(row for row in result['success'] if row['maze']=='maze4' and row['arm']==ARM and row['seed']==42)
            self.assertAlmostEqual(success['primary'],.25)
            self.assertAlmostEqual(success['final'],.3)
            contrast=next(row for row in result['registered'] if row['maze']=='maze4' and row['reference']=='acs2er_episodic' and row['metric']=='primary')
            self.assertAlmostEqual(contrast['mean'],-.105)
            self.assertGreater(contrast['se'],0)
            values=[row for row in result['supplementary'] if row['maze']=='maze4' and row['arm']==ARM and row['seed']==42]
            pooled=next(row for row in values if row['view']=='pooled')
            uniform=next(row for row in values if row['view']=='uniform_pairs')
            self.assertAlmostEqual(pooled['F'],(0+0+1/4)/3)
            self.assertAlmostEqual(uniform['F'],1/7)
            self.assertAlmostEqual(uniform['C'],3/7)
            self.assertAlmostEqual(uniform['F_minus_C'],-2/7)
            self.assertNotEqual(uniform['F_minus_C'],pooled['F_minus_C'])
            for row in result['supplementary']:
                self.assertAlmostEqual(row['F_minus_C'],row['F_mod_minus_C']+row['F_minus_F_mod'])
            self.assertEqual(result['decisions']['outcome'],'does not help')
            first=next(row for row in result['curves'] if row['maze']=='maze4' and row['arm']==ARM and row['seed']==42 and row['nominal_step']==10)
            self.assertNotEqual(first['F'],pooled['F'])
            bad=dict(configs[0],targets=[10,20,30])
            with self.assertRaisesRegex(ValueError,'every registered checkpoint'):
                calculate([bad],inputs,{'maze4':model})
