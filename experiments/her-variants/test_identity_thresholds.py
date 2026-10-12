import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import unittest

from identity import compare
from thresholds import BASELINE_CROSSINGS, THRESHOLDS, crossings, mean_curve, success_count


class ThresholdTests(unittest.TestCase):
    def test_archived_baseline_counts_reproduce_both_exact_thresholds_and_all_crossings(self):
        fixture=json.loads((Path(__file__).resolve().parents[2]/'fixtures/pair_baseline_success_counts.json').read_text())
        self.assertEqual(fixture['seeds'],list(range(42,62)))
        self.assertEqual({config['id'] for config in fixture['configurations']},set(THRESHOLDS))
        for config in fixture['configurations']:
            for agent,rows in config['agents'].items():
                curve=mean_curve(rows,config['targets'])
                self.assertEqual(curve[config['targets'][-1]],Fraction(THRESHOLDS[config['id']][agent+'_final']))
                observed=crossings(rows,config['targets'],THRESHOLDS[config['id']])
                self.assertEqual(observed,{name:values[agent] for name,values in BASELINE_CROSSINGS[config['id']].items()})

    def test_an_arm_exactly_at_the_threshold_is_not_missed_by_floating_rounding(self):
        counts=[53]*6+[52]*14
        rows=[dict(seed=seed,nominal_step=1000,evaluation_blocks=dict(test=dict(success=count/222,
                   evaluation_pairs=222,evaluation_distribution='exhaustive_exact_weights')))
              for seed,count in zip(range(42,62),counts)]
        self.assertEqual(mean_curve(rows,[1000])[1000],Fraction(523,2220))
        self.assertLess(float(Fraction(523,2220)),0.23558558558558576)
        self.assertEqual(crossings(rows,[1000],{'exact':'523/2220','higher':'524/2220'}),{'exact':1000,'higher':None})
        with self.assertRaises(ValueError): mean_curve(rows[:-1],[1000])
        with self.assertRaises(ValueError): mean_curve(rows+[rows[0]],[1000])
        with self.assertRaises(ValueError): crossings(rows,[1000],{'floating':float(Fraction(523,2220))})

    def test_crossings_use_first_nominal_points_without_interpolation(self):
        rows=[dict(seed=seed,nominal_step=point,actual_steps=point+4,successes=count,evaluation_pairs=10)
              for seed in range(42,62) for point,count in [(1000,1),(3000,5),(10000,4)]]
        self.assertEqual(crossings(rows,[1000,3000,10000],{'reached':'1/2','absent':'3/5'}),{'reached':3000,'absent':None})
        with self.assertRaises(ValueError): mean_curve(rows,[3000,1000,10000])

    def test_threshold_conversion_refuses_non_counts_and_non_exhaustive_success(self):
        row=dict(evaluation_blocks=dict(test=dict(success=1/3,evaluation_pairs=3,evaluation_distribution='exhaustive_exact_weights')))
        self.assertEqual(success_count(row),(1,3))
        for key,value in [('success',0.1),('success',float('nan')),('evaluation_pairs',0),('evaluation_distribution','sampled')]:
            changed=copy.deepcopy(row)
            changed['evaluation_blocks']['test'][key]=value
            with self.assertRaises(ValueError): success_count(changed)


class IdentityTests(unittest.TestCase):
    def rows(self):
        return [dict(agent='her_values',seed=seed,nominal_step=point,knowledge=0.5,commit='old',
                     learning_update_counts=dict(full=point,values_only=2*point),
                     evaluation_blocks={name:dict(success=0.25) for name in ['training','validation','test']})
                for seed in range(42,47) for point in [1000,3000]]

    def test_tuning_identity_uses_shared_validation_and_pins_all_learning_and_knowledge_bits(self):
        reference=self.rows()
        for row in reference: row['evaluation_blocks']={'validation':row['evaluation_blocks']['validation']}
        current=self.rows()
        for row in current: row.update(commit='new',arm='her_values_p2of3',wall_seconds_train=5)
        self.assertEqual(compare(current,reference,range(42,47),[1000,3000],['validation'])['rows'],10)
        for key,value in [('knowledge',math.nextafter(0.5,1)),('learning_update_counts',dict(full=0,values_only=0))]:
            changed=copy.deepcopy(current)
            changed[0][key]=value
            with self.assertRaises(ValueError): compare(changed,reference,range(42,47),[1000,3000],['validation'])
        changed=copy.deepcopy(current)
        changed[0]['evaluation_blocks']['validation']['success']=0.0
        with self.assertRaises(ValueError): compare(changed,reference,range(42,47),[1000,3000],['validation'])
        with self.assertRaises(ValueError): compare(current[:-1],reference,range(42,47),[1000,3000],['validation'])

    def test_pilot_identity_includes_all_three_blocks_and_catches_a_changed_test_result(self):
        reference=self.rows()
        current=copy.deepcopy(reference)
        blocks=['training','validation','test']
        self.assertEqual(compare(current,reference,[42],[1000,3000],blocks)['rows'],2)
        current[0]['evaluation_blocks']['test']['success']=0.0
        with self.assertRaises(ValueError): compare(current,reference,[42],[1000,3000],blocks)
        current=copy.deepcopy(reference)
        current[0]['knowledge']=-0.0
        reference[0]['knowledge']=0.0
        with self.assertRaises(ValueError): compare(current,reference,[42],[1000,3000],blocks)


if __name__ == '__main__': unittest.main()
