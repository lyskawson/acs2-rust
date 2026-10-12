import copy
import math
import json
from pathlib import Path
import unittest

from analysis import analyze_arm, analyze_maze, indexed, metrics, paired, update_counts
from execution import approved
from test_protocol import row_and_configuration


class AnalysisTests(unittest.TestCase):
    def rows(self):
        template,config=row_and_configuration('her_values')
        template.update(wall_seconds_train=1,wall_seconds_eval=1,wall_seconds_total=2)
        rows=[]
        for seed in range(42,62):
            for point,count in [(1000,1),(3000,3)]:
                row=copy.deepcopy(template)
                row.update(seed=seed,nominal_step=point,actual_steps=point)
                row['evaluation_blocks']['test'].update(success=count/4,evaluation_pairs=4)
                row['learning_update_counts'].update(full=2*point,values_only=point)
                rows.append(row)
        return rows,dict(config,targets=[1000,3000])

    def test_primary_averages_points_inside_seed_before_twenty_seed_summary(self):
        rows,config=self.rows()
        result=analyze_arm(rows,config,{'equal':'1/2','above':'4/5'})
        self.assertEqual(result['primary']['n'],20)
        self.assertEqual(result['primary']['mean'],0.5)
        self.assertEqual(result['primary']['se'],0)
        self.assertEqual(result['threshold_crossings'],{'equal':3000,'above':None})
        self.assertEqual(result['exact_mean_test_curve'],{'1000':'1/4','3000':'3/4'})
        self.assertEqual(result['checkpoint_means']['full_steps_per_environment_step']['mean'],2)

    def test_paired_intervals_use_seed_differences_and_registered_t19(self):
        left={'seed_scores':[dict(seed=seed,success=(seed-42)/100) for seed in range(42,62)]}
        right={'seed_scores':[dict(seed=seed,success=(seed-42)/100-0.05) for seed in range(42,62)]}
        result=paired(left,right,'success')
        self.assertAlmostEqual(result['mean'],0.05)
        self.assertLess(result['se'],1e-16)
        right={'seed_scores':[dict(seed=seed,success=0) for seed in range(42,62)]}
        result=paired(left,right,'success')
        self.assertAlmostEqual(result['se'],math.sqrt(35)/100/math.sqrt(20))
        self.assertAlmostEqual(result['ci_high']-result['mean'],2.093024054408263*result['se'])

    def test_final_dataset_cannot_reuse_five_validation_only_tuning_seeds(self):
        rows,config=self.rows()
        with self.assertRaises(ValueError): indexed(rows[:10],config['targets'])
        rows[0]['evaluation_blocks']={'validation':rows[0]['evaluation_blocks']['validation']}
        with self.assertRaises(ValueError): indexed(rows,config['targets'])

    def test_model_volume_and_objective_cost_use_realized_counts_and_keep_routes_separate(self):
        row,_=row_and_configuration('her_values_shortened')
        row.update(wall_seconds_train=1,wall_seconds_eval=1,wall_seconds_total=2)
        full,values=update_counts(row)
        result=metrics(row)
        self.assertEqual(result['full_steps_per_environment_step'],full/row['actual_steps'])
        self.assertEqual(result['values_steps_per_environment_step'],values/row['actual_steps'])
        self.assertEqual(result['objective_scoring_objective_evaluations_per_draw'],2)
        self.assertEqual(result['objective_selection_objective_evaluations_per_draw'],0)
        self.assertEqual(result['route_original_draws'],row['replay_diagnostics']['original'])
        self.assertEqual(result['route_relabeled_draws'],row['replay_diagnostics']['relabeled'])
        baseline=dict(row,agent='acs2',online_updates=1000,replay_updates=0)
        baseline.pop('learning_update_counts')
        self.assertEqual(update_counts(baseline),(1000,0))

    def test_maze_analysis_reproduces_baseline_crossings_and_attaches_curves_to_every_values_comparison(self):
        plan=approved(Path(__file__).with_name('plan.json'))
        config=plan['configurations'][0]
        fixture=json.loads((Path(__file__).resolve().parents[2]/'fixtures/pair_baseline_success_counts.json').read_text())['configurations'][0]
        dataset={}
        names=plan['agents']+['her_values_p2of3','acs2','acs2er']
        template,_=row_and_configuration('her_values')
        template.update(wall_seconds_train=1,wall_seconds_eval=1,wall_seconds_total=2)
        for name in names:
            data=[]
            counts=fixture['agents']['acs2' if name=='acs2' else 'acs2er']
            for count in counts:
                row=copy.deepcopy(template)
                row.update(seed=count['seed'],nominal_step=count['nominal_step'],actual_steps=count['nominal_step'])
                row['evaluation_blocks']['test'].update(success=count['successes']/count['evaluation_pairs'],evaluation_pairs=count['evaluation_pairs'])
                data.append(row)
            dataset[name]=data
        result=analyze_maze(dataset,config,plan)
        self.assertEqual(len(result['comparisons']),10)
        values=[pair for pair in result['comparisons'] if 'model_updates_and_knowledge' in pair]
        self.assertEqual(len(values),5)
        for pair in values:
            for side in pair['model_updates_and_knowledge'].values():
                self.assertEqual(len(side['points']),8)
                self.assertIn('full_steps_per_environment_step',side)
                self.assertIn('knowledge',side)
        self.assertEqual(result['arms']['acs2']['threshold_crossings'],{'acs2_final':2000000,'acs2er_final':300000})


if __name__ == '__main__': unittest.main()
