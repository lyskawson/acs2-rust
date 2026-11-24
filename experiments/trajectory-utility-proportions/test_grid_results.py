from collections import defaultdict
from fractions import Fraction
import unittest

from grid_results import NEW_ARMS, REFERENCE_ARMS, speed_points, statistics
from proportions import MEAN, RANDOM, S1, label


class RegisteredResultTests(unittest.TestCase):
    def test_final_benchmark_and_sustained_threshold_are_separate(self):
        self.assertEqual(speed_points([1000, 2000, 4000, 6000], [.87, .97, .92, .98], .96),
                         dict(first_095=2000, sustained_095=6000, reaches_s1_final=2000))

    def test_extended_uses_its_floor_and_only_the_six_measured_arms(self):
        task = 'aliased_maze'
        seeds = list(range(42, 62))
        configs = {task: dict(cap=5, group='aliasing', targets=[1000, 3000, 10000])}
        study = dict(seeds=seeds, analysis=dict(contrasts=[(MEAN, RANDOM), (MEAN, S1), (RANDOM, S1)]))
        reference = {task: dict(registered_floor=Fraction(1, 10), extended_floor=Fraction(3, 4), shortest=3.0)}
        observations = defaultdict(list)
        cost = []
        for index, arm in enumerate([S1, *REFERENCE_ARMS[1:], *NEW_ARMS]):
            for seed in seeds:
                fluctuation = (seed-42) * (index+1) / 10000
                cost.append(dict(task=task, arm=arm, seed=seed, process_cpu_seconds=(seed-40)*(index+1),
                                 utility_cpu_seconds=(seed-40)*index/2, peak_rss_kib=100+seed))
                for block in ('registered', 'extended') if arm in NEW_ARMS else ('registered',):
                    for step, base in zip(configs[task]['targets'], (.2, .3, .4)):
                        rows = observations[task,arm,block]
                        rows.append(dict(seed=seed, nominal_step=step, success=base+fluctuation,
                                         mean_steps=3+index/10, mean_steps_over_shortest=(3+index/10)/3,
                                         requested_tu_share=.5, requested_hindsight_share=.25,
                                         realized_tu_share=.45, realized_hindsight_share=.20))
        result = statistics(study,configs,{task:'Aliased'},reference,{task:True},observations,cost)
        labels = {row['test_cap']: row for row in result['labels']}
        self.assertFalse(labels['registered']['floor'])
        self.assertTrue(labels['extended']['floor'])
        self.assertTrue(labels['extended']['too_easy'])
        extended = [row for row in result['comparisons'] if row['test_cap']=='extended']
        self.assertEqual(len(extended), 6)
        self.assertTrue(all(row['reference']!=S1 and row['n']==20 and not row['counted'] for row in extended))
        measured = next(row for row in result['maze-costs'] if row['arm']==label(MEAN,.25))
        self.assertEqual(measured['median_process_to_s1'],4)
        self.assertEqual(measured['median_utility_to_s1'],1.5)
        self.assertEqual(measured['peak_rss_kib'],161)
