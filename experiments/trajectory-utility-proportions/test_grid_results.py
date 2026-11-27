from collections import defaultdict
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from grid_results import NEW_ARMS, REFERENCE_ARMS, speed_points, statistics, test_references as read_test_references, verdict
from proportions import MEAN, RANDOM, S1, label


class RegisteredResultTests(unittest.TestCase):
    def test_final_benchmark_and_sustained_threshold_are_separate(self):
        self.assertEqual(speed_points([1000, 2000, 4000, 6000], [.87, .97, .92, .98], .96),
                         dict(first_095=2000, sustained_095=6000, reaches_s1_final=2000))

    def maze_results(self, curves=None, easy=False):
        curves = curves or {}
        task = 'aliased_maze'
        seeds = list(range(42, 62))
        configs = {task: dict(cap=5, group='aliasing', targets=[1000, 3000, 10000])}
        study = dict(seeds=seeds, analysis=dict(contrasts=[(MEAN, RANDOM), (MEAN, S1), (RANDOM, S1)]))
        reference = {task: dict(registered_floor=Fraction(1, 10), extended_floor=Fraction(3, 4), shortest=3.0)}
        observations = defaultdict(list)
        cost = []
        for index, arm in enumerate([S1, *REFERENCE_ARMS[1:], *NEW_ARMS]):
            for seed in seeds:
                fluctuation = (seed-51) * (index+1) / 40000
                cost.append(dict(task=task, arm=arm, seed=seed, process_cpu_seconds=(seed-40)*(index+1),
                                 utility_cpu_seconds=(seed-40)*index/2, peak_rss_kib=100+seed))
                for block in ('registered', 'extended') if arm in NEW_ARMS else ('registered',):
                    for step, base in zip(configs[task]['targets'], curves.get(arm, (.2, .3, .4))):
                        rows = observations[task,arm,block]
                        rows.append(dict(seed=seed, nominal_step=step, success=base+fluctuation,
                                         mean_steps=3+index/10, mean_steps_over_shortest=(3+index/10)/3,
                                         requested_tu_share=.5, requested_hindsight_share=.25,
                                         realized_tu_share=.45, realized_hindsight_share=.20))
        return statistics(study, configs, {task: 'Aliased'}, reference, {task: easy}, observations, cost)

    def matched_contrast(self, results, measure):
        return next(row for row in results['comparisons']
                    if (row['test_cap'], row['measure'], row['favorable'], row['reference'])
                    == ('registered', measure, label(MEAN, .25), label(RANDOM, .25)))

    def test_extended_uses_its_floor_and_only_the_six_measured_arms(self):
        result = self.maze_results(easy=True)
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

    def test_odd_group_requires_ceiling_of_half(self):
        self.assertEqual(verdict(1, 0, 3), 'no clear effect')
        self.assertEqual(verdict(0, 1, 3), 'no clear effect')
        self.assertEqual(verdict(2, 0, 3), 'improves')
        self.assertEqual(verdict(0, 2, 3), 'worsens')

    def test_floor_requires_every_compared_arm_at_the_floor(self):
        curves = {arm: (.03, .05, .07) for arm in (*NEW_ARMS, S1)}
        for above_floor in (False, True):
            with self.subTest(above_floor=above_floor):
                if above_floor:
                    curves[label(MEAN, .25)] = (.1, .2, .3)
                result = self.maze_results(curves)
                registered = next(row for row in result['labels'] if row['test_cap'] == 'registered')
                self.assertEqual(registered['floor'], not above_floor)
                self.assertEqual(self.matched_contrast(result, 'final')['counted'], above_floor)

    def test_ceiling_requires_both_arms_of_the_contrast(self):
        mean, random = label(MEAN, .25), label(RANDOM, .25)
        for random_final in (.85, .996):
            with self.subTest(random_final=random_final):
                result = self.maze_results({mean: (.4, .7, .995), random: (.35, .6, random_final)})
                row = self.matched_contrast(result, 'final')
                self.assertEqual(row['ceiling'], random_final == .996)
                self.assertEqual(row['counted'], random_final == .85)

    def test_too_easy_excludes_only_the_primary_measure(self):
        result = self.maze_results(easy=True)
        primary = self.matched_contrast(result, 'primary')
        final = self.matched_contrast(result, 'final')
        self.assertTrue(primary['too_easy'] and final['too_easy'])
        self.assertFalse(primary['counted'])
        self.assertTrue(final['counted'])

    def test_test_references_calculates_the_exact_100_step_floor(self):
        goal, start = [1, 0], [0, 0]
        geometry = [dict(task='two_cells', walkable_cells=[start, goal], distances=[[0, 1], [1, 0]],
                         successors=[[0]*4 + [1]*4, [0]*4 + [1]*4])]
        test = dict(goal_starts=[dict(goal=goal, starts=[start])], evaluation_pairs=1,
                    pair_weight='1', random_floor_exact='3/4')
        configs = {'two_cells': dict(cap=2, registration={'blocks': {'test': test}})}
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'geometry.json').write_text(json.dumps(geometry))
            with patch('grid_results.ABLATION', root):
                reference = read_test_references(configs)['two_cells']
        self.assertEqual(reference['registered_floor'], Fraction(3, 4))
        self.assertEqual(reference['extended_floor'], 1 - Fraction(1, 2**100))
        self.assertEqual((reference['shortest'], reference['pairs']), (1, 1))
