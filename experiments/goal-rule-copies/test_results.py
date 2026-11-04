import math
import unittest
from protocol import ARM, ARMS, REFERENCES, configurations
from results import paired, contrasts, decisions


class RegisteredResultsTests(unittest.TestCase):
    def test_pairing_before_sample_variance_for_both_seed_counts(self):
        for n, critical in [(5, 2.7764451051977987), (20, 2.093024054408263)]:
            seeds = list(range(42, 42+n))
            left = {seed: 10000 + seed*100 + index for index, seed in enumerate(seeds)}
            right = {seed: 10000 + seed*100 for seed in seeds}
            measured = paired(left, right, seeds)
            se = math.sqrt(n*(n+1)/12) / math.sqrt(n)
            self.assertAlmostEqual(measured['mean'], (n-1)/2)
            self.assertAlmostEqual(measured['se'], se)
            self.assertAlmostEqual(measured['ci_low'], (n-1)/2-critical*se)
            right[42] = None
            self.assertIsNone(paired(left, right, seeds)['mean'])
            with self.assertRaises(ValueError):
                paired(left, {42: 0}, seeds)

    def test_contrast_driver_keeps_references_metrics_seed_sets_and_final_point(self):
        success, mechanisms = {}, {}
        for config in configurations():
            for index, arm in enumerate(ARMS):
                for seed in range(42, 62):
                    success[config['task'], arm, seed] = dict(primary=index*seed, final=index*seed*2)
                    if seed < 47:
                        mechanisms[config['task'], arm, seed] = {metric: index*seed*(k+1) for k, metric in enumerate(
                            ['F', 'F_mod', 'estimate_minus_q', 'estimate_above_v', 'within_start_slope'])}
        rows = contrasts(success, mechanisms, configurations())
        self.assertEqual(len(rows), 84)
        for row in rows:
            index = ARMS.index(row['reference'])
            seeds = range(42, 62) if row['group'] == 'success' else range(42, 47)
            multiplier = {'primary': 1, 'final': 2, 'F': 1, 'F_mod': 2, 'estimate_minus_q': 3,
                          'estimate_above_v': 4, 'within_start_slope': 5}[row['metric']]
            self.assertEqual(row['seed_differences'], [-index*seed*multiplier for seed in seeds])
            self.assertGreater(row['se'], 0)

    def test_outcome_counts_mazes_and_requires_joint_mechanism_on_same_maze(self):
        mazes = [config['task'] for config in configurations()]
        def rows(primary, final, modal=(1,1,1,0), over=(-1,-1,-1,0)):
            result = []
            for i, maze in enumerate(mazes):
                for group, metric, sign in [('success','primary',primary[i]), ('success','final',final[i]),
                        ('mechanism','F_mod', modal[i] if i<4 else 0), ('mechanism','estimate_above_v',over[i] if i<4 else 0)]:
                    result.append(dict(maze=maze, reference='acs2her_shortened', group=group, metric=metric,
                                       ci_low=sign if sign else 0, ci_high=sign if sign else 0))
            return result
        for primary, final, expected in [([1,1,1,0,0,0], [0]*6, 'helps'),
                ([-1,0,0,0,0,0], [1,1,1,0,0,0], 'trade-off'),
                ([1,1,1,-1,0,0], [-1,0,0,0,0,0], 'trade-off'),
                ([1,1,0,0,0,0], [1]*6, 'does not help'),
                ([0]*6, [0]*6, 'does not help')]:
            measured = decisions(rows(primary, final), mazes)
            self.assertEqual(measured['outcome'], expected)
            self.assertTrue(measured['mechanism_confirmed'])
        self.assertFalse(decisions(rows([0]*6, [0]*6, over=(0,-1,-1,-1)), mazes)['mechanism_confirmed'])

    def test_trade_off_precedes_helps_when_primary_improves_and_final_worsens(self):
        mazes = [config['task'] for config in configurations()]
        rows = []
        for index, maze in enumerate(mazes):
            intervals = {
                ('success', 'primary'): (.1, .2) if index < 3 else (-.1, .1),
                ('success', 'final'): (-.2, -.1) if index == 0 else (-.1, .1),
                ('mechanism', 'F_mod'): (-.1, .1),
                ('mechanism', 'estimate_above_v'): (-.1, .1),
            }
            for (group, metric), (low, high) in intervals.items():
                rows.append(dict(maze=maze, reference='acs2her_shortened', group=group,
                                 metric=metric, ci_low=low, ci_high=high))
        measured = decisions(rows, mazes)
        self.assertEqual(measured['positive']['primary'], 3)
        self.assertEqual(measured['negative']['primary'], 0)
        self.assertEqual(measured['negative']['final'], 1)
        self.assertEqual(measured['outcome'], 'trade-off')
