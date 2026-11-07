import math
import unittest

from protocol import (ARMS, E, E1, S, S1, G, MAZES, SUCCESS_CONTRASTS, MECHANISM_CONTRASTS,
                      MECHANISM_METRICS, configurations)
from results import contrasts, decisions, interval


class ResultsTests(unittest.TestCase):
    def test_intervals_use_paired_seed_variance_and_the_registered_t_values(self):
        for n, t in [(5, 2.7764451051977987), (20, 2.093024054408263)]:
            values = {seed: (seed - 42) ** 2 + 3 for seed in range(42, 42 + n)}
            actual = interval(values, list(values))
            mean = sum(values.values()) / n
            se = math.sqrt(sum((value - mean) ** 2 for value in values.values()) / (n - 1) / n)
            self.assertGreater(se, 0)
            self.assertAlmostEqual(actual['mean'], mean)
            self.assertAlmostEqual(actual['se'], se)
            self.assertAlmostEqual(actual['ci_low'], mean - t * se)
            self.assertAlmostEqual(actual['ci_high'], mean + t * se)
            values[42] = None
            self.assertIsNone(interval(values, list(values))['ci_low'])
            self.assertEqual(interval(values, list(values))['n'], n - 1)
        with self.assertRaises(ValueError): interval({42: 1}, [42])
        with self.assertRaises(ValueError): interval({seed: float('nan') for seed in range(42, 47)}, list(range(42, 47)))

    def test_all_contrasts_separate_arm_seed_maze_and_metric_and_interaction_is_per_seed(self):
        success, mechanism = {}, {}
        for m, config in enumerate(configurations()):
            for a, arm in enumerate(ARMS):
                for seed in range(42, 62):
                    s = seed - 41
                    success[config['task'], arm, seed] = dict(primary=(a + 1) ** 2 * (s + m) + a * s * s,
                        final=(a + 2) ** 3 * (s + 2 * m) - a * s * s)
                    if seed < 47:
                        mechanism[config['task'], arm, seed] = {metric: (a + 1) ** 2 * (k + 3) * (s + 3 * m) + a * s * s
                            for k, metric in enumerate(MECHANISM_METRICS)}
        rows = contrasts(success, mechanism, configurations())
        self.assertEqual(len(rows), 120)
        for row in rows:
            source = success if row['group'] == 'success' else mechanism
            seeds = range(42, 62) if row['group'] == 'success' else range(42, 47)
            expected = []
            for seed in seeds:
                value = lambda arm: source[row['maze'], arm, seed][row['metric']]
                expected.append({'S+1 - S': lambda: value(S1) - value(S),
                    'E+1 - E': lambda: value(E1) - value(E),
                    'S+1 - G': lambda: value(S1) - value(G),
                    'interaction': lambda: (value(S1) - value(S)) - (value(E1) - value(E))}[row['comparison']]())
            self.assertEqual(row['seed_differences'], expected)
            self.assertGreater(row['se'], 0)
        interaction = next(row for row in rows if row['comparison'] == 'interaction' and row['metric'] == 'primary')
        differences = interaction['seed_differences']
        se = math.sqrt(sum((value - sum(differences) / 20) ** 2 for value in differences) / 19 / 20)
        self.assertAlmostEqual(interaction['se'], se)
        success['maze4', S1, 42]['primary'] += 1234
        changed = contrasts(success, mechanism, configurations())
        affected = [row for before, row in zip(rows, changed) if before != row]
        self.assertEqual({row['comparison'] for row in affected}, {'S+1 - S', 'S+1 - G', 'interaction'})
        self.assertTrue(all(row['maze'] == 'maze4' and row['metric'] == 'primary' for row in affected))

    def fixture(self, primary, final, modal=(1, 1, 1, 0), slope=(1, 1, 1, 0)):
        rows = []
        for comparison in SUCCESS_CONTRASTS:
            for metric, signs in [('primary', primary), ('final', final)]:
                for maze, sign in zip(MAZES, signs):
                    rows.append(dict(maze=maze, comparison=comparison, group='success', metric=metric,
                                     ci_low=sign - .1, ci_high=sign + .1))
        for comparison in MECHANISM_CONTRASTS:
            for m, maze in enumerate(MAZES):
                for metric, signs in [('F_mod', modal), ('within_start_slope', slope)]:
                    sign = signs[m] if m < 4 else 1
                    rows.append(dict(maze=maze, comparison=comparison, group='mechanism', metric=metric,
                                     ci_low=sign - .1, ci_high=sign + .1))
        return rows

    def test_disjoint_verdicts_cover_all_six_mazes_and_keep_primary_and_final_separate(self):
        for signs, expected in [([1, 1, 1, 0, 0, 0], 'improves'),
                ([-1, -1, -1, 0, 0, 0], 'worsens'), ([1, 1, 1, -1, 0, 0], 'mixed'),
                ([1, 1, 0, 0, 0, 0], 'no clear effect'), ([1, 0, 0, 0, 1, 1], 'improves'),
                ([0] * 6, 'no clear effect')]:
            result = decisions(self.fixture(signs, [-sign for sign in signs]))
            for row in result['verdicts']:
                wanted = expected if row['metric'] == 'primary' else {'improves': 'worsens', 'worsens': 'improves'}.get(expected, expected)
                if row['comparison'] == 'interaction':
                    wanted = {'improves': 'copies gain more from per-goal values',
                        'worsens': 'copies gain less from per-goal values', 'no clear effect': 'no clear difference'}.get(wanted, wanted)
                self.assertEqual(row['verdict'], wanted)
            self.assertTrue(all(row['confirmed'] for row in result['mechanism']))
        rows = self.fixture([1, 1, 1, 0, 0, 0], [0] * 6)
        rows[0]['ci_low'] = 0
        result = decisions(rows)
        self.assertEqual(result['verdicts'][0]['A_positive'], 2)
        self.assertEqual(result['verdicts'][0]['verdict'], 'no clear effect')
        with self.assertRaises(ValueError): decisions(rows + [rows[0]])

    def test_mechanism_requires_both_positive_intervals_on_the_same_three_core_mazes(self):
        result = decisions(self.fixture([0] * 6, [0] * 6, slope=(0, 1, 1, 1)))
        self.assertFalse(any(row['confirmed'] for row in result['mechanism']))
        rows = self.fixture([0] * 6, [0] * 6)
        for row in rows:
            if row['group'] == 'mechanism' and row['comparison'] == 'E+1 - E' and row['maze'] == 'maze4':
                row['ci_low'] = 0
        result = decisions(rows)
        self.assertTrue(result['mechanism'][0]['confirmed'])
        self.assertFalse(result['mechanism'][1]['confirmed'])
        self.assertEqual(result['mechanism'][1]['mazes'], ['maze5', 'maze6'])
