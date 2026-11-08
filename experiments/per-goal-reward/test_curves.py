import math
import unittest

from protocol import configurations, T4
from plot_results import ARMS, summarize


class CurveTests(unittest.TestCase):
    def fixture(self):
        return [dict(maze=config['task'], arm=arm, seed=seed, nominal_step=point, view='pooled', post_hoc=True,
            F=.01 * (a + 1) + .003 * m + .0001 * p + .002 * (seed - 42),
            F_mod=.02 * (a + 1) + .004 * m + .0002 * p + .003 * (seed - 42))
            for m, config in enumerate(configurations()) for a, arm in enumerate(ARMS)
            for p, point in enumerate(config['targets']) for seed in range(42, 47)]

    def test_all_curves_keep_distinct_arms_points_mazes_and_nonzero_seed_variance(self):
        rows = summarize(self.fixture())
        self.assertEqual(len(rows), 392)
        for row in rows:
            m = [config['task'] for config in configurations()].index(row['maze'])
            a = ARMS.index(row['arm'])
            p = configurations()[m]['targets'].index(row['nominal_step'])
            if row['metric'] == 'F': mean, step = .01 * (a + 1) + .003 * m + .0001 * p + .004, .002
            else: mean, step = .02 * (a + 1) + .004 * m + .0002 * p + .006, .003
            se = step / math.sqrt(2)
            self.assertAlmostEqual(row['mean'], mean)
            self.assertAlmostEqual(row['se'], se)
            self.assertAlmostEqual(row['ci_low'], mean - T4 * se)
            self.assertAlmostEqual(row['ci_high'], mean + T4 * se)
            self.assertTrue(row['post_hoc'])

    def test_incomplete_duplicate_or_unlabelled_curves_are_refused_without_dropping_null_seeds(self):
        rows = self.fixture()
        for values in (rows[:-1], rows + [rows[0]], rows[5:]):
            with self.assertRaises(ValueError): summarize(values)
        rows[0]['post_hoc'] = False
        with self.assertRaises(ValueError): summarize(rows)
        rows = self.fixture()
        rows[0]['F'] = None
        values = summarize(rows)
        row = next(value for value in values if (value['maze'], value['arm'], value['nominal_step'], value['metric'])
                   == (rows[0]['maze'], rows[0]['arm'], rows[0]['nominal_step'], 'F'))
        self.assertIsNone(row['mean'])
        self.assertEqual(row['n'], 4)
