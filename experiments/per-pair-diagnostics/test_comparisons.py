import unittest

from comparisons import registered_results
from protocol import ARMS


class ComparisonTests(unittest.TestCase):
    def test_registered_views_use_exact_points_paired_differences_and_common_all_pairs(self):
        metrics = {}
        for index, arm in enumerate(ARMS):
            for seed in range(42, 47):
                for point in (30000, 100000, 2000000):
                    metrics['maze4', arm, seed, point, 'pooled'] = dict(F=.5 + index * .05 + point * 1e-9,
                        C=.4, M=.6 + index * .01, D=.3 + index * .02,
                        values=dict(all=dict(estimate_minus_q=index * 10 + seed, winner_goal_share=.2 + index * .1, winner_goal_mass=.1 + index * .05),
                                    failed=dict(estimate_minus_q=-9999)))
        rows = registered_results(metrics, [dict(task='maze4', targets=[30000, 100000, 2000000])])
        self.assertEqual(len(rows), 38)
        policy = next(row for row in rows if row['comparison'] == 'acs2her - acs2er_episodic' and row['metric'] == 'F')
        self.assertAlmostEqual(policy['mean'], .05)
        self.assertLess(policy['se'], 1e-14)
        self.assertEqual(policy['nominal_step'], 2000000)
        first = next(row for row in rows if row['comparison'] == 'acs2: F - C' and row['nominal_step'] == 30000)
        self.assertAlmostEqual(first['mean'], .10003)
        value = next(row for row in rows if row['comparison'] == 'acs2her - acs2er_episodic' and row['metric'] == 'estimate_minus_Q*')
        self.assertEqual(value['mean'], 10)
        winner = next(row for row in rows if row['comparison'] == 'acs2' and row['metric'] == 'winner_goal_mass')
        self.assertEqual(winner['mean'], .1)
        self.assertTrue(all(row['view'] == 'pooled' and row['registered'] for row in rows))


if __name__ == '__main__':
    unittest.main()
