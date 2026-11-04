import unittest
from plot_results import summarize_curves


class CurveTests(unittest.TestCase):
    def test_curve_intervals_keep_nonzero_seed_variance_points_and_undefined_slots(self):
        rows=[dict(post_hoc=True,view='pooled',maze='maze4',arm='acs2her_shortened_goal_rules',nominal_step=point,
                   seed=seed,F=(seed-41)*.1+point*.0001,F_mod=.4) for point in (10,20) for seed in range(42,47)]
        measured=summarize_curves(rows)
        f=[row for row in measured if row['metric']=='F']
        self.assertAlmostEqual(f[0]['mean'],.301)
        self.assertAlmostEqual(f[1]['mean'],.302)
        self.assertAlmostEqual(f[0]['se'],.1/2**.5)
        self.assertAlmostEqual(f[0]['ci_low'],.301-2.7764451051977987*.1/2**.5)
        rows[0]['F']=None
        first=next(row for row in summarize_curves(rows) if row['metric']=='F' and row['nominal_step']==10)
        self.assertEqual(first['n'],4)
        self.assertIsNone(first['mean'])
        with self.assertRaises(ValueError): summarize_curves(rows[:-1])
        with self.assertRaises(ValueError): summarize_curves(rows+[rows[0]])
