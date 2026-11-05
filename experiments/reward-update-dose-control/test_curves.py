import unittest
from plot_results import summarize_curves
from protocol import ARM, G, S, E


class CurveTests(unittest.TestCase):
    def test_series_keep_random_goal_and_shortened_arms_and_exclude_episodic_reference(self):
        rows=[dict(post_hoc=True,view='pooled',maze='maze4',arm=arm,nominal_step=100,
                   seed=seed,F=index*.1+(seed-42)*.01,F_mod=index*.2+(seed-42)*.02)
              for index,arm in enumerate((ARM,G,S,E)) for seed in range(42,47)]
        measured=summarize_curves(rows)
        self.assertEqual({row['arm'] for row in measured},{ARM,G,S})
        self.assertEqual(len(measured),6)
        for index,arm in enumerate((ARM,G,S)):
            row=next(row for row in measured if row['arm']==arm and row['metric']=='F')
            self.assertAlmostEqual(row['mean'],index*.1+.02)
            self.assertGreater(row['se'],0)

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
