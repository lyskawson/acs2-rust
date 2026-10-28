from fractions import Fraction
import math
import unittest

from analysis import contrasts, crossing, paired, seed_score


class AnalysisTests(unittest.TestCase):
    def scores(self):
        return {name: {seed: Fraction(base + (seed - 42) ** 2 * slope, 10000) for seed in range(42, 62)}
                for name, base, slope in [('S@selected', 5000, 1), ('S@2/3', 4000, 2),
                    ('H@1/2', 3000, 3), ('H@2/3', 2000, 4), ('E', 1000, 5)]}

    def test_rule_and_dose_contrasts_use_distinct_arms_and_paired_errors(self):
        scores = self.scores()
        results = contrasts(scores, 0.5)
        rule = results['S@2/3 - H@1/2']
        dose = results['H@1/2 - H@2/3']
        self.assertEqual(rule, paired(scores['S@2/3'], scores['H@1/2']))
        self.assertEqual(dose, paired(scores['H@1/2'], scores['H@2/3']))
        self.assertNotEqual(rule['mean'], paired(scores['S@2/3'], scores['H@2/3'])['mean'])
        differences = [scores['S@2/3'][seed] - scores['H@1/2'][seed] for seed in range(42, 62)]
        exact_mean = sum(differences, Fraction()) / 20
        expected_se = math.sqrt(float(sum((value - exact_mean) ** 2 for value in differences) / (19 * 20)))
        self.assertAlmostEqual(rule['se'], expected_se, places=15)
        self.assertEqual(Fraction(rule['exact_mean']) + Fraction(dose['exact_mean']),
                         Fraction(paired(scores['S@2/3'], scores['H@2/3'])['exact_mean']))
        self.assertNotIn('S@selected - S@2/3', contrasts(scores, 2 / 3))
        bad = self.scores()
        bad['E'].pop(61)
        with self.assertRaises(ValueError):
            contrasts(bad, 0.5)

    def test_success_uses_test_block_and_both_views_with_exact_rational_threshold(self):
        rows = [dict(nominal_step=point, evaluation_blocks={
            'validation': dict(evaluation_pairs=100, success=0.9),
            'test': dict(evaluation_pairs=10, success=value)}) for point, value in [(1000, 0.2), (3000, 0.6)]]
        self.assertEqual(seed_score(rows, [1000, 3000], 'checkpoint_mean'), Fraction(2, 5))
        self.assertEqual(seed_score(rows, [1000, 3000], 'final_point'), Fraction(3, 5))
        paired_rows = {seed: rows for seed in range(42, 62)}
        self.assertEqual(crossing(paired_rows, [1000, 3000], '3/5'), 3000)
        self.assertIsNone(crossing(paired_rows, [1000, 3000], '60000000000000001/100000000000000000'))


if __name__ == '__main__':
    unittest.main()
