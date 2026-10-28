from fractions import Fraction
import math
import unittest

from validate_tables import CRITICAL, select_exact, validate_difference


class TableTests(unittest.TestCase):
    def record(self):
        mean = Fraction(19, 80)
        se = math.sqrt(7) / 80
        return dict(n=20, exact_mean=str(mean), mean=float(mean), se=se,
                    ci_low=float(mean) - CRITICAL * se, ci_high=float(mean) + CRITICAL * se)

    def scores(self):
        return ({seed: Fraction(seed - 42, 20) for seed in range(42, 62)},
                {seed: Fraction(seed - 42, 40) for seed in range(42, 62)})

    def test_independent_fraction_reconstruction_accepts_the_paired_interval(self):
        left, right = self.scores()
        self.assertEqual(validate_difference(self.record(), left, right), Fraction(19, 80))

    def test_unpaired_errors_wrong_seeds_and_wrong_critical_value_are_rejected(self):
        left, right = self.scores()
        altered = self.record()
        altered['se'] *= math.sqrt(5)
        with self.assertRaises(ValueError):
            validate_difference(altered, left, right)
        rotated = {seed: right[42 + (seed - 41) % 20] for seed in range(42, 62)}
        with self.assertRaises(ValueError):
            validate_difference(self.record(), left, rotated)
        altered = self.record()
        altered['ci_low'] = altered['mean'] - 1.96 * altered['se']
        with self.assertRaises(ValueError):
            validate_difference(altered, left, right)

    def test_exact_validation_selection_uses_the_registered_small_p_tie_rule(self):
        values = {1 / 3: [Fraction(3, 5)] * 5, 0.5: [Fraction(3, 5) + Fraction(1, 10**13)] * 5,
                  2 / 3: [Fraction(1, 2)] * 5, 0.8: [Fraction(2, 5)] * 5}
        self.assertEqual(select_exact(values), 1 / 3)
        values[0.5] = [Fraction(3, 5) + Fraction(1, 10**10)] * 5
        self.assertEqual(select_exact(values), 0.5)


if __name__ == '__main__':
    unittest.main()
