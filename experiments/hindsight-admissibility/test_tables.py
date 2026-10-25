import copy
from fractions import Fraction
import unittest

from analysis import paired
from validate_tables import first_crossing, verify_allocation_cost, verify_summary


class TableTests(unittest.TestCase):
    def test_independent_pair_variance_refuses_an_unpaired_interval(self):
        left = {seed: Fraction(seed - 42, 100) for seed in range(42, 62)}
        right = {seed: left[seed] - Fraction(1, 10) for seed in range(42, 62)}
        values = {seed: left[seed] - right[seed] for seed in left}
        result = paired(left, right)
        verify_summary(values, result)
        changed = copy.deepcopy(result)
        changed.update(se=0.02, ci_low=0.05813951891183474, ci_high=0.14186048108816526)
        with self.assertRaises(ValueError):
            verify_summary(values, changed)

    def test_independent_crossing_counts_exact_equality(self):
        curve = {100: Fraction(1, 10), 200: Fraction(1, 5)}
        self.assertEqual(first_crossing(curve, [100, 200], '1/10'), 100)
        self.assertIsNone(first_crossing(curve, [100, 200], '1/4'))

    def test_historical_reference_cpu_is_kept_outside_the_new_budget(self):
        costs = [dict(reused_reference='False', allocation_cpu_seconds='100'), dict(reused_reference='True', allocation_cpu_seconds='9000')]
        execution = dict(grid_allocated_cpu_seconds=100, pilot_allocated_cpu_seconds=1723, allocated_cpu_seconds=1823)
        self.assertEqual(verify_allocation_cost(costs, execution), 9000)
        with self.assertRaises(ValueError):
            verify_allocation_cost(costs, dict(execution, allocated_cpu_seconds=10823))


if __name__ == '__main__':
    unittest.main()
