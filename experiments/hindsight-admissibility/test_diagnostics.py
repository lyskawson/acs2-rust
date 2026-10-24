import unittest

from diagnostics import value_groups


class DiagnosticTests(unittest.TestCase):
    def block(self):
        return dict(evaluation_pairs=4, success=0.25, evaluation_distribution='exhaustive_exact_weights', value_diagnostics=dict(
            all=dict(mean_first_action_estimate=400.0, mean_discounted_return=250.0, mean_estimate_minus_return=150.0),
            successful=dict(mean_first_action_estimate=1000.0, mean_discounted_return=1000.0, mean_estimate_minus_return=0.0)))

    def test_failed_pairs_recover_the_independent_three_pair_mean(self):
        result = value_groups(self.block())
        self.assertEqual(result['failed_pairs'], 3)
        self.assertEqual(result['failed']['mean_first_action_estimate'], 200)
        self.assertEqual(result['failed']['mean_discounted_return'], 0)
        self.assertEqual(result['failed']['mean_estimate_minus_return'], 200)

    def test_empty_conditional_groups_stay_absent(self):
        block = self.block()
        block['success'] = 1 - 1e-15
        self.assertIsNone(value_groups(block)['failed'])
        block['success'] = 0
        block['value_diagnostics']['successful'] = None
        result = value_groups(block)
        self.assertIsNone(result['successful'])
        self.assertEqual(result['failed'], block['value_diagnostics']['all'])

    def test_inexact_success_counts_are_refused(self):
        block = self.block()
        block['success'] = 0.3
        with self.assertRaises(ValueError):
            value_groups(block)
        block = self.block()
        block['evaluation_distribution'] = 'sampled'
        with self.assertRaises(ValueError):
            value_groups(block)


if __name__ == '__main__':
    unittest.main()
