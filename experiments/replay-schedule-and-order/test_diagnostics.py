import unittest

from diagnostics import intervals, value_gaps


class DiagnosticTests(unittest.TestCase):
    def test_update_rates_are_checkpoint_interval_differences_including_the_first_interval(self):
        rows = [dict(nominal_step=10, actual_steps=12, replay_updates=36),
                dict(nominal_step=20, actual_steps=22, replay_updates=46),
                dict(nominal_step=30, actual_steps=33, replay_updates=57)]
        result = intervals(rows)
        self.assertEqual([cell['updates_per_interaction_exact'] for cell in result], ['3', '1', '1'])
        self.assertEqual([cell['interval_interactions'] for cell in result], [12, 10, 11])
        self.assertEqual([cell['interval_updates'] for cell in result], [36, 10, 11])

    def test_value_gap_failure_group_uses_its_own_denominator_and_empty_groups_are_null(self):
        block = dict(success=0.25, evaluation_pairs=100,
            value_diagnostics=dict(all=dict(mean_estimate_minus_return=100), successful=dict(mean_estimate_minus_return=40)))
        self.assertEqual(value_gaps(block), dict(all=100, successful=40, failed=120))
        block['success'] = 1.0
        self.assertIsNone(value_gaps(block)['failed'])
        block['success'] = 0.0
        self.assertIsNone(value_gaps(block)['successful'])
        self.assertEqual(value_gaps(block)['failed'], 100)


if __name__ == '__main__':
    unittest.main()
