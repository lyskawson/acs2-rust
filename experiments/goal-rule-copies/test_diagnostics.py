import unittest
from diagnostics import value_dose


class ValueDoseTests(unittest.TestCase):
    def test_observed_opportunity_denominators_do_not_count_draws_or_include_goal_rules_as_blind(self):
        original=dict(draws=3,with_goal=4,without_goal=6,skipped_without_goal=0,draws_with_goal=2,action_set_rules=10)
        relabeled=dict(draws=5,with_goal=7,without_goal=0,skipped_without_goal=11,draws_with_goal=4,action_set_rules=18)
        row=dict(replay_diagnostics=dict(reward_updates=dict(original=original,relabeled=relabeled)))
        measured=value_dose(row)
        self.assertEqual(measured['relabeled_goal_action_set_share'],4/5)
        self.assertEqual(measured['skipped_reward_opportunity_share'],11/28)
        self.assertEqual(measured['retained_blind_opportunity_share'],6/17)
        for counts in row['replay_diagnostics']['reward_updates'].values():
            for key in counts: counts[key]=0
        self.assertTrue(all(value is None for value in value_dose(row).values()))
