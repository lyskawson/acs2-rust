import copy
import unittest
from goal_rule_schema import ARM, validate_reward_updates


def sample():
    def route(draws, goal, blind, skipped, hits):
        return dict(draws=draws, with_goal=goal, without_goal=blind, skipped_without_goal=skipped,
                    draws_with_goal=hits, action_set_rules=goal + blind + skipped)
    return dict(replay_diagnostics=dict(original=3, relabeled=5, reward_updates=dict(
        original=route(3, 4, 6, 0, 2), relabeled=route(5, 7, 0, 11, 4))))


class GoalRuleCountersTests(unittest.TestCase):
    def test_macroclassifier_updates_and_draws_reconcile_separately(self):
        validate_reward_updates(sample(), ARM)
        for origin, field, value in [('original', 'skipped_without_goal', 1), ('relabeled', 'without_goal', 1),
                ('original', 'draws', 4), ('relabeled', 'draws_with_goal', 6),
                ('relabeled', 'action_set_rules', 19), ('relabeled', 'with_goal', 0),
                ('original', 'without_goal', True), ('relabeled', 'draws_with_goal', 0)]:
            row = sample()
            row['replay_diagnostics']['reward_updates'][origin][field] = value
            with self.subTest(origin=origin, field=field), self.assertRaises(ValueError):
                validate_reward_updates(row, ARM)

    def test_only_the_exact_arm_accepts_the_counter_schema(self):
        with self.assertRaises(ValueError):
            validate_reward_updates(sample(), ARM + '_other')
        row = sample()
        del row['replay_diagnostics']['reward_updates']
        with self.assertRaises(ValueError):
            validate_reward_updates(row, ARM)
        validate_reward_updates(row, 'acs2her_shortened')
        row = sample()
        row['replay_diagnostics']['reward_updates']['relabeled']['extra'] = 0
        with self.assertRaises(ValueError):
            validate_reward_updates(row, ARM)
