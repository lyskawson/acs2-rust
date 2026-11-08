import copy
import unittest

from verify_grid import validate_aggregates


class VerificationTests(unittest.TestCase):
    def test_weighted_pairs_bind_success_both_values_and_the_environment_model(self):
        truth = {((0, 0), (0, goal)): dict(cap=3, shortest_distance=goal,
            value=1000 * .95 ** (goal - 1), action_values=[1000 * .95 ** (goal - 1)] + [0.] * 7,
            optimal_actions=[0]) for goal in (1, 2, 3)}
        block = dict(success=.75, value_diagnostics=dict(all=dict(mean_first_action_estimate=42.5,
            mean_discounted_return=620.0)), starts=[dict(start='(0, 0)',
            goal=f'Goal {{ symbols: [Token(0), Token({goal})] }}', weight=weight, success=goal != 2,
            estimated_first_action_value=estimate, discounted_return=returned, shortest_distance=goal,
            first_action=0, first_decision=dict(random_fallback=False, winners=[]))
            for goal, weight, estimate, returned in [(1, .125, 10., 1000.), (2, .25, 30., 0.), (3, .625, 54., 792.)]])
        validate_aggregates(block, truth)
        for field in ('success', 'estimated_first_action_value', 'discounted_return', 'shortest_distance'):
            changed = copy.deepcopy(block)
            changed['starts'][1][field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError): validate_aggregates(changed, truth)
        changed = copy.deepcopy(truth)
        changed[((0, 0), (0, 2))]['value'] += 1
        with self.assertRaises(ValueError): validate_aggregates(block, changed)
