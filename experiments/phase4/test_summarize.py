import copy
import unittest

from summarize import METRICS, RULES, STRATEGIES, mean_and_se, summarize


def rows():
    result = []
    for seed in (42, 43):
        for strategy in STRATEGIES:
            for rule in RULES:
                for filtered in (False, True):
                    expected = dict.fromkeys(METRICS, 0.0)
                    expected.update(admissible_share=1.0, relabeled=1.0, mean_reward_evaluations=1.0,
                                    mean_reach_evaluations=2.0, mean_objective_evaluations=3.0,
                                    mean_reward=float(seed))
                    result.append(dict(schema=1, seed=seed, configuration='test', strategy=strategy,
                                       rule=rule, candidate_filter=filtered, actual_steps=20000,
                                       target_steps=20000, cap=5, episodes=4000, goal_pool='full',
                                       candidate_count_at_end=4, commit='test', source_state='clean',
                                       composition_uses_rng=False, relabeled_proportion=1.0, expected=expected))
    return result


class CompositionSummaryTests(unittest.TestCase):
    def test_seed_standard_error_uses_the_sample_variance(self):
        self.assertEqual(mean_and_se([1.0, 3.0]), (2.0, 1.0))
        self.assertEqual(mean_and_se([2.0, 2.0, 2.0]), (2.0, 0.0))

    def test_summary_weights_independent_seeds_equally(self):
        summary = summarize(rows(), (42, 43), ('test',))
        self.assertEqual(len(summary), 24)
        self.assertEqual(summary[0]['mean_reward_mean'], 42.5)
        self.assertEqual(summary[0]['mean_reward_se'], 0.5)

    def test_missing_and_duplicate_seeds_are_rejected(self):
        data = rows()
        for changed in (data[:-1], data + data[:1]):
            with self.assertRaises(ValueError):
                summarize(changed, (42, 43), ('test',))

    def test_variants_must_share_the_same_raw_trajectory(self):
        data = rows()
        data[0]['episodes'] += 1
        with self.assertRaises(ValueError):
            summarize(data, (42, 43), ('test',))

    def test_invalid_fallback_or_filter_composition_is_rejected(self):
        for metric in ('outside_candidates', 'fallback'):
            data = copy.deepcopy(rows())
            data[1]['expected'][metric] = 0.5
            with self.assertRaises(ValueError):
                summarize(data, (42, 43), ('test',))

    def test_forbidden_counterfactual_continuations_are_rejected(self):
        data = rows()
        counterfactual = next(row for row in data if row['rule'] == 'counterfactual_episode')
        counterfactual['expected']['after_counterfactual_end'] = 0.1
        with self.assertRaises(ValueError):
            summarize(data, (42, 43), ('test',))


if __name__ == '__main__':
    unittest.main()
