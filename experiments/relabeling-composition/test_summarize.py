import copy
import unittest

from summarize import METRICS, RULES, STRATEGIES, mean_and_se, report, summarize


def route(share=0.0, reward=0.0, **flags):
    values = dict(share=share, already_reached=0.0, after_counterfactual_end=0.0, done=0.0,
                  outside_candidates=0.0, reward=reward)
    values.update(flags)
    return values


def expected_values(seed, relabeled_share=1.0, relabeled_done=0.0):
    fallback = 1.0 - relabeled_share
    expected = dict.fromkeys(METRICS, 0.0)
    expected.update(admissible_share=relabeled_share, relabeled=relabeled_share, original=fallback,
                    fallback=fallback, no_admissible_goal=fallback, done=relabeled_done,
                    mean_reward_evaluations=1.0, mean_reach_evaluations=2.0, mean_objective_evaluations=3.0,
                    mean_scoring_evaluations=2.0, mean_selection_evaluations=0.0,
                    mean_provenance_evaluations=1.0, mean_reward=float(seed))
    expected['routes'] = {
        'original': route(fallback),
        'relabeled': route(relabeled_share, float(seed), done=relabeled_done),
    }
    return expected


def rows(shares=None):
    result = []
    for seed in (42, 43):
        for strategy in STRATEGIES:
            for rule in RULES:
                for filtered in (False, True):
                    share, done = (shares or {}).get(seed, (1.0, 0.0))
                    result.append(dict(schema=2, seed=seed, configuration='test', strategy=strategy,
                                       rule=rule, candidate_filter=filtered, actual_steps=20000,
                                       target_steps=20000, cap=5, episodes=4000, goal_pool='full',
                                       candidate_count_at_end=4, commit='test', source_state='clean',
                                       composition_uses_rng=False, relabeled_proportion=1.0,
                                       expected=expected_values(seed, share, done)))
    return result


def rejects(test, data):
    with test.assertRaises(ValueError):
        summarize(data, (42, 43), ('test',))


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
            rejects(self, changed)

    def test_variants_must_share_the_same_raw_trajectory(self):
        data = rows()
        data[0]['episodes'] += 1
        rejects(self, data)

    def test_schema_one_rows_are_rejected(self):
        data = rows()
        data[0]['schema'] = 1
        rejects(self, data)

    def test_invalid_fallback_or_filter_composition_is_rejected(self):
        for metric in ('outside_candidates', 'fallback'):
            data = copy.deepcopy(rows())
            data[1]['expected'][metric] = 0.5
            rejects(self, data)

    def test_forbidden_counterfactual_continuations_are_rejected(self):
        data = rows()
        counterfactual = next(row for row in data if row['rule'] == 'counterfactual_episode')
        counterfactual['expected']['after_counterfactual_end'] = 0.1
        rejects(self, data)

    def test_route_only_values_are_per_seed_ratios_of_step_weighted_joint_quantities(self):
        summary = summarize(rows({42: (0.5, 0.25), 43: (0.1, 0.09)}), (42, 43), ('test',))[0]
        self.assertEqual(summary['relabeled_only_seeds'], 2)
        self.assertAlmostEqual(summary['relabeled_only_done_mean'], 0.7, places=12)
        self.assertAlmostEqual(summary['relabeled_only_done_se'], 0.2, places=12)
        self.assertNotAlmostEqual(summary['relabeled_only_done_mean'], (0.25 + 0.09) / (0.5 + 0.1), places=3)
        self.assertAlmostEqual(summary['relabeled_only_mean_reward_mean'], (42 / 0.5 + 43 / 0.1) / 2, places=9)
        self.assertEqual(summary['original_only_seeds'], 2)
        self.assertEqual(summary['original_only_done_mean'], 0.0)

    def test_a_route_without_draws_has_no_mean_and_its_seed_count_says_so(self):
        summary = summarize(rows({42: (0.5, 0.0)}), (42, 43), ('test',))[0]
        self.assertEqual(summary['original_only_seeds'], 1)
        self.assertEqual(summary['original_only_done_mean'], 0.0)
        self.assertIsNone(summary['original_only_done_se'])
        summary = summarize(rows(), (42, 43), ('test',))[0]
        self.assertEqual(summary['original_only_seeds'], 0)
        self.assertIsNone(summary['original_only_mean_reward_mean'])
        self.assertIn('| — |', report([summary]))

    def test_routes_must_reproduce_the_pooled_values(self):
        for name in ('done', 'reward', 'share'):
            data = rows({42: (0.5, 0.25), 43: (0.1, 0.09)})
            data[3]['expected']['routes']['relabeled'][name] += 0.01
            rejects(self, data)

    def test_purposes_must_reproduce_the_total(self):
        for name in ('mean_scoring_evaluations', 'mean_selection_evaluations', 'mean_provenance_evaluations'):
            data = rows()
            data[5]['expected'][name] += 0.25
            rejects(self, data)

    def test_every_transition_selection_reads_no_objective_call(self):
        data = rows()
        row = next(row for row in data if row['rule'] == 'every_transition')
        row['expected']['mean_selection_evaluations'] = 0.5
        row['expected']['mean_provenance_evaluations'] = 0.5
        rejects(self, data)
        data = rows()
        row = next(row for row in data if row['rule'] == 'from_non_goal_state')
        row['expected']['mean_selection_evaluations'] = 0.5
        row['expected']['mean_provenance_evaluations'] = 0.5
        summarize(data, (42, 43), ('test',))

    def test_problematic_goals_on_the_wrong_route_are_rejected(self):
        for target, flag in (('original', 'already_reached'), ('original', 'outside_candidates'),
                             ('original', 'after_counterfactual_end')):
            data = rows({42: (0.5, 0.0), 43: (0.5, 0.0)})
            data[0]['expected']['routes'][target][flag] = 0.1
            data[0]['expected']['routes']['relabeled'][flag] = 0.0
            data[0]['expected'][flag] = 0.1
            rejects(self, data)
        data = rows()
        row = next(row for row in data if row['rule'] == 'from_non_goal_state')
        row['expected']['routes']['relabeled']['already_reached'] = 0.1
        row['expected']['already_reached'] = 0.1
        rejects(self, data)


if __name__ == '__main__':
    unittest.main()
