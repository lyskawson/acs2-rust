import copy
import unittest

from protocol import CANDIDATES, SHORTENED, configurations, label, setting
from execution import choose
from tuning import select


def row(p, seed, point, success=0.5):
    return dict(agent=SHORTENED, arm=label(SHORTENED, p), seed=seed, nominal_step=point,
                agent_parameters=dict(relabeled_proportion=p), knowledge=p,
                replay_updates=3 * point, evaluation_blocks={'validation': dict(success=success)})


class TuningTests(unittest.TestCase):
    def rows(self):
        return [row(p, seed, point, p / 2 + (seed - 42) / 100 + point / 100000)
                for p in CANDIDATES for seed in range(42, 47) for point in [1000, 3000]]

    def test_complete_validation_grid_uses_within_seed_mean_and_reports_se(self):
        result = select(self.rows(), CANDIDATES, [1000, 3000])
        self.assertEqual(result['proportion'], 0.8)
        for item in result['validation_scores']:
            self.assertAlmostEqual(item['mean'], item['proportion'] / 2 + 0.04)
            self.assertGreater(item['se'], 0)
            self.assertEqual(item['n'], 5)

    def test_reader_rejects_test_access_and_wrong_candidates_seeds_points_or_labels(self):
        for change in ['test_only', 'extra_test', 'seed', 'point', 'p', 'arm', 'agent', 'nonfinite']:
            rows = self.rows()
            if change == 'test_only':
                rows[0]['evaluation_blocks'] = {'test': dict(success=0.99)}
            elif change == 'extra_test':
                rows[0]['evaluation_blocks']['test'] = dict(success=0.99)
            elif change == 'seed':
                rows[0]['seed'] = 47
            elif change == 'point':
                rows[0]['nominal_step'] = 2000
            elif change == 'p':
                rows[0]['agent_parameters']['relabeled_proportion'] = 0.7
            elif change == 'arm':
                rows[0]['arm'] = label(SHORTENED, 0.5)
            elif change == 'agent':
                rows[0]['agent'] = 'acs2her'
            else:
                rows[0]['evaluation_blocks']['validation']['success'] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                select(rows, CANDIDATES, [1000, 3000])
        for rows in [self.rows()[:-1], self.rows() + [self.rows()[0]]]:
            with self.assertRaises(ValueError):
                select(rows, CANDIDATES, [1000, 3000])

    def test_ties_within_tolerance_choose_smaller_p_but_larger_gaps_do_not(self):
        for delta, expected in [(0.0, 1 / 3), (5e-13, 1 / 3), (2e-12, 0.5)]:
            rows = [row(p, seed, point, 0.6 + (delta if p == 0.5 else 0) if p in (1 / 3, 0.5) else 0.1)
                    for p in CANDIDATES for seed in range(42, 47) for point in [1000, 3000]]
            self.assertEqual(select(rows, CANDIDATES, [1000, 3000])['proportion'], expected)

    def test_registered_candidates_can_include_endpoints_and_another_interior_value(self):
        points = [10, 30]
        candidates = [0.0, 2 / 3, 0.731, 1.0]
        rows = [row(p, seed, point, p) for p in candidates for seed in range(42, 47) for point in points]
        self.assertEqual(select(rows, candidates, points)['proportion'], 1.0)

    def test_selection_checks_every_two_thirds_identity_before_choosing(self):
        entries, references = {}, {}
        for config in configurations():
            for p in CANDIDATES:
                for seed in range(42, 47):
                    rows = [row(p, seed, point, p) for point in config['targets']]
                    entries[setting(config, SHORTENED, p, ['validation'])['id'], SHORTENED, seed] = dict(rows=rows)
                    if p == 2 / 3:
                        refs = copy.deepcopy(rows)
                        for value in refs:
                            value['evaluation_blocks']['test'] = dict(success=0.01)
                            value['evaluation_blocks']['training'] = dict(success=0.99)
                        references[config['id'], seed] = dict(rows=refs)
        plan = dict(configurations=configurations(), candidates=CANDIDATES,
                    tuning=dict(seeds=list(range(42, 47)), selection='test'))
        selected = choose(plan, entries, references)
        self.assertTrue(all(value['proportion'] == 0.8 for value in selected['selections'].values()))
        key = next(iter(references))
        references[key]['rows'][0]['knowledge'] = 0.123
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            choose(plan, entries, references)


if __name__ == '__main__':
    unittest.main()
