import unittest

from analysis_rules import carry_share
from proportions import MEAN, S1, SHARES, label, pilot_cells
from validate_proportions import capped_steps


class PilotRegistrationTest(unittest.TestCase):
    def test_pilot_covers_all_seed_42_cells_and_reservations(self):
        settings, _, configs, runs = pilot_cells()
        self.assertEqual(len(configs), 28)
        self.assertEqual(len(runs), 168)
        self.assertEqual(len({run['run_id'] for run in runs}), 168)
        self.assertEqual({run['seed'] for run in runs}, {42})
        self.assertTrue(all(run['last_target'] >= 300_000 for run in runs))
        self.assertLessEqual(settings['reservation_seconds'] + settings['build_limit_seconds'], 40 * 3600)

    def test_derived_capped_steps(self):
        self.assertEqual(capped_steps(0.0, None, 10), 10)
        self.assertEqual(capped_steps(1.0, 4, 10), 4)
        self.assertEqual(capped_steps(0.375, 2, 10), 7)


class CarryRuleTest(unittest.TestCase):
    def test_validation_wins_even_when_test_favors_another_share(self):
        study = dict(seeds=range(42, 62))
        configs = dict(a=dict(group='aliasing', targets=[1000, 2000, 3000]),
                       b=dict(group='no_aliasing', targets=[1000, 3000]),
                       c=dict(group='no_aliasing', targets=[1000, 2000, 3000, 4000]))
        scores = dict(a=[0.31, 0.59, 0.65, 0.7, 0.9],
                      b=[0.72, 0.84, 0.9, 0.8, 0.8],
                      c=[0.6, 0.85, 0.79 + 1e-13, 0.5, 0.4])
        rows = {}
        for task, config in configs.items():
            targets = config['targets']
            for i, share in enumerate(sorted((0, 0.5, *SHARES))):
                arm = S1 if share == 0 else label(MEAN, share)
                for seed in study['seeds']:
                    rows[task, arm, seed] = [
                        dict(nominal_step=target, evaluation_blocks={
                            'validation': {'success': scores[task][i] + 0.02 * point - (seed - 42) * 0.001},
                            'test': {'success': 0.9 - share * 0.7},
                        }) for point, target in enumerate(targets)
                    ]
        actual = carry_share(study, configs, rows)
        self.assertEqual(actual['no_aliasing']['tu_share'], 0.25)
        self.assertEqual(actual['aliasing']['tu_share'], 1.0)
        self.assertAlmostEqual(actual['no_aliasing']['validation_primary_by_share'][0.25], 0.8555)
        for records in rows.values():
            for checkpoint in records:
                del checkpoint['evaluation_blocks']['test']
        self.assertEqual(carry_share(study, configs, rows), actual)


class LimitModelTest(unittest.TestCase):
    def test_limits_cover_archived_seed_and_attempt_spreads(self):
        from cost_forecast import time_limits

        ordinary = time_limits([300, 500, 700], 500, 1.2, False)
        self.assertEqual(ordinary['ordinary_limit_seconds'], 1320)
        self.assertEqual(ordinary['archive_seed_spread'], 1.4)
        self.assertIsNone(ordinary['scout_limit_seconds'])

        heavy = time_limits([300, 500, 700], 500, 1.2, True)
        self.assertEqual(heavy['scout_limit_seconds'], 1020)
        self.assertEqual(heavy['after_scout_initial_limit_seconds'], 2160)
        self.assertIsNone(heavy['ordinary_limit_seconds'])
