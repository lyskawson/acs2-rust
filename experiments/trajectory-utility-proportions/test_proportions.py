import unittest
from unittest.mock import patch

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

    def test_grid_labels_use_the_same_fraction_notation_as_hindsight_proportion(self):
        self.assertEqual([label(MEAN, p) for p in (0.25, 0.5, 0.75, 1.0)],
                         [f'{MEAN}_p{suffix}' for suffix in ('1of4', '1of2', '3of4', '1of1')])

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


class MeasurementValidationTest(unittest.TestCase):
    def capped_row(self, extended):
        registered = [
            dict(success=True, steps=2, shortest_distance=1, weight=0.2),
            dict(success=False, steps=5, shortest_distance=4, weight=0.3),
            dict(success=False, steps=5, shortest_distance=7, weight=0.5),
        ]
        shortest = 4.9
        mean = sum(pair['weight'] * long['steps'] for pair, long in zip(registered, extended))
        return dict(wall_seconds_eval=1, evaluation_blocks={'test': dict(
            success=0.2, mean_success_steps=2, starts=registered)}, test_caps=dict(
            registered=dict(cap=5, success=0.2, mean_shortest_path=shortest, mean_steps=4.4,
                            mean_steps_over_shortest=4.4 / shortest),
            extended=dict(cap=100, success=0.5, mean_shortest_path=shortest, mean_steps=mean,
                          mean_steps_over_shortest=mean / shortest, starts=extended,
                          heads=None, cumulative_cpu_seconds=0.1)))

    def test_pair_prefix_despite_consistent_aggregates(self):
        from validate_proportions import validate_test_caps

        config = dict(cap=5, second_test_cap=100, per_start_diagnostics=True)
        tail = [dict(success=True, steps=7), dict(success=False, steps=100)]
        validate_test_caps(self.capped_row([dict(success=True, steps=2), *tail]), config)
        with self.assertRaisesRegex(ValueError, 'disagrees with its registered pair'):
            validate_test_caps(self.capped_row([dict(success=True, steps=3), *tail]), config)
        with self.assertRaisesRegex(ValueError, 'disagrees with its registered pair'):
            validate_test_caps(self.capped_row([dict(success=True, steps=2),
                                               dict(success=True, steps=5), tail[-1]]), config)

    def test_unit_share_rejects_even_one_hindsight_request(self):
        from proportions import BLOCKS
        from validate_proportions import validate_row

        config = dict(second_test_cap=100, evaluation_blocks=BLOCKS, per_start_diagnostics=True,
                      trajectory_utility=dict(tu_share=1.0, candidates=8, refresh_steps=2000))
        draws = dict(original=5, hindsight=0, trajectory_utility=10)
        row = dict(agent=MEAN, test_caps={}, agent_parameters={'trajectory_utility': dict(
            tu_share=1.0, route_probabilities=dict(original=1 / 3, hindsight=0, trajectory_utility=2 / 3))},
            trajectory_utility=dict(diagnostics=dict(draw_requests=draws)))
        with (patch('validate_proportions.validate_test_caps'),
              patch('validate_proportions.validate_measures') as delegate):
            validate_row(row, config, MEAN, 42, 'pinned', 1000)
            delegate.assert_called_once()
            draws.update(hindsight=1, trajectory_utility=9)
            with self.assertRaisesRegex(ValueError, 'has a hindsight request'):
                validate_row(row, config, MEAN, 42, 'pinned', 1000)


class LimitModelTest(unittest.TestCase):
    def test_limits_cover_margins_and_floor_the_upper_ratio(self):
        from cost_forecast import time_limits

        full = time_limits([300, 500, 700], 500, 1.2, False, True)
        self.assertEqual(full['ordinary_limit_seconds'], 1320)
        self.assertEqual(full['archive_seed_spread'], 1.4)
        self.assertIsNone(full['scout_limit_seconds'])
        below_one = time_limits([300, 500, 700], 500, 0.7, False, True)
        self.assertEqual(below_one['ordinary_limit_seconds'], 1140)
        prefix = time_limits([300, 500, 700], 500, 1.2, False, False)
        self.assertEqual(prefix['ordinary_limit_seconds'], 1980)
        heavy = time_limits([300, 500, 700], 500, 1.2, True, False)
        self.assertEqual(heavy['scout_limit_seconds'], 1200)
        self.assertEqual(heavy['after_scout_initial_limit_seconds'], 2520)
        self.assertIsNone(heavy['ordinary_limit_seconds'])

    def test_ratio_band_starts_at_100k(self):
        from cost_forecast import late_ratios

        records = [dict(checkpoint=step, overhead_ratio=ratio)
                   for step, ratio in [(30000, 3.0), (99999, 0.1), (100000, 1.4),
                                       (300000, 1.1), (1000000, 1.2)]]
        self.assertEqual(late_ratios(records), [1.4, 1.1, 1.2])

    def test_reused_and_gated_cells_reserve_exactly_nineteen_seed_slots(self):
        from cost_forecast import reservation_seconds

        rows = [dict(scout_limit_seconds=1000, after_scout_initial_limit_seconds=2300),
                dict(scout_limit_seconds=None, full_pilot_reused=True, ordinary_limit_seconds=300),
                dict(scout_limit_seconds=None, full_pilot_reused=False, ordinary_limit_seconds=450)]
        self.assertEqual(reservation_seconds(rows, 20), 59400)
