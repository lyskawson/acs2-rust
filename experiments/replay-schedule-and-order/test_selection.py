import copy
import unittest

from execution import choose, authorization_manifest
from protocol import SHORTENED, SCHEDULES, SCREEN_SETTINGS, configurations, setting
from tuning import select
import test_protocol


def row(seed, point, count, pairs=100):
    return dict(seed=seed, nominal_step=point, knowledge=0.25,
                evaluation_blocks=dict(validation=dict(success=count / pairs, evaluation_pairs=pairs)))


def rows(counts, targets=(1000, 3000), pairs=100):
    return [row(seed, point, count, pairs) for seed in range(42, 47) for point, count in zip(targets, counts)]


class SelectionTests(unittest.TestCase):
    def test_selection_uses_within_seed_point_means_and_never_the_final_point_only(self):
        result = select(dict(base=rows([70, 30]), candidate=rows([10, 80])), [1000, 3000], ['candidate'])
        self.assertEqual(result['setting'], 'base')
        self.assertEqual(result['table']['base']['exact_mean'], '1/2')
        self.assertEqual(result['table']['candidate']['exact_mean'], '9/20')

    def test_base_wins_exact_and_tolerance_ties_but_not_a_larger_difference(self):
        for delta, expected in [(0, 'base'), (1, 'base'), (100, 'candidate')]:
            data = dict(base=rows([5_000_000_000_000] * 2, pairs=10**13),
                        candidate=rows([5_000_000_000_000 + delta] * 2, pairs=10**13))
            self.assertEqual(select(data, [1000, 3000], ['candidate'])['setting'], expected)

    def test_point_completeness_duplicate_and_test_block_access_are_refused(self):
        valid = dict(base=rows([40, 50]), candidate=rows([50, 60]))
        for kind in ('missing_seed', 'duplicate', 'test_block', 'missing_point', 'extra_setting'):
            data = copy.deepcopy(valid)
            if kind == 'duplicate': data['candidate'].append(data['candidate'][0])
            elif kind == 'test_block': data['candidate'][0]['evaluation_blocks']['test'] = dict(success=1.0)
            elif kind == 'missing_point': data['candidate'].pop()
            elif kind == 'extra_setting': data['control'] = rows([40, 50])
            else: data['candidate'] = data['candidate'][:-2]
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                select(data, [1000, 3000], ['candidate'])

    def fixture(self):
        plan = test_protocol.plan()
        entries, references, defaults = {}, {}, {}
        for config in configurations():
            points = config['targets']
            for seed in range(42, 47):
                references['screening-base', config['id'], seed] = dict(rows=[row(seed, point, 40) for point in points])
                for mode in SCREEN_SETTINGS:
                    entries[setting(config, SHORTENED, mode, ['validation'])['id'], SHORTENED, seed] = dict(
                        rows=[row(seed, point, 50 if mode == 'backward' else 40) for point in points])
            default = [dict(row(42, point, 40), evaluation_blocks=dict(
                validation=dict(success=0.4, evaluation_pairs=100), test=dict(success=0.9), training=dict(success=0.8))) for point in points]
            references['test', config['id'], 42] = dict(rows=default)
            defaults[config['id'], 42] = copy.deepcopy(default)
        return plan, entries, references, defaults

    def test_each_axis_is_selected_separately_and_requires_default_identity_first(self):
        plan, entries, refs, defaults = self.fixture()
        selected = choose(plan, entries, refs, defaults)
        self.assertTrue(all(cell['B']['setting'] == 'base' and cell['C']['setting'] == 'backward'
                            for cell in selected['selections'].values()))
        next(iter(defaults.values()))[0]['knowledge'] = 0.5
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            choose(plan, entries, refs, defaults)

    def test_default_identity_and_screening_catalog_cannot_be_missing_or_have_extra_entries(self):
        for kind in ('defaults', 'missing', 'extra'):
            plan, entries, refs, defaults = self.fixture()
            if kind == 'defaults': defaults.pop(next(iter(defaults)))
            elif kind == 'missing': entries.pop(next(iter(entries)))
            else: entries['unregistered', SHORTENED, 42] = dict(rows=[])
            with self.assertRaises(ValueError):
                choose(plan, entries, refs, defaults)

    def test_stage_one_cannot_authorize_any_screening_or_test_grid(self):
        for purpose in ('screening', 'test'):
            with self.assertRaisesRegex(ValueError, 'explicit plan and CPU-limit approval'):
                authorization_manifest(dict(stage='approved_grid', purpose=purpose, authorization_plan='/unapproved/plan.json'))


if __name__ == '__main__':
    unittest.main()
