import copy
import hashlib
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from analysis import analyze_maze, diagnostic_summaries, metrics, scores, value_groups
from analyze import allocation_inventory
from verify import verify_pilot_inventory
from protocol import HER, SHORTENED, label


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = json.loads(Path(__file__).with_name('plan.json').read_text())

    def row(self, seed, point, index, point_index, count, pairs):
        def block(k):
            return dict(success=k / pairs, evaluation_pairs=pairs, evaluation_distribution='exhaustive_exact_weights',
                value_diagnostics=dict(all=dict(mean_estimate_minus_return=100 + point_index * 40 + index + seed),
                    successful=dict(mean_estimate_minus_return=-30 - point_index) if k else None))
        return dict(seed=seed, nominal_step=point, actual_steps=point, knowledge=0.4 + 0.05 * point_index,
            population_classifiers=100 + 20 * point_index + seed, population_numerosity=110 + 20 * point_index + seed,
            population_known_bytes_lower_bound=4096 + 100 * point_index, replay_logical_bytes=0, trajectory_logical_bytes=1024,
            wall_seconds_train=point_index + 1, wall_seconds_eval=0.2 * point_index, wall_seconds_total=point_index * 1.2 + 1,
            online_updates=0, replay_updates=3 * point, learning_update_counts=dict(full=3 * point), replay_diagnostics=None,
            evaluation_blocks=dict(training=block(pairs), validation=block(pairs - 1), test=block(count)))

    def dataset(self, config, selected=0.5):
        pairs = config['registration']['blocks']['test']['evaluation_pairs']
        arms = [label(SHORTENED, 2 / 3), label(HER, 2 / 3), 'acs2er_episodic', 'acs2', 'acs2er', label(HER, 0.5)]
        if selected != 2 / 3:
            arms.append(label(SHORTENED, selected))
        thresholds = self.plan['analysis']['thresholds'][config['id']]
        checks = self.plan['analysis']['baseline_threshold_crossing_checks'][config['id']]
        result = {}
        for index, arm in enumerate(arms):
            result[arm] = []
            for point_index, point in enumerate(config['targets']):
                if arm in ('acs2', 'acs2er'):
                    reached = [Fraction(thresholds[threshold]) for threshold, names in checks.items()
                               if names[arm] is not None and names[arm] <= point]
                    desired = max(reached, default=Fraction())
                    if point == config['targets'][-1]:
                        desired = Fraction(thresholds['acs2_final' if arm == 'acs2' else 'acs2er_final'])
                    total = desired * pairs * 20
                    self.assertEqual(total.denominator, 1)
                    base, remainder = divmod(total.numerator, 20)
                    counts = [base + int(seed < remainder) for seed in range(20)]
                else:
                    counts = [(seed * seed + 11 * index + point_index * 3) % (pairs + 1) for seed in range(20)]
                for offset, count in enumerate(counts):
                    result[arm].append(self.row(42 + offset, point, index, point_index, count, pairs))
        return result

    def test_registered_views_pair_same_seeds_and_preserve_exact_decomposition(self):
        for config in self.plan['configurations']:
            dataset = self.dataset(config)
            result = analyze_maze(dataset, config, self.plan, 0.5)
            self.assertEqual(len(result['comparisons']), 10)
            self.assertEqual(result['baseline_crossing_checks'], 4)
            self.assertEqual(result['decomposition']['point_checks'], 20 * len(config['targets']))
            rule = next(item for item in result['comparisons'] if item['view'] == 'checkpoint_mean' and item['left_alias'] == 'S@2/3')
            self.assertEqual(rule['right'], label(HER, 0.5))
            left = scores(dataset[rule['left']], config['targets'], 'checkpoint_mean')
            right = scores(dataset[rule['right']], config['targets'], 'checkpoint_mean')
            differences = [left[seed] - right[seed] for seed in range(42, 62)]
            mean = sum(differences, Fraction()) / 20
            se = math.sqrt(float(sum((value - mean) ** 2 for value in differences) / (19 * 20)))
            self.assertEqual(rule['difference']['exact_mean'], str(mean))
            self.assertAlmostEqual(rule['difference']['se'], se, places=15)
            self.assertAlmostEqual(rule['difference']['ci_low'], float(mean) - 2.093024054408263 * se, places=15)
            self.assertNotEqual(result['arms'][label(HER, 0.5)]['success']['checkpoint_mean']['mean'],
                                result['arms'][label(HER, 0.5)]['success']['final_point']['mean'])

    def test_two_thirds_selection_reuses_the_arm_and_omits_coincident_contrast(self):
        config = self.plan['configurations'][0]
        result = analyze_maze(self.dataset(config, 2 / 3), config, self.plan, 2 / 3)
        self.assertEqual(len(result['arms']), 6)
        self.assertEqual(len(result['comparisons']), 8)
        self.assertFalse(any(item['left_alias'] == 'S@selected' and item['right_alias'] == 'S@2/3' for item in result['comparisons']))

    def test_final_diagnostics_differ_from_checkpoint_mean_and_missing_walls_remain_null(self):
        config = self.plan['configurations'][0]
        rows = self.dataset(config)[label(SHORTENED, 2 / 3)]
        primary = diagnostic_summaries(rows, config['targets'], 'checkpoint_mean')
        final = diagnostic_summaries(rows, config['targets'], 'final_point')
        self.assertNotEqual(primary['population_classifiers']['mean'], final['population_classifiers']['mean'])
        self.assertNotEqual(primary['test_value_gap_all']['mean'], final['test_value_gap_all']['mean'])
        self.assertIsNone(final['wall_copy_share']['mean'])
        self.assertEqual(final['wall_copy_share']['n'], 0)

    def test_route_ratios_use_copy_or_all_draw_denominators_and_objective_cost_purposes(self):
        row = self.row(42, 1000, 0, 0, 2, 10)
        row['replay_diagnostics'] = dict(draws=100, fallbacks=10,
            routes=dict(relabeled=dict(draws=20, already_reached=2, after_counterfactual_end=3, done=4, reward_sum=4000)),
            wall_bumps=dict(unchanged_state=6, unchanged_state_terminal=0),
            objective_evaluations_by_purpose=dict(selection=dict(objective_evaluations=80, reach_evaluations=80, reward_evaluations=0)))
        value = metrics(row)
        self.assertEqual(value['realized_relabeled_share'], 0.2)
        self.assertEqual(value['fallback_share'], 0.1)
        self.assertEqual(value['trivial_copy_share'], 0.1)
        self.assertEqual(value['after_end_copy_share'], 0.15)
        self.assertEqual(value['rewarded_copy_share'], 0.2)
        self.assertEqual(value['wall_copy_share'], 0.3)
        self.assertEqual(value['objective_selection_objective_evaluations_per_draw'], 0.8)
        row['replay_diagnostics']['routes']['relabeled']['reward_sum'] = 5000
        with self.assertRaises(ValueError):
            metrics(row)

    def test_value_groups_recover_failures_and_keep_empty_groups_unmeasured(self):
        row = self.row(42, 1000, 0, 0, 2, 10)
        block = row['evaluation_blocks']['test']
        groups = value_groups(block)
        expected = (10 * block['value_diagnostics']['all']['mean_estimate_minus_return'] + 2 * 30) / 8
        self.assertAlmostEqual(groups['failed']['mean_estimate_minus_return'], expected)
        block['success'] = 1.0
        self.assertIsNone(value_groups(block)['failed'])
        block['success'] = 0.0
        self.assertIsNone(value_groups(block)['successful'])
        self.assertEqual(value_groups(block)['failed'], block['value_diagnostics']['all'])

    def test_validation_only_diagnostics_do_not_invent_test_value_gaps(self):
        row = self.row(42, 1000, 0, 0, 2, 10)
        row['evaluation_blocks'] = dict(validation=row['evaluation_blocks']['validation'])
        value = metrics(row)
        self.assertIsNone(value['test_value_gap_all'])
        self.assertIsNotNone(value['validation_value_gap_all'])

    def test_pilot_admission_log_is_preserved_as_an_exact_append_only_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = b'pilot allocation record\n'
            (root / 'pilot-admission-decisions.jsonl').write_bytes(original)
            (root / 'admission-decisions.jsonl').write_bytes(original + b'new allocation record\n')
            (root / 'checksums.json').write_text(json.dumps(dict(files={'admission-decisions.jsonl': hashlib.sha256(original).hexdigest()})))
            verify_pilot_inventory(root)
            (root / 'admission-decisions.jsonl').write_bytes(b'changed pilot allocation record\n')
            with self.assertRaises(ValueError):
                verify_pilot_inventory(root)

    def test_all_attempts_are_charged_and_process_cpu_includes_system_time(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / 'control'
            batch.mkdir()
            (root / 'scope.json').write_text(json.dumps(dict(batch_roots=['/cluster/control'])))
            ledger = []
            for index, cpu in [(1, (17, 3)), (2, (2, 1))]:
                path = batch / f'attempt-{index}'
                path.mkdir()
                (path / 'completion.json').write_text(json.dumps(dict(user_cpu_seconds=cpu[0], system_cpu_seconds=cpu[1])))
                ledger.append(dict(array_id=index, runs=[dict(configuration_id='maze', agent='acs2her', seed=42,
                    attempt_id=str(index), cause='first' if index == 1 else 'retry', relative_directory=path.name, limit_seconds=600)]))
            (batch / 'ledger.json').write_text(json.dumps(ledger))
            (batch / 'accounting.psv').write_text('JobID|JobIDRaw|CPUTimeRAW|State|ExitCode|\n1_0|101|100|FAILED|1:0|\n2_0|102|200|COMPLETED|0:0|\n')
            values = allocation_inventory(root)
            self.assertEqual(sum(value['allocation_cpu_seconds'] for value in values), 300)
            self.assertEqual([value['process_cpu_seconds'] for value in values], [20, 3])
            self.assertEqual(len(values), 2)


if __name__ == '__main__':
    unittest.main()
