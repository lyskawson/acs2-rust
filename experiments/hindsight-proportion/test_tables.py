from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest

from analysis import analyze_maze, summary
from analyze import write_csv
import test_results
from validate_tables import CRITICAL, PLAN_SHA256, read_csv, select_exact, validate, validate_difference


class TableTests(unittest.TestCase):
    def record(self):
        mean = Fraction(19, 80)
        se = math.sqrt(7) / 80
        return dict(n=20, exact_mean=str(mean), mean=float(mean), se=se,
                    ci_low=float(mean) - CRITICAL * se, ci_high=float(mean) + CRITICAL * se)

    def scores(self):
        return ({seed: Fraction(seed - 42, 20) for seed in range(42, 62)},
                {seed: Fraction(seed - 42, 40) for seed in range(42, 62)})

    def test_independent_fraction_reconstruction_accepts_the_paired_interval(self):
        left, right = self.scores()
        self.assertEqual(validate_difference(self.record(), left, right), Fraction(19, 80))

    def test_unpaired_errors_wrong_seeds_and_wrong_critical_value_are_rejected(self):
        left, right = self.scores()
        altered = self.record()
        altered['se'] *= math.sqrt(5)
        with self.assertRaises(ValueError):
            validate_difference(altered, left, right)
        rotated = {seed: right[42 + (seed - 41) % 20] for seed in range(42, 62)}
        with self.assertRaises(ValueError):
            validate_difference(self.record(), left, rotated)
        altered = self.record()
        altered['ci_low'] = altered['mean'] - 1.96 * altered['se']
        with self.assertRaises(ValueError):
            validate_difference(altered, left, right)

    def test_exact_validation_selection_uses_the_registered_small_p_tie_rule(self):
        values = {1 / 3: [Fraction(3, 5)] * 5, 0.5: [Fraction(3, 5) + Fraction(1, 10**13)] * 5,
                  2 / 3: [Fraction(1, 2)] * 5, 0.8: [Fraction(2, 5)] * 5}
        self.assertEqual(select_exact(values), 1 / 3)
        values[0.5] = [Fraction(3, 5) + Fraction(1, 10**10)] * 5
        self.assertEqual(select_exact(values), 0.5)

    def exported(self, directory):
        plan_path = Path(__file__).with_name('plan.json')
        plan = json.loads(plan_path.read_text())
        fixture = test_results.ResultTests()
        fixture.plan = plan
        mazes, counts, scores, comparisons, thresholds, tuning, tuning_seeds = [], [], [], [], [], [], []
        tuning_means = {}
        for config in plan['configurations']:
            cfg = config['id']
            data = fixture.dataset(config)
            maze = analyze_maze(data, config, plan, 0.5)
            mazes.append(maze)
            pairs = config['registration']['blocks']['test']['evaluation_pairs']
            for arm, rows in data.items():
                counts.extend(dict(configuration_id=cfg, arm=arm, seed=row['seed'], nominal_step=row['nominal_step'],
                    successes=round(pairs * row['evaluation_blocks']['test']['success']), evaluation_pairs=pairs) for row in rows)
                for view in ('checkpoint_mean', 'final_point'):
                    scores.extend(dict(configuration_id=cfg, arm=arm, view=view, seed=seed, score_exact=value)
                        for seed, value in maze['arms'][arm]['success'][view]['seed_scores'].items())
            comparisons.extend(dict(configuration_id=cfg, view=row['view'], left_alias=row['left_alias'], right_alias=row['right_alias'],
                                    left=row['left'], right=row['right'], **row['difference']) for row in maze['comparisons'])
            thresholds.extend(dict(configuration_id=cfg, alias=alias, arm=maze['aliases'][alias], threshold=threshold,
                threshold_exact=maze['thresholds'][threshold], first_nominal_step=point)
                for alias, crossings in maze['threshold_crossings'].items() for threshold, point in crossings.items())
            for p, base in zip(plan['candidates'], (Fraction(2, 5), Fraction(3, 5), Fraction(1, 2), Fraction(9, 20))):
                values = [base + Fraction(seed - 42, 10000) for seed in range(42, 47)]
                tuning_seeds.extend(dict(configuration_id=cfg, proportion=p, seed=seed, score_exact=str(value))
                    for seed, value in zip(range(42, 47), values))
                value = summary(values)
                tuning.append(dict(configuration_id=cfg, proportion=p, n=5, mean=value['mean'], se=value['se'], selected=p == 0.5))
                tuning_means[cfg, p] = sum(values, Fraction()) / 5
        purposes = [dict(purpose=name, allocation_cpu_seconds=cost) for name, cost in
                    (('pilot', 4784), ('tuning', 100), ('control', 200), ('test', 300))]
        result = dict(plan_sha256=PLAN_SHA256, mazes=mazes,
                      selections=dict(selections={config['id']: dict(proportion=0.5) for config in plan['configurations']}),
                      execution=dict(allocated_cpu_seconds=5384))
        (directory / 'results.json').write_text(json.dumps(result))
        exports = {'success-counts': counts, 'seed-scores': scores, 'comparisons': comparisons, 'thresholds': thresholds,
                   'tuning': tuning, 'tuning-seed-scores': tuning_seeds, 'allocation-purposes': purposes,
                   'allocation-attempts': [dict(job_id_raw=index, **row) for index, row in enumerate(purposes)],
                   'tuning-across-mazes': [dict(proportion=p, mean=float(sum((tuning_means[config['id'], p]
                       for config in plan['configurations']), Fraction()) / 6)) for p in plan['candidates']]}
        for name, rows in exports.items():
            write_csv(directory / (name + '.csv'), list(rows[0]), rows)
        return plan_path

    def test_complete_export_is_reconstructed_and_wrong_dose_arm_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            plan = self.exported(directory)
            proof = validate(directory, plan)
            self.assertEqual(proof['paired_comparisons'], 60)
            self.assertEqual(proof['registered_thresholds'], 24)
            self.assertEqual(proof['baseline_crossing_checks'], 24)
            self.assertEqual(proof['tuning_rows'], 24)
            path = directory / 'comparisons.csv'
            rows = read_csv(path)
            next(row for row in rows if row['left_alias'] == 'S@2/3')['right'] = 'acs2her_p2of3'
            write_csv(path, list(rows[0]), rows)
            with self.assertRaisesRegex(ValueError, 'arm or proportion'):
                validate(directory, plan)


if __name__ == '__main__':
    unittest.main()
