import hashlib
import csv
import statistics
import json
from pathlib import Path
import tempfile
import unittest

from analysis import summarize, truth_index
from analyze import analyze, process_row, process_view, distance_values
from post_hoc import curves, final_contrasts
from protocol import ARMS, configurations
from prepare_execution import APPROVED_SHA256
from validate_exports import validate


class PipelineTests(unittest.TestCase):
    def test_attached_pooled_pipeline_keeps_every_block_and_uses_action_q(self):
        truth = truth_index(dict(gamma=.95, reward=1000, actions=8, cap=3, pairs=[dict(start=[1, 1], goal=[1, 3],
            shortest_distance=2, value=950.0, action_values=[950.0, 902.5] + [0.0] * 6, optimal_actions=[0])]))
        def raw(action, weight):
            return dict(start='(1, 1)', goal='Goal { symbols: [Token(1), Token(3)] }', shortest_distance=2,
                first_action=action, weight=weight, success=action == 0, estimated_first_action_value=960.0,
                first_decision=dict(random_fallback=False, winners=[dict(specified_goal_attributes=2)]))
        row = dict(evaluation_blocks=dict(training=dict(starts=[raw(0, 1)]), validation=dict(starts=[raw(1, 1)]), test=dict(starts=[raw(1, 1)])))
        views = process_row(row, truth)
        self.assertEqual([record['q_star'] for record in views['pooled']], [950, 902.5, 902.5])
        measured, extra, flat = process_view(views['pooled'])
        self.assertEqual(measured['F'], 1 / 3)
        self.assertAlmostEqual(flat['estimate_minus_q'], (10 + 57.5 + 57.5) / 3)
        self.assertEqual(flat['winner_goal_attributes_2'], 1)
        self.assertTrue(extra['post_hoc'])
        distances = distance_values(views['pooled'])
        self.assertEqual(next(item for item in distances if item['group'] == 'failed')['estimate_minus_q'], 57.5)
        self.assertTrue(all(item['post_hoc'] for item in distances))

    def test_post_hoc_contrasts_and_curves_keep_five_paired_seeds_and_labels(self):
        metrics = {('maze4', arm, seed, 100, 'uniform_pairs'): dict(F=index + (seed - 42) * .1)
                   for index, arm in enumerate(ARMS) for seed in range(42, 47)}
        rows = final_contrasts(metrics, [dict(task='maze4', targets=[100])], 'uniform_pairs', ['F'])
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(item['post_hoc'] and item['view'] == 'uniform_pairs' for item in rows))
        chosen = next(item for item in rows if item['comparison'] == 'acs2her - acs2er_episodic')
        self.assertAlmostEqual(chosen['mean'], 1)
        self.assertLess(chosen['se'], 1e-14)
        plotted = curves(metrics, ['F'])
        self.assertEqual(len(plotted), 5)
        actual = next(item for item in plotted if item['arm'] == 'acs2')
        self.assertAlmostEqual(actual['mean'], .2)
        self.assertAlmostEqual(actual['se'], summarize([0, .1, .2, .3, .4])['se'])
        self.assertTrue(all(item['post_hoc'] for item in plotted))


    def test_complete_synthetic_pipeline_separates_registered_and_post_hoc_exports(self):
        source = Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('plan.json', 'approval.json'):
                (root / ('approved-plan.json' if name == 'plan.json' else name)).write_bytes((source / name).read_bytes())
            (root / 'models').mkdir()
            model_index, runs = [], []
            for config in configurations():
                pairs = []
                for index in range(7):
                    values = [902.5] * 8
                    values[index % 3] = 950.0
                    pairs.append(dict(start=[1, 1], goal=[2, index + 1], shortest_distance=2, value=950.0,
                                      action_values=values, optimal_actions=[index % 3]))
                model = dict(task=config['task'], gamma=.95, reward=1000, actions=8, cap=config['cap'], pairs=pairs)
                relative = 'models/' + config['task'] + '.json'
                (root / relative).write_text(json.dumps(model))
                model_index.append(dict(task=config['task'], file=relative, sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest()))
                for arm_index, arm in enumerate(ARMS):
                    for seed in range(42, 47):
                        rows = []
                        for point_index, point in enumerate(config['targets']):
                            blocks = {}
                            for index, name in enumerate(('training', 'validation', 'test')):
                                indices = ([0], [1, 2], [3, 4, 5, 6])[index]
                                records = []
                                for pair in indices:
                                    action = (seed + arm_index + index + pair + point_index) % 3
                                    records.append(dict(start='(1, 1)', goal='Goal { symbols: [Token(2), Token(%d)] }' % (pair + 1),
                                        shortest_distance=2, first_action=action, weight=1.0 / len(indices), success=action == pair % 3,
                                        estimated_first_action_value=960.0 + seed + point_index, first_decision=dict(random_fallback=False,
                                            winners=[dict(specified_goal_attributes=(arm_index + index) % 3)])))
                                blocks[name] = dict(starts=records)
                            rows.append(dict(nominal_step=point, evaluation_blocks=blocks))
                        relative = config['task'] + '_' + arm + '_' + str(seed) + '.jsonl'
                        (root / relative).write_text(''.join(json.dumps(row) + '\n' for row in rows))
                        runs.append(dict(configuration_id=config['id'], arm=arm, seed=seed, file=relative,
                            sha256=hashlib.sha256((root / relative).read_bytes()).hexdigest()))
            (root / 'model-index.json').write_text(json.dumps(model_index))
            verification = root / 'verification.json'
            verification.write_text(json.dumps(dict(verified=True, plan_sha256=APPROVED_SHA256, runs_index=runs)))
            output = root / 'analysis'
            result = analyze(root, output, verification)
            self.assertEqual(result['registered_comparisons'], 228)
            self.assertEqual(result['uniform_comparisons'], 330)
            self.assertEqual(result['modal_comparisons'], 180)
            self.assertEqual(result['seed_cells'], 5050)
            checked = validate(output, source / 'plan.json')
            self.assertTrue(checked['verified'])
            self.assertEqual(checked['registered_comparisons'], 228)
            task = configurations()[0]
            final_index = len(task['targets']) - 1
            rates = []
            for seed in range(42, 47):
                hits = sum((seed + block + pair + final_index) % 3 == pair % 3
                    for block, indices in enumerate(([0], [1, 2], [3, 4, 5, 6])) for pair in indices)
                rates.append(hits / 7)
            cells = list(csv.DictReader((output / 'post-hoc-seed-metrics.csv').read_text().splitlines()))
            uniform = [row for row in cells if row['maze'] == task['task'] and row['arm'] == 'acs2'
                       and row['view'] == 'uniform_pairs']
            self.assertEqual(len(uniform), 5)
            for row in uniform:
                self.assertEqual(float(row['F']), rates[int(row['seed']) - 42])
                self.assertAlmostEqual(float(row['C']), 3 / 7)
            pooled = [row for row in cells if row['maze'] == task['task'] and row['arm'] == 'acs2'
                      and row['view'] == 'pooled' and int(row['nominal_step']) == task['targets'][-1]]
            self.assertTrue(any(float(a['F']) != float(b['F']) for a, b in zip(uniform, pooled)))
            comparisons = list(csv.DictReader((output / 'post-hoc-uniform-comparisons.csv').read_text().splitlines()))
            baseline = next(row for row in comparisons if row['maze'] == task['task'] and row['comparison'] == 'acs2: F - C')
            differences = [rate - 3 / 7 for rate in rates]
            self.assertAlmostEqual(float(baseline['mean']), statistics.mean(differences))
            self.assertAlmostEqual(float(baseline['se']), statistics.stdev(differences) / 5**.5)
            self.assertGreater(float(baseline['se']), 0)



if __name__ == '__main__':
    unittest.main()
