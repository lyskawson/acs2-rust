import json
from pathlib import Path
import tempfile
import unittest

from export_metadata import annotate_outputs, column_semantics
from grid_protocol import atomic_json
from protocol import E1, S1, MECHANISM_METRICS
from grid_identity import field_bits
from decision_metrics import decision_metrics


class ExportTests(unittest.TestCase):
    def test_annotation_preserves_all_seed_values_csv_and_registered_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [dict(maze='maze4', arm=arm, seed=seed,
                **{metric: (a + 1) * .17 + m * .031 + (seed - 42) * .007 for m, metric in enumerate(MECHANISM_METRICS)},
                D_tie_implied=.11 + a * .07, D_tie_share=.37 + (seed - 42) * .013,
                D_tie_weight=.09 + a * .03, D_different_weight=.51 + a * .02)
                for a, arm in enumerate((E1, S1)) for seed in range(42, 47)]
            payload = dict(plan_sha256='a' * 64, rows=rows)
            atomic_json(root / 'mechanism-seed-metrics.json', payload)
            preserved = {name: b'unchanged numeric export\n' for name in (
                'mechanism-seed-metrics.csv', 'registered-comparisons.csv', 'registered-comparisons.json', 'registered-decisions.json')}
            for name, data in preserved.items(): (root / name).write_bytes(data)
            annotate_outputs(root)
            result = json.loads((root / 'mechanism-seed-metrics.json').read_text())
            self.assertEqual(field_bits(result['rows']), field_bits(rows))
            self.assertEqual(result['plan_sha256'], payload['plan_sha256'])
            self.assertEqual(result['column_semantics'], column_semantics())
            scope = json.loads((root / 'output-scope.json').read_text())
            self.assertEqual(scope['column_semantics']['mechanism-seed-metrics'], result['column_semantics'])
            self.assertEqual(scope['registered_seed_metrics']['mechanism-seed-metrics'], list(MECHANISM_METRICS))
            self.assertEqual(set(scope['post_hoc_columns']['mechanism-seed-metrics']),
                {'D_tie_implied', 'D_tie_share', 'D_tie_weight', 'D_different_weight'})
            for name, data in preserved.items(): self.assertEqual((root / name).read_bytes(), data)
            first = {p.name: p.read_bytes() for p in root.iterdir()}
            annotate_outputs(root)
            self.assertEqual({p.name: p.read_bytes() for p in root.iterdir()}, first)

    def test_goal_specific_scores_can_fill_the_inherited_tie_counter_without_any_tie(self):
        records = []
        for goal, scores in enumerate(((10., 1.), (1., 10.))):
            self.assertEqual(scores.count(max(scores)), 1)
            action = scores.index(max(scores))
            records.append(dict(start=(0, 0), goal=(goal, 1), weight=.5, optimal={goal},
                first_action=action, random_fallback=False, success=True,
                estimated_first_action_value=max(scores), q_star=10., v_star=10.,
                winners=[dict(specified_goal_attributes=0)]))
        measured = decision_metrics(records)
        self.assertEqual(measured['D_tie_implied'], 1.)
        self.assertEqual(measured['D_tie_share'], 1.)
        self.assertIn('not evidence of action-score ties', column_semantics()['D_tie_warning'])
