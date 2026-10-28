import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis import analyze_maze
from protocol import HER, SHORTENED, label
import test_results
import test_tables
from validate_tables import validate
import verify


class VerificationGapTests(unittest.TestCase):
    def separated_curves(self, fixture, config):
        data = test_results.ResultTests.dataset(fixture, config)
        threshold = Fraction(fixture.plan['analysis']['thresholds'][config['id']]['acs2er_final'])
        pairs = config['registration']['blocks']['test']['evaluation_pairs']
        total = threshold * pairs * 20
        self.assertEqual(total.denominator, 1)
        base, remainder = divmod(total.numerator, 20)
        for row in data[label(HER, 0.5)]:
            k = 0 if row['nominal_step'] == config['targets'][0] else base + int(row['seed'] - 42 < remainder)
            row['evaluation_blocks']['test']['success'] = k / pairs
            row['evaluation_blocks']['test']['value_diagnostics']['successful'] = dict(mean_estimate_minus_return=0.0) if k else None
        for row in data[label(SHORTENED, 0.5)]:
            row['evaluation_blocks']['test']['success'] = 0.0
            row['evaluation_blocks']['test']['value_diagnostics']['successful'] = None
        return data

    def test_table_validator_accepts_an_exact_threshold_tie_inclusively(self):
        original = test_results.ResultTests.dataset
        def separated(fixture, config):
            with patch.object(test_results.ResultTests, 'dataset', original):
                return self.separated_curves(fixture, config)
        with tempfile.TemporaryDirectory() as temporary, patch.object(test_results.ResultTests, 'dataset', separated):
            directory = Path(temporary)
            plan = test_tables.TableTests().exported(directory)
            proof = validate(directory, plan)
            self.assertEqual(proof['registered_thresholds'], 24)

    def test_dose_control_crossings_use_its_own_distinct_curve(self):
        fixture = test_results.ResultTests()
        fixture.plan = json.loads(Path(__file__).with_name('plan.json').read_text())
        for config in fixture.plan['configurations']:
            result = analyze_maze(self.separated_curves(fixture, config), config, fixture.plan, 0.5)
            self.assertEqual(result['threshold_crossings']['H@1/2']['acs2er_final'], config['targets'][1])
            self.assertIsNone(result['threshold_crossings']['S@selected']['acs2er_final'])

    def retry_fixture(self, root, changed):
        plan = json.loads(Path(__file__).with_name('plan.json').read_text())
        configs = plan['configurations']
        selection = dict(selections={config['id']: dict(proportion=2 / 3) for config in configs})
        (root / 'selections.json').write_text(json.dumps(selection))
        tuning = {(config['id'], SHORTENED, seed): dict(rows=[], sha256='a' * 64)
                  for config in configs for seed in range(42, 62)}
        current = [dict(seed=42, nominal_step=1000, knowledge=0.25,
                        evaluation_blocks={name: dict(success=0.5) for name in ('training', 'validation', 'test')})]
        partial = copy.deepcopy(current)
        if changed:
            partial[0]['knowledge'] = 0.5
        control = {(config['id'], HER, seed): dict(rows=current, sha256='b' * 64)
                   for config in configs for seed in range(42, 62)}
        (root / 'selection-verification.json').write_text(json.dumps(dict(plan_sha256=verify.APPROVED_SHA256,
            selection_sha256=hashlib.sha256((root / 'selections.json').read_bytes()).hexdigest(),
            input_rows_sha256={'|'.join(map(str, key)): value['sha256'] for key, value in tuning.items()})))
        manifest = dict(configurations=[dict(config, registered_id=config['id'], row_arm='control',
            evaluation_blocks=['training', 'validation', 'test']) for config in configs])
        for name in ('tuning', 'control'):
            (root / name).mkdir()
            (root / name / 'ledger.json').write_text('[]')
        (root / 'recovery.json').write_text('{}')
        attempt = dict(configuration_id=configs[0]['id'], agent=HER, seed=42, array_id=99,
                       task_id=0, attempt_id='failed-prefix', cause='retry', path=root / 'failed')
        return plan, selection, tuning, control, manifest, attempt, partial

    def verify_retry(self, root, changed):
        plan, selection, tuning, control, manifest, attempt, partial = self.retry_fixture(root, changed)
        def catalog(path):
            return (tuning if Path(path).name == 'tuning' else control), dict(spent_seconds=100)
        with patch.object(verify, 'approved', return_value=plan), \
             patch.object(verify, 'verify_pilot_inventory', return_value=dict(files={})), \
             patch.object(verify, 'audit', return_value=(dict(allocation_cpu_seconds=4784, identity_rows=49), [], {})), \
             patch.object(verify, 'verify_manifest', return_value=manifest), \
             patch.object(verify, 'catalog', side_effect=catalog), \
             patch.object(verify, 'registered_references', return_value={}), \
             patch.object(verify, 'choose', return_value=selection), \
             patch.object(verify, 'archived_snapshot', return_value=dict(spent_seconds=4984, reserved_seconds=0)), \
             patch.object(verify, 'verify_admissions', return_value={}), \
             patch.object(verify, 'recovery_arrays', return_value={99}), \
             patch.object(verify, 'require_prefix'), \
             patch.object(verify, 'attempts', side_effect=lambda path, _: [attempt] if Path(path).name == 'control' else []), \
             patch.object(verify, 'inspect_attempt', return_value=(dict(status='failed'), partial)):
            return verify.verify_study(root)

    def test_verify_accepts_equal_partial_retry_and_refuses_a_changed_learning_field(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.verify_retry(Path(temporary), False)
            self.assertEqual(result[-1]['partial_retry_identity_rows'], 1)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, 'learning, evaluation or knowledge identity changed'):
                self.verify_retry(Path(temporary), True)


if __name__ == '__main__':
    unittest.main()
