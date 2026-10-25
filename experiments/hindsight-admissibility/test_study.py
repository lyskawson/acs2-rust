import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from execution import APPROVED_SHA256, approved, batches, grid_manifest, pilot_references, pilot_subset, study_scope
from protocol import ROOT
from grid_authorization import require_derivation
from grid_identity import verify_references
from submit import submit


class StudyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.plan = copy.deepcopy(approved(Path(__file__).with_name('plan.json')))
        fixture = json.loads((ROOT / 'fixtures/hindsight_admissibility_regression.json').read_text())
        rows = [row for config in fixture['configurations'] for row in config['rows']]
        data = ''.join(json.dumps(row) + '\n' for row in rows).encode()
        (self.root / 'pilot-rows.jsonl').write_bytes(data)
        self.plan['cost_pilot']['rows_sha256'] = hashlib.sha256(data).hexdigest()
        for config in self.plan['configurations']:
            config['targets'] = [1000, 3000]
        (self.root / 'pilot').mkdir()
        for config, agent in batches(self.plan):
            (self.root / 'pilot' / (config['id'] + '_' + agent + '.jsonl')).write_bytes(pilot_subset(self.plan, config, agent, self.root))

    def test_only_the_approved_plan_and_thirty_hour_confirmation_are_accepted(self):
        plan = Path(__file__).with_name('plan.json').read_bytes()
        approval = Path(__file__).with_name('approval.json').read_bytes()
        (self.root / 'approved-plan.json').write_bytes(plan)
        (self.root / 'approval.json').write_bytes(approval)
        self.assertEqual(approved(self.root / 'approved-plan.json')['runs'], 240)
        (self.root / 'approved-plan.json').write_bytes(plan + b' ')
        with self.assertRaises(ValueError):
            approved(self.root / 'approved-plan.json')
        (self.root / 'approved-plan.json').write_bytes(plan)
        changed = json.loads(approval)
        changed['phase_cpu_hour_limit'] = 31
        (self.root / 'approval.json').write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            approved(self.root / 'approved-plan.json')

    def test_every_twenty_seed_manifest_has_one_fixed_arm_and_all_pilot_blocks(self):
        scope = study_scope(self.plan, self.root)
        self.assertEqual(len(scope['batch_roots']), 12)
        for index, (config, agent) in enumerate(batches(self.plan)):
            manifest = grid_manifest(self.plan, index, self.root, self.root)
            self.assertEqual(manifest['configurations'], [config])
            self.assertEqual(manifest['agents'], [agent])
            self.assertEqual(manifest['seeds'], list(range(42, 62)))
            self.assertEqual(manifest['phase_cpu_hour_limit'], 30)
            self.assertEqual(manifest['pilot_allocation_cpu_seconds'], 1723)
            reference = next(iter(manifest['identity_references'].values()))[0]
            self.assertEqual(reference['blocks'], ['training', 'validation', 'test'])
            self.assertEqual(reference['targets'], config['targets'])
            for key, value in [('seeds', [42]), ('phase_cpu_hour_limit', 31), ('unknown', 1)]:
                with self.assertRaises(ValueError):
                    require_derivation(dict(manifest, **{key: value}), manifest)

    def test_pilot_references_are_partitioned_by_task_and_arm_before_bit_comparison(self):
        for index, (config, agent) in enumerate(batches(self.plan)):
            manifest = grid_manifest(self.plan, index, self.root, self.root)
            reference = next(iter(manifest['identity_references'].values()))[0]
            rows = [json.loads(line) for line in Path(reference['path']).read_text().splitlines()]
            self.assertEqual({(row['task'], row['agent']) for row in rows}, {(config['task'], agent)})
            result = verify_references(manifest, dict(configuration_id=config['id'], agent=agent, seed=42), rows)
            self.assertEqual(result[0]['rows'], 2)
            rows[0]['population_classifiers'] += 1
            with self.assertRaises(ValueError):
                verify_references(manifest, dict(configuration_id=config['id'], agent=agent, seed=42), rows)

    def test_invalid_batch_indices_and_substituted_pilot_cells_are_refused(self):
        for index in (-1, 12, True):
            with self.assertRaises(ValueError):
                grid_manifest(self.plan, index, self.root, self.root)
        config, agent = batches(self.plan)[0]
        path = self.root / 'pilot' / (config['id'] + '_' + agent + '.jsonl')
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaises(ValueError):
            pilot_references(self.plan, config, agent, self.root, self.root)

    def test_foreign_batch_root_is_refused_before_accounting_or_scheduler_calls(self):
        scope = self.root / 'scope.json'
        scope.write_text(json.dumps(study_scope(self.plan, self.root)))
        manifest = grid_manifest(self.plan, 0, self.root, self.root)
        manifest.update(budget_scope=str(scope), budget_scope_sha256=hashlib.sha256(scope.read_bytes()).hexdigest(),
                        authorization_sha256=APPROVED_SHA256)
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        with patch('submit.execute') as executor:
            with self.assertRaisesRegex(ValueError, 'outside the registered accounting scope'):
                submit(self.root / 'foreign', path, self.root / 'source')
            executor.assert_not_called()


if __name__ == '__main__':
    unittest.main()
