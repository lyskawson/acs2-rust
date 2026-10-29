import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from approved_execution import APPROVED_SHA256
from execution import derive, measurement_identity
from protocol import EXPERIMENT, ROOT
from verify import resolved_batch, verify_manifest


class VerificationTests(unittest.TestCase):
    def test_archive_mapping_uses_current_and_failed_roots_before_the_shared_parent(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary).resolve()
            root = base / 'active'
            root.mkdir()
            (root / 'pilot-registration.json').write_text(json.dumps(dict(destination='/cluster/study/active',
                previous_pilot=dict(root='/cluster/study', mirror_relative='../original'),
                failed_pilot=dict(root='/cluster/study/failed', mirror_relative='../failed'))))
            self.assertEqual(resolved_batch(root, '/cluster/study/active/pilot'), root / 'pilot')
            self.assertEqual(resolved_batch(root, '/cluster/study/failed/pilot'), base / 'failed/pilot')
            self.assertEqual(resolved_batch(root, '/cluster/study/pilot'), base / 'original/pilot')
            with self.assertRaises(ValueError):
                resolved_batch(root, '/outside/pilot')

    def test_offline_manifest_reconstruction_refuses_budget_seed_proportion_and_identity_changes(self):
        plan = json.loads((ROOT / 'experiments/replay-schedule-and-order/plan.json').read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / 'screening'
            batch.mkdir()
            (root / 'scope.json').write_text('{}')
            (root / 'approval.json').write_text('{}')
            (batch / 'approval.json').write_text('{}')
            (batch / 'ledger.json').write_text('[]')
            operations = root / 'execution-operations/manifest.json'
            operations.parent.mkdir()
            operations.write_text(json.dumps(dict(commit='a' * 40, files={})))
            declared = Path('/cluster/study/active')
            references = {'required': [dict(path='frozen', sha256='b' * 64)]}
            manifest = dict(derive(plan, 'screening'), experiment=EXPERIMENT, partition=plan['execution']['partition'],
                cpu_model=plan['execution']['cpu_model'], memory_mib=plan['forecast']['memory_mib'], concurrency=20,
                phase_cpu_hour_limit=60, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28',
                **measurement_identity(), identity_references=references,
                authorization_plan=str(declared / 'approved-plan.json'), authorization_sha256=APPROVED_SHA256,
                budget_scope=str(declared / 'scope.json'), budget_scope_sha256=hashlib.sha256((root / 'scope.json').read_bytes()).hexdigest(),
                stop_path=str(declared / 'STOP.json'), operations_manifest=str(declared / 'execution-operations/manifest.json'),
                operations_manifest_sha256=hashlib.sha256(operations.read_bytes()).hexdigest(), operations_commit='a' * 40,
                job_name='tu-replay-screening')
            def save(value):
                data = (json.dumps(value) + '\n').encode()
                (batch / 'plan.json').write_bytes(data)
                (batch / 'plan.sha256').write_text(hashlib.sha256(data).hexdigest())
            with patch('verify.study_references', return_value=references):
                save(manifest)
                self.assertEqual(verify_manifest(root, batch, plan), manifest)
                for kind in ('budget', 'seed', 'proportion', 'identity'):
                    changed = copy.deepcopy(manifest)
                    if kind == 'budget': changed['phase_cpu_hour_limit'] = 61
                    elif kind == 'seed': changed['seeds'][-1] = 61
                    elif kind == 'identity': changed['identity_references'] = {}
                    else: changed['configurations'][0]['agent_settings']['acs2her_shortened']['relabeled_proportion'] = 0.25
                    save(changed)
                    with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, 'mechanical derivation'):
                        verify_manifest(root, batch, plan)


if __name__ == '__main__':
    unittest.main()
