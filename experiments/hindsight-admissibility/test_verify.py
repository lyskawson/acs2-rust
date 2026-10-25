import copy
from pathlib import Path
import tempfile
import unittest

from execution import approved
from verify import verify_admissions, verify_attempt_scope


class VerificationTests(unittest.TestCase):
    def ledger(self):
        plan = approved(Path(__file__).with_name('plan.json'))
        config, agent = plan['configurations'][0], plan['agents'][0]
        manifest = dict(configurations=[config], agents=[agent], seeds=plan['seeds'])
        runs = [dict(configuration_id=config['id'], agent=agent, seed=seed,
            limit_seconds=config['limits_minutes'][agent] * 60, attempt_id=f"{config['id']}_{agent}_s{seed}_a01")
            for seed in plan['seeds']]
        command = ['sbatch', '--partition=lem-cpu-normal', '--cpus-per-task=1', '--no-requeue', '--array=0-19%20',
                   '--time=' + str(config['limits_minutes'][agent])]
        return manifest, [dict(runs=runs, array_id=123, submission_response='123', command=command)]

    def test_missing_attempts_substituted_limits_and_receipts_are_refused(self):
        manifest, ledger = self.ledger()
        verify_attempt_scope(manifest, ledger)
        for field, value in [('seed', 62), ('limit_seconds', 9999), ('agent', 'acs2her')]:
            changed = copy.deepcopy(ledger)
            changed[0]['runs'][0][field] = value
            with self.assertRaises(ValueError):
                verify_attempt_scope(manifest, changed)
        changed = copy.deepcopy(ledger)
        changed[0]['submission_response'] = '124'
        with self.assertRaises(ValueError):
            verify_attempt_scope(manifest, changed)
        with self.assertRaises(ValueError):
            verify_attempt_scope(manifest, ledger * 2)

    def test_admissions_must_cover_exactly_the_registered_roots(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scope = dict(batch_roots=['/declared/one', '/declared/two'])
            valid = [dict(root=name, balance=dict(spent_seconds=0, reserved_seconds=0), remaining_limit_seconds=6000,
                new_limit_seconds=6000, bound_seconds=7723) for name in scope['batch_roots']]
            path = root / 'admission-decisions.jsonl'
            path.write_text(''.join(json.dumps(event) + '\n' for event in valid))
            verify_admissions(root, scope)
            for changed in [valid[:-1], valid + [valid[0]], valid + [dict(valid[0], root='/foreign')]]:
                path.write_text(''.join(json.dumps(event) + '\n' for event in changed))
                with self.assertRaises(ValueError):
                    verify_admissions(root, scope)
            changed = copy.deepcopy(valid)
            changed[0]['bound_seconds'] = 1
            path.write_text(''.join(json.dumps(event) + '\n' for event in changed))
            with self.assertRaises(ValueError):
                verify_admissions(root, scope)


if __name__ == '__main__':
    unittest.main()
