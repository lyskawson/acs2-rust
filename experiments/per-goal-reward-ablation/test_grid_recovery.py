import json
from pathlib import Path
import tempfile
import unittest

from grid_execution import inspect, permitted_attempts, registered_plan, registered_runs
from grid_protocol import atomic_json
from grid_recovery import ARCHIVED_STOP, start_recovery


class InfrastructureRecoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.registered = registered_plan()
        self.runs = registered_runs(self.registered)
        ordinary = next(run for run in self.runs if run['phase'] == 'ordinary')
        scout = next(run for run in self.runs if run['phase'] == 'scout')
        self.ledger = [dict(array_id=10, runs=[ordinary]), dict(array_id=11, runs=[scout], depends_on=[10])]
        self.records = [
            dict(JobID='10_0', JobIDRaw='12', State='FAILED', ExitCode='1:0', CPUTimeRAW='13'),
            dict(JobID='11_0', JobIDRaw='11', State='CANCELLED', ExitCode='0:0', CPUTimeRAW='0', Start='None'),
        ]
        self.diagnostics = self.root / 'diagnostics.json'
        stop = dict(job='12', error='scontrol failed')
        atomic_json(self.root / 'ledger.json', self.ledger)
        atomic_json(self.root / 'STOP.json', stop)
        (self.root / 'logs').mkdir()
        (self.root / 'logs/10_0.err').write_text("Socket timed out\nCommand '['scontrol', 'show', 'job'\n")

        def proof(run, row):
            return dict(attempt=run['attempt_id'], job=row['JobID'], job_id_raw=row['JobIDRaw'],
                        state=row['State'], allocation_cpu_seconds=int(row['CPUTimeRAW']),
                        launch_record=False, measurement_bytes=0, resources_bytes=0)

        atomic_json(self.diagnostics, dict(stop=stop, allocations={
            'scontrol timeout before launch': [proof(ordinary, self.records[0])],
            'cancelled before allocation start': [proof(scout, self.records[1])],
        }))

    def recover(self):
        return start_recovery(self.root, self.ledger, self.records,
                              dict(spent_seconds=12978, reserved_seconds=0), self.runs,
                              self.diagnostics, 3583, lambda cmd: '0' if cmd[0] == 'sshare' else '')

    def test_infrastructure_retry_keeps_first_attempts_and_requires_a_fresh_scout(self):
        original_stop = (self.root / 'STOP.json').read_bytes()
        original_ledger = (self.root / 'ledger.json').read_bytes()
        result = self.recover()
        self.assertEqual(result['recovered'], 2)
        self.assertEqual((self.root / ARCHIVED_STOP).read_bytes(), original_stop)
        self.assertEqual((self.root / 'ledger.json').read_bytes(), original_ledger)
        receipt = json.loads((self.root / 'infrastructure-recovery.json').read_text())
        permitted = permitted_attempts(self.registered, receipt['excluded'])
        retries = [run for run in permitted.values() if run['attempt_id'].endswith('_a02')]
        self.assertEqual(len(retries), 2)
        for entry in self.ledger:
            original = entry['runs'][0]
            retry = permitted[original['attempt_id'][:-2] + '02']
            for field in ('seed', 'agent', 'phase', 'limit_seconds', 'forecast_high_cpu_seconds'):
                self.assertEqual(retry[field], original[field])
            self.assertNotEqual(retry['relative_directory'], original['relative_directory'])
        status = inspect(self.root, {}, self.registered, self.ledger, self.records)
        self.assertEqual(status['allocations'], dict(FAILED=1, CANCELLED=1))
        self.assertEqual(status['terminal_failures'], [])
        self.assertEqual(status['scout_gate']['checks'], [])
        retry_scout = next(run for run in retries if run['phase'] == 'scout')
        ledger = self.ledger + [dict(array_id=20, runs=[retry_scout])]
        timeout = dict(JobID='20_0', JobIDRaw='20', State='TIMEOUT', CPUTimeRAW='200')
        status = inspect(self.root, {}, self.registered, ledger, self.records + [timeout])
        self.assertEqual(status['terminal_failures'][0]['state'], 'TIMEOUT')
        self.assertTrue((self.root / 'STOP.json').exists())

    def test_timeout_measurement_and_unrelated_cancellation_are_ineligible(self):
        row = self.records[0]
        row['State'] = 'TIMEOUT'
        with self.assertRaisesRegex(ValueError, 'no matching infrastructure diagnosis'):
            self.recover()
        row['State'] = 'FAILED'
        (self.root / self.ledger[0]['runs'][0]['relative_directory']).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'measurement artifacts exist'):
            self.recover()
        (self.root / self.ledger[0]['runs'][0]['relative_directory']).rmdir()
        self.ledger[1]['depends_on'] = []
        with self.assertRaisesRegex(ValueError, 'lacks a dependency'):
            self.recover()
        self.assertTrue((self.root / 'STOP.json').exists())
        self.assertFalse((self.root / ARCHIVED_STOP).exists())
