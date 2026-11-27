import sys
import unittest
from unittest.mock import patch


with patch.dict(sys.modules):
    sys.modules.pop('pilot', None)
    import grid_stage


class AllocationStatusTests(unittest.TestCase):
    def account(self, *records):
        raw = 'JobID|JobIDRaw|State|CPUTimeRAW|Start|AllocCPUS|ExitCode\n'
        raw += '\n'.join(records) + '\n'
        ledger = [dict(array_id=50, allocation_count=len(records), limit_seconds=600)]
        with patch.object(grid_stage, 'sacct', return_value=(raw, {})):
            return grid_stage.allocation_status(ledger)[1]

    def test_pending_slots_with_shared_raw_id_are_distinct_queue_entries(self):
        try:
            account = self.account('50_0|50|PENDING|0|Unknown|1|0:0',
                                   '50_1|50|PENDING|0|Unknown|1|0:0')
        except ValueError as error:
            self.fail(f'Shared raw ID for two unstarted pending slots: {error}')
        self.assertEqual(account['states'], {'PENDING': 2})
        self.assertEqual(set(account['jobs']), {'50_0', '50_1'})
        self.assertEqual(account['spent_seconds'], 0)

    def test_running_allocation_reserves_its_unused_limit(self):
        account = self.account('50_0|51|RUNNING|180|2026-11-26T22:00:00|1|0:0',
                               '50_1|52|COMPLETED|110|2026-11-26T22:00:00|1|0:0')
        self.assertEqual(account['spent_seconds'], 290)
        self.assertEqual(account['reserved_seconds'], 420)
