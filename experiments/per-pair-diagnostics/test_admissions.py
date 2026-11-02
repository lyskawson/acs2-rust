import copy
import unittest

from archive import admissions_for, validate_pair_aggregates
from verify import verify_admissions


class AdmissionTests(unittest.TestCase):
    def test_pilot_audit_filters_later_shared_admissions(self):
        events = [dict(root='/study/pilot', bound_seconds=100), dict(root='/study/replication', bound_seconds=200)]
        self.assertEqual(admissions_for(events, '/study'), [events[0]])

    def test_total_cost_includes_pilot_and_every_array_hard_limit(self):
        ledgers = dict(pilot=[dict(array_id=1, runs=[dict(limit_seconds=60)] * 30)],
                       replication=[dict(array_id=2, runs=[dict(limit_seconds=120)] * 20)])
        accounts = {f'{array}_{index}': dict(CPUTimeRAW='10') for array, count in ((1, 30), (2, 20)) for index in range(count)}
        events = [dict(root='/study/' + name, test_only='accepted', grant='100', queue='',
                       balance=dict(spent_seconds=spent, reserved_seconds=0), new_limit_seconds=new,
                       bound_seconds=spent + new) for name, spent, new in [('pilot', 0, 1800), ('replication', 300, 2400)]]
        self.assertEqual(verify_admissions(events, ledgers, accounts, '/study'), 500)
        changed = copy.deepcopy(events)
        changed[1]['balance']['spent_seconds'] = 0
        with self.assertRaises(ValueError):
            verify_admissions(changed, ledgers, accounts, '/study')
        changed = copy.deepcopy(events)
        changed[1]['test_only'] = ''
        with self.assertRaises(ValueError):
            verify_admissions(changed, ledgers, accounts, '/study')
        with self.assertRaises(ValueError):
            verify_admissions(events, ledgers, accounts, '/study', limit=.1)


    def test_pair_success_and_value_records_must_reproduce_their_block_aggregates(self):
        block = dict(starts=[dict(weight=.25, success=True, estimated_first_action_value=200.0, discounted_return=1000.0),
                             dict(weight=.75, success=False, estimated_first_action_value=600.0, discounted_return=0.0)],
                     success=.25, value_diagnostics=dict(all=dict(mean_first_action_estimate=500.0, mean_discounted_return=250.0)))
        validate_pair_aggregates(block)
        for key, value in [('success', False), ('estimated_first_action_value', 201.0), ('discounted_return', 0.0)]:
            changed = copy.deepcopy(block)
            changed['starts'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'aggregate'):
                validate_pair_aggregates(changed)


if __name__ == '__main__':
    unittest.main()
