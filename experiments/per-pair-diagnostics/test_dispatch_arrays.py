import unittest

from dispatch_arrays import array_units


class ArrayVerificationTests(unittest.TestCase):
    def test_every_completed_array_is_verified_exactly_once_in_ledger_order(self):
        ledger = [dict(array_id=index, runs=[dict(seed=42 + index)]) for index in range(3)]
        units = array_units(ledger)
        self.assertEqual(units, [[ledger[0]], [ledger[1]], [ledger[2]]])
        self.assertEqual([entry for unit in units for entry in unit], ledger)

    def test_retries_are_refused_instead_of_losing_cross_array_prefix_checks(self):
        with self.assertRaisesRegex(ValueError, 'prefix'):
            array_units([dict(array_id=2, retry_of=1)])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            array_units([dict(array_id=2)], recovered={1})


if __name__ == '__main__':
    unittest.main()
