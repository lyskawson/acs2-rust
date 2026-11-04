import copy
import unittest
from prepare_plan import forecast
from protocol import ARM, configurations
from archive import deduplicate


class ForecastTests(unittest.TestCase):
    def test_unequal_maze_costs_and_actual_array_limits_include_pilot_and_all_114_runs(self):
        summary=dict(allocation_cpu_seconds=999,runs=[dict(configuration_id=config['id'],agent=ARM,seed=42,
            cost=dict(body_wall_seconds=40+index*60,allocation_cpu_seconds=50+index*60,peak_rss_kib=10000+index*1000))
            for index,config in enumerate(configurations())])
        measured=forecast(summary)
        self.assertEqual(measured['future_runs'],114)
        self.assertEqual([array['runs'] for array in measured['arrays']],[24,30,30,30])
        self.assertAlmostEqual(measured['expected_total_cpu_hours'],(999+19*sum(50+i*60 for i in range(6)))/3600)
        limits=[3,5,7,9,11,13]
        self.assertEqual([cell['limit_minutes'] for cell in measured['cells']],limits)
        self.assertAlmostEqual(measured['hard_reservations_cpu_hours'],(999+60*(24*13+30*(5+9+13)))/3600)
        self.assertGreaterEqual(measured['requested_cpu_hour_limit'],measured['hard_reservations_cpu_hours'])
        changed=copy.deepcopy(summary)
        changed['runs'][0]['cost']['body_wall_seconds']=0
        with self.assertRaises(ValueError): forecast(changed)
        with self.assertRaises(ValueError): forecast(dict(summary,runs=summary['runs'][:-1]))

    def test_allocation_deduplication_uses_jobidraw_and_refuses_conflicts(self):
        first=dict(JobID='12_0',JobIDRaw='100',CPUTimeRAW='10')
        second=dict(JobID='12_1',JobIDRaw='101',CPUTimeRAW='20')
        self.assertEqual(deduplicate([first,second,dict(first)]),[first,second])
        with self.assertRaises(ValueError): deduplicate([first,dict(first,CPUTimeRAW='11')])
