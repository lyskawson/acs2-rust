import math
import json
from pathlib import Path
import tempfile
import unittest

from grid_analysis import ANALYSIS_ARMS, calculate, measurement_step
from grid_limits import approved_registration
from registration import MEAN, RANDOM, S1, TRIAL, checksum


class RegisteredAnalysisTest(unittest.TestCase):
    def test_measurement_cost_uses_the_timed_step_for_a_shared_allocation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            attempt = root / 'attempts' / 'a03'
            attempt.mkdir(parents=True)
            step = attempt / 'step.json'
            step.write_text(json.dumps(dict(job_id='902_0', job_id_raw='903', step_id='5', limit_seconds=180)))
            raw = attempt / 'measurement' / 'run' / 'rows.jsonl'
            manifest = {str(step.relative_to(root)): checksum(step)}
            rows = {
                '903.4': dict(JobID='902_0.4', State='COMPLETED', ExitCode='0:0', AllocCPUS='1', CPUTimeRAW='29'),
                '903.5': dict(JobID='902_0.5', State='COMPLETED', ExitCode='0:0', AllocCPUS='1', CPUTimeRAW='145'),
            }
            self.assertEqual(measurement_step(raw, manifest, root, rows), ('902_0.5', '903.5', 145))
            rows['903.5']['CPUTimeRAW'] = '181'
            with self.assertRaisesRegex(ValueError, 'does not match its accepted allocation'):
                measurement_step(raw, manifest, root, rows)

    def test_paired_nonconstant_seeds_floor_and_contrast_ceiling(self):
        study, _, _ = approved_registration()
        provider = dict(targets=[1000, 3000, 10000], registration={
            'blocks': {'test': {'random_floor_exact': '1/5', 'random_floor': .2, 'reachable_within_cap': 1.0}}
        })
        configs = {'distinct_checkpoints': dict(provider, group='no_aliasing'),
                   'floor': dict(provider, group='aliasing')}
        names = {task: task for task in configs}
        inputs = {}
        for offset, seed in enumerate(study['seeds']):
            fluctuation = offset / 1000
            for arm in ANALYSIS_ARMS:
                inputs['floor', arm, seed] = [.02, .04 + fluctuation, .05 + fluctuation]
            inputs['distinct_checkpoints', S1, seed] = [.4, .5, .96]
            inputs['distinct_checkpoints', RANDOM, seed] = [.45, .54, .97]
            inputs['distinct_checkpoints', MEAN, seed] = [.5 + fluctuation, .65, .991]
            inputs['distinct_checkpoints', TRIAL, seed] = [.63, .66 + 2 * fluctuation, .995]
        result, carried = calculate(study, configs, names, inputs,
                                    {'distinct_checkpoints': False, 'floor': True})
        compare = f'{TRIAL} - {MEAN}'
        pair = next(row for row in result['comparisons'] if row['task'] == 'distinct_checkpoints'
                    and row['contrast'] == compare and row['measure'] == 'primary')
        self.assertAlmostEqual(pair['mean'], (.13 + .01 + .004 + .0095) / 3)
        self.assertAlmostEqual(pair['se'], math.sqrt(35) / 3000 / math.sqrt(20))
        self.assertGreater(pair['ci_low'], 0)
        self.assertTrue(pair['counted'])
        final = next(row for row in result['comparisons'] if row['task'] == 'distinct_checkpoints'
                     and row['contrast'] == compare and row['measure'] == 'final')
        self.assertTrue(final['ceiling'])
        self.assertFalse(final['counted'])
        groups = {row['group']: row for row in result['grouped'] if row['contrast'] == compare and row['measure'] == 'primary'}
        self.assertEqual((groups['no_aliasing']['A_plus'], groups['no_aliasing']['verdict']), (1, 'improves'))
        self.assertEqual((groups['aliasing']['n'], groups['aliasing']['verdict']), (0, 'not assessed'))
        self.assertNotIn('verdict', groups['all'])
        self.assertEqual(carried, TRIAL)
        self.assertEqual(len(result['comparisons']), 24)
