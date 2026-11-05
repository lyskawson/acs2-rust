import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from protocol import ARM, ROOT
from execution import approved, continuation, continuation_scope, authorization_manifest, validate_inputs, APPROVED_SHA256
from grid_authorization import require_derivation
from grid_protocol import atomic_json
from submit import run_batches


class ContinuationTests(unittest.TestCase):
    def test_all_registered_assignments_are_derived_once_without_repeating_seed42(self):
        plan,_=approved()
        assignments=[]
        for purpose,sizes,seeds in [('diagnostics',[24],range(43,47)),('compact',[30,30,30],range(47,62))]:
            manifest=continuation(plan,purpose)
            batches=run_batches(manifest)
            self.assertEqual([len(batch) for batch in batches],sizes)
            self.assertEqual(manifest['seeds'],list(seeds))
            self.assertEqual(manifest['phase_cpu_hour_limit'],14)
            self.assertEqual(manifest['concurrency'],20)
            self.assertEqual(manifest['pilot_allocation_cpu_seconds'],0)
            for batch in batches:
                for config,arm,seed in batch:
                    original=next(item for item in plan['configurations'] if item['id']==config['id'])
                    for field in ('targets','budget_steps','registration','cap','protocol','encoding'):
                        self.assertEqual(config[field],original[field])
                    self.assertEqual(config.get('per_start_diagnostics',False),purpose=='diagnostics')
                    self.assertEqual(config['agent_settings'],{})
                    self.assertEqual(arm,ARM)
                    assignments.append((config['id'],arm,seed))
        self.assertEqual(len(assignments),114)
        self.assertEqual(len(set(assignments)),114)
        self.assertEqual({seed for _,_,seed in assignments},set(range(43,62)))
        hard=827+sum(max(config['limits_minutes'][arm] for config,arm,_ in batch)*60*len(batch)
            for purpose in ('diagnostics','compact') for batch in run_batches(continuation(plan,purpose)))
        self.assertEqual(hard,49787)
        self.assertLessEqual(hard,14*3600)
        with self.assertRaises(ValueError): continuation(plan,'pilot')

    def test_exact_approval_rejects_changed_limits_and_pilot_offsets(self):
        source=Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'approved-plan.json').write_bytes((source/'plan.json').read_bytes())
            original=json.loads((source/'approval.json').read_text())
            atomic_json(root/'approval.json',original)
            self.assertEqual(approved(root/'approved-plan.json')[1],original)
            for field,value in [('approved',False),('phase_cpu_hour_limit',16),('includes_pilot_cpu_seconds',0),('plan_sha256','x'*64)]:
                atomic_json(root/'approval.json',dict(original,**{field:value}))
                with self.subTest(field=field),self.assertRaises(ValueError): approved(root/'approved-plan.json')

    def test_scoped_manifests_reject_mutated_seeds_limits_targets_and_binary(self):
        source=Path(__file__).parent
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'approved-plan.json').write_bytes((source/'plan.json').read_bytes())
            (root/'approval.json').write_bytes((source/'approval.json').read_bytes())
            (root/'continuation').mkdir()
            atomic_json(root/'continuation/scope.json',continuation_scope(root))
            proposed=dict(stage='approved_grid',purpose='compact',authorization_plan=str(root/'approved-plan.json'),authorization_sha256=APPROVED_SHA256)
            with patch('execution.validate_inputs'):
                expected=authorization_manifest(proposed,root/'approved-plan.json')
            for field,value in [('seeds',[42]),('agents',['acs2her_shortened']),('phase_cpu_hour_limit',16),('binary_sha256','x'*64)]:
                with self.subTest(field=field),self.assertRaises(ValueError): require_derivation(dict(expected,**{field:value}),expected)
            changed=copy.deepcopy(expected)
            changed['configurations'][0]['targets']=[1000]
            with self.assertRaises(ValueError): require_derivation(changed,expected)
            scope=continuation_scope(root)
            self.assertEqual(scope['immutable_batch_roots'],[str(root/'pilot')])
            self.assertEqual(scope['accounting_root'],str(root))
            self.assertEqual(scope['batch_roots'],[str(root/name) for name in ('pilot','diagnostics','compact')])

    def test_every_registered_pilot_input_checksum_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            names=('pilot-summary','pilot-registration','reference-index','model-index')
            for name in names: (root/(name+'.json')).write_text('{}\n')
            plan=dict(cost_pilot={name+'_sha256':hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest() for name in names})
            validate_inputs(root,plan)
            for name in names:
                path=root/(name+'.json')
                path.write_text('[]\n')
                with self.subTest(name=name),self.assertRaises(ValueError): validate_inputs(root,plan)
                path.write_text('{}\n')
