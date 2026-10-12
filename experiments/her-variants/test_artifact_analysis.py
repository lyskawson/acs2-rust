import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analyze import allocation_report, analyze, entries, matched_entries, reference_resolver, require_derivation
from grid_protocol import atomic_json
from grid_scope import SACCT_FIELDS


class ArtifactAnalysisTests(unittest.TestCase):
    def test_reported_allocation_counts_every_scope_root_including_unread_and_empty_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            amounts={'tuning':15,'main-grid':20,'control-needed':3,'control-reused':37,'control-empty':0}
            scope=dict(batch_roots=['/registered/study/'+name for name in amounts],
                       pilot_allocation_cpu_seconds=137,authorization_sha256='approved')
            atomic_json(root/'scope.json',scope)
            child=dict(budget_scope='/registered/study/scope.json',
                       budget_scope_sha256=hashlib.sha256((root/'scope.json').read_bytes()).hexdigest(),
                       pilot_allocation_cpu_seconds=137,authorization_sha256='approved')
            for index,(name,amount) in enumerate(amounts.items()):
                if not amount: continue
                batch=root/name
                batch.mkdir()
                atomic_json(batch/'ledger.json',[dict(array_id=index+10,runs=[dict(limit_seconds=100)])])
                (batch/'accounting.psv').write_text(SACCT_FIELDS.replace(',','|')+'|\n'+
                    f'{index+10}_0|{index+10}_0|{amount}|00:00:01|1|1|0|COMPLETED|0:0|node|lem-cpu-normal|a|b|\n')
            plan=dict(configurations=[dict(id=name) for name in ['needed','reused','empty']],
                      forecast=dict(pilot_allocation_cpu_seconds=137),phase_cpu_hour_limit=60)
            checked={name:dict(allocation_cpu_seconds=amounts[name]) for name in ['tuning','main-grid','control-needed']}
            self.assertEqual(allocation_report(root,plan,child,checked),dict(
                allocated_cpu_seconds=212,pilot_allocated_cpu_seconds=137,allocation_by_root=amounts))

    def test_analysis_serializes_derived_pilot_cost_and_explicit_root_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            allocation=dict(allocated_cpu_seconds=212,pilot_allocated_cpu_seconds=137,
                            allocation_by_root={'empty':0,'nonempty':75})
            data=(dict(configurations=[]),{}, {}, {}, {}, {},allocation)
            with patch('analyze.collect',return_value=data):
                result=analyze('study','baseline','pilots',Path(temporary)/'analysis')
            for field,value in allocation.items():
                self.assertEqual(result[field],value)

    def test_analysis_reads_archived_verification_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'verification.json').write_text('archived')
            with patch('analyze.verify',return_value={'checked':True}), patch('analyze.catalog',return_value=({},None)):
                self.assertEqual(entries(root,None),({},dict(checked=True)))
            self.assertEqual((root/'verification.json').read_text(),'archived')

    def test_matched_two_thirds_reuses_twenty_final_grid_seeds_and_three_blocks(self):
        main={('maze','her_values',seed):dict(source='grid',rows=[dict(evaluation_blocks={name:{} for name in ['training','validation','test']})]) for seed in range(42,62)}
        extra={('maze','her_values',seed):dict(source='control',rows=main['maze','her_values',seed]['rows']) for seed in range(42,62)}
        group=matched_entries('maze',2/3,main,extra,list(range(42,62)))
        self.assertEqual(len(group),20)
        self.assertTrue(all(entry['source']=='grid' for entry in group))
        group=matched_entries('maze',1/3,main,extra,list(range(42,62)))
        self.assertTrue(all(entry['source']=='control' for entry in group))
        with self.assertRaises(ValueError): matched_entries('maze',2/3,main,extra,list(range(42,47)))
        changed=copy.deepcopy(main)
        changed['maze','her_values',42]['rows'][0]['evaluation_blocks']={'validation':{}}
        with self.assertRaises(ValueError): matched_entries('maze',2/3,changed,extra,list(range(42,62)))

    def test_local_verification_resolves_only_owned_reference_roots_without_rewriting_manifests(self):
        resolve=reference_resolver(Path('/local/study'),Path('/local/pilots'))
        self.assertEqual(resolve('/home/alelys2099/tu_runs/5/approved-her-study/tuning/rows.jsonl'),Path('/local/study/tuning/rows.jsonl'))
        self.assertEqual(resolve('/home/alelys2099/tu_runs/5/cost-pilot/rows.jsonl'),Path('/local/pilots/cost-pilot/rows.jsonl'))
        with self.assertRaises(ValueError): resolve('/home/alelys2099/mpx_runs/rows.jsonl')

    def test_analysis_rejects_nonmechanical_execution_manifests(self):
        expected=dict(seeds=list(range(42,62)),agents=['her_values'],configurations=[dict(cap=5,proportion=2/3)])
        require_derivation(dict(expected,metadata='extra'),expected)
        changed=copy.deepcopy(expected)
        changed['configurations'][0]['proportion']=1/3
        with self.assertRaises(ValueError): require_derivation(changed,expected)


if __name__ == '__main__': unittest.main()
