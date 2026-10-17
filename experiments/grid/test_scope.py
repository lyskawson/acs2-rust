import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from grid_identity import verify_references
from grid_scope import SACCT_FIELDS, archived_snapshot, require_batch_root, snapshot, stop_path, verify_operations


class ScopeTests(unittest.TestCase):
    def test_batch_root_membership_resolves_paths_and_verifies_the_scope_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            registered = base / 'registered'
            registered.mkdir()
            alias = base / 'alias'
            alias.symlink_to(registered, target_is_directory=True)
            path = base / 'scope.json'
            path.write_text(json.dumps(dict(batch_roots=[str(alias)])))
            plan = dict(budget_scope=str(path), budget_scope_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            require_batch_root(registered, plan)
            require_batch_root(alias, plan)
            with self.assertRaisesRegex(ValueError, 'batch root.*accounting scope'):
                require_batch_root(base / 'sibling', plan)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                require_batch_root(registered, plan)
            require_batch_root(base / 'legacy', {})

    def test_archived_accounting_includes_unread_roots_and_explicit_zero_roots_with_registered_pilots(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            names=['tuning','main-grid','control-needed','control-reused','empty','not-created']
            scope=dict(batch_roots=['/registered/study/'+name for name in names],
                       authorization_sha256='approved',pilot_allocation_cpu_seconds=137)
            path=base/'scope.json'
            path.write_text(json.dumps(scope))
            plan=dict(budget_scope='/registered/study/scope.json',
                      budget_scope_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      authorization_sha256='approved',pilot_allocation_cpu_seconds=137)
            for index,name in enumerate(names[:4]):
                root=base/name
                root.mkdir()
                (root/'ledger.json').write_text(json.dumps([dict(array_id=index+10,runs=[dict(limit_seconds=100)])]))
                (root/'accounting.psv').write_text(SACCT_FIELDS.replace(',','|')+'|\n'+
                    f'{index+10}_0|{index+10}_0|{index+1}|00:00:01|1|1|0|COMPLETED|0:0|node|lem-cpu-normal|a|b|\n')
            (base/'empty').mkdir()
            (base/'empty/ledger.json').write_text('[]')
            result=archived_snapshot(base,plan)
            self.assertEqual(result['spent_seconds'],10)
            self.assertEqual(result['pilot_allocation_cpu_seconds'],137)
            self.assertEqual(result['reserved_seconds'],0)
            self.assertEqual(set(result['batch_states']),set(names))
            for name in names[4:]:
                self.assertEqual(result['batch_states'][name],dict(spent_seconds=0,reserved_seconds=0))

    def test_accounting_counts_all_roots_once_and_missing_future_roots_reserve_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            roots=[base/'tuning',base/'grid',base/'control']
            for root,array in zip(roots[:2],[10,20]):
                root.mkdir()
                (root/'ledger.json').write_text(json.dumps([dict(array_id=array,runs=[dict(limit_seconds=100)])]))
            scope=dict(batch_roots=list(map(str,roots)),pilot_allocation_cpu_seconds=5445,
                       authorization_sha256='approved',accounting_since='2026-10-12')
            path=base/'scope.json'
            path.write_text(json.dumps(scope))
            plan=dict(budget_scope=str(path),budget_scope_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      pilot_allocation_cpu_seconds=5445,authorization_sha256='approved')
            raw=SACCT_FIELDS.replace(',','|')+'|\n'
            raw+='10_0|10_0|30|00:00:25|1|30|0|COMPLETED|0:0|node|lem-cpu-normal|a|b|\n'
            raw+='20_0|20_0|20|00:00:15|1|20|0|RUNNING|0:0|node|lem-cpu-normal|a|b|\n'
            commands=[]
            def execute(command):
                commands.append(command)
                return raw
            result=snapshot(plan,execute)
            self.assertEqual((result['spent_seconds'],result['reserved_seconds']),(50,80))
            self.assertEqual(result['array_ids'],[10,20])
            self.assertIn('10,20',commands[0])
            self.assertEqual((roots[0]/'accounting.psv').read_text(),raw)
            (roots[1]/'ledger.json').write_text(json.dumps([dict(array_id=None,runs=[dict(limit_seconds=100)])]))
            with self.assertRaises(ValueError): snapshot(plan,execute)
            (roots[1]/'ledger.json').write_text(json.dumps([dict(array_id=10,runs=[dict(limit_seconds=100)])]))
            with self.assertRaises(ValueError): snapshot(plan,execute)

    def test_operations_snapshot_and_shared_stop_are_explicit_and_default_baselines_stay_local(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            file=base/'tool.py'
            file.write_text('value = 1\n')
            manifest=base/'operations.json'
            manifest.write_text(json.dumps(dict(files={'tool.py':hashlib.sha256(file.read_bytes()).hexdigest()})))
            plan=dict(operations_manifest=str(manifest),operations_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),stop_path=str(base/'STOP.json'))
            verify_operations(plan)
            file.write_text('value = 2\n')
            with self.assertRaises(ValueError): verify_operations(plan)
            self.assertEqual(stop_path(base/'child',plan),base/'STOP.json')
            self.assertEqual(stop_path(base/'child',{}),base/'child/STOP.json')

    def test_node_identity_guard_checks_artifact_hash_and_learning_fields_before_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'rows.jsonl'
            row=dict(agent='her_values',seed=42,nominal_step=1000,knowledge=0.5,
                     evaluation_blocks={'validation':{'success':0.25}})
            path.write_text(json.dumps(row)+'\n')
            key='maze|her_values|42'
            ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),targets=[1000],blocks=['validation'])
            plan=dict(identity_references={key:[ref]})
            entry=dict(configuration_id='maze',agent='her_values',seed=42)
            self.assertEqual(verify_references(plan,entry,[row])[0]['differences'],0)
            with self.assertRaises(ValueError): verify_references(plan,entry,[dict(row,knowledge=0.6)])
            path.write_text(json.dumps(dict(row,knowledge=0.6))+'\n')
            with self.assertRaises(ValueError): verify_references(plan,entry,[row])


if __name__ == '__main__': unittest.main()
