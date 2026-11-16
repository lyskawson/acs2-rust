import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import runpy
import statistics
import struct
import tempfile
import unittest
from unittest.mock import patch

import artifact_analysis as artifacts
from protocol import ARMS,A,A1,E,E1,S,S1,OLD_TASKS


class ArtifactAnalysisTests(unittest.TestCase):
    def data(self):
        from test_analysis import AnalysisTests
        return AnalysisTests().data()

    def write_source(self,path,rows,task='example',arm=E,seed=42):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        return dict(task=task,agent=arm,seed=seed,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def test_source_reader_normalizes_archived_E_keeps_test_values_and_discards_pair_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rows.jsonl'
            raw=dict(task='example',agent='er_episodic',seed=42,nominal_step=1000,knowledge=.7,
                population_classifiers=33,replay_diagnostics=dict(per_goal_rewards=dict(existing_pairs=165,updated_entries=11)),
                evaluation_blocks=dict(training=dict(success=.8),validation=dict(success=.6),
                    test=dict(success=.17,starts=[dict(large_unused_payload=list(range(1000)))])))
            entry=self.write_source(path,[raw]);rows,secondary=artifacts.read_source(entry)
            self.assertEqual(rows[0]['agent'],E)
            self.assertEqual(rows[0]['evaluation_blocks'],dict(test=dict(success=.17)))
            self.assertEqual(secondary[0]['per_goal_existing_pairs'],165)
            self.assertEqual(secondary[0]['knowledge'],.7)
            self.assertNotIn('large_unused_payload',json.dumps(rows))
            changed=dict(entry,seed=43)
            with self.assertRaises(ValueError):artifacts.read_source(changed)
            path.write_text(path.read_text()+' ')
            with self.assertRaises(ValueError):artifacts.read_source(entry)

    def test_seed_tables_and_curves_keep_pairing_group_and_checkpoint_identities(self):
        configs,inputs,names=self.data();tables=artifacts.tables(configs,inputs,names)
        self.assertEqual(len(tables['seed_scores']),7*6*20)
        self.assertEqual(len(tables['paired_differences']),7*7*2*20)
        self.assertEqual(len(tables['curves']),7*6*3)
        expected=lambda arm,seed:statistics.mean(r['evaluation_blocks']['test']['success'] for r in inputs['m3',arm,seed])
        row=next(r for r in tables['paired_differences'] if (r['task'],r['contrast'],r['measure'],r['seed'])==('m3','memory x values','primary',42))
        self.assertEqual(row['difference'],(expected(E1,42)-expected(E,42))-(expected(A1,42)-expected(A,42)))
        self.assertEqual(row['group'],'aliasing');self.assertEqual(row['maze'],'Thesis 3')
        curve=next(r for r in tables['curves'] if (r['task'],r['arm'],r['nominal_step'])==('m4',S1,3000))
        self.assertEqual(curve['mean'],statistics.mean(inputs['m4',S1,seed][1]['evaluation_blocks']['test']['success'] for seed in range(42,62)))
        self.assertEqual(curve['n'],20)
        entry=dict(task='unexpected',agent=E,seed=42,source='unused',sha256='unused')
        with self.assertRaises(ValueError):artifacts.load_sources([entry],configs)
        with self.assertRaises(ValueError):artifacts.load_sources([],configs)

    def write_analysis_inputs(self, root):
        configs,original,names=self.data()
        rename={f'm{i}':task for i,task in enumerate((*OLD_TASKS,'extra'))}
        inputs={}
        for config in configs:config['task']=rename[config['task']]
        names={rename[task]:name for task,name in names.items()}
        for (task,arm,seed),rows in original.items():
            rows=copy.deepcopy(rows)
            for row in rows:row['task']=rename[task]
            inputs[rename[task],arm,seed]=rows
        earlier=[]
        for task in OLD_TASKS:
            for label in ['E+1 - E','S+1 - S','interaction']:
                for measure in ['primary','final']:
                    values=[]
                    for seed in range(42,62):
                        scores={arm:(statistics.mean(r['evaluation_blocks']['test']['success'] for r in inputs[task,arm,seed])
                            if measure=='primary' else inputs[task,arm,seed][-1]['evaluation_blocks']['test']['success']) for arm in [E,E1,S,S1]}
                        value=(scores[E1]-scores[E]) if label=='E+1 - E' else ((scores[S1]-scores[S]) if label=='S+1 - S'
                            else (scores[S1]-scores[S])-(scores[E1]-scores[E]))
                        values.append(value)
                    mean=statistics.mean(values);se=statistics.stdev(values)/20**.5
                    earlier.append(dict(maze=task,comparison=label,metric=measure,mean=mean,se=se,
                        ci_low=mean-2.093024054408263*se,ci_high=mean+2.093024054408263*se))
        entries=[]
        for index,((task,arm,seed),rows) in enumerate(inputs.items()):
            raw=copy.deepcopy(rows)
            if arm==E:
                for row in raw:row['agent']='er_episodic'
            entries.append(self.write_source(root/f'run-{index}.jsonl',raw,task,arm,seed))
        index=root/'index.json'
        index.write_text(json.dumps(entries))
        reference=root/'earlier.json'
        reference.write_text(json.dumps(earlier))
        return configs, names, entries, index, reference

    def test_analysis_command_reads_full_sources_checks_36_intervals_and_writes_all_outputs(self):
        import protocol
        import grid_execution
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            configs, names, entries, index, reference = self.write_analysis_inputs(root)
            (root/'maze-names.json').write_text(json.dumps(names))
            out=root/'output'
            with patch.object(protocol,'HERE',root),patch.object(grid_execution,'registered_plan',return_value=dict(configurations=configs)),patch('sys.argv',
                    ['artifact_analysis.py','--index',str(index),'--earlier',str(reference),'--out',str(out)]),contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(Path(artifacts.__file__)),run_name='__main__')
            proof=json.loads((out/'analysis-verification.json').read_text())
            self.assertEqual(proof['earlier_intervals_bit_identical'],36)
            self.assertEqual(proof['full_runs'],840)
            with (out/'comparisons.csv').open() as stream:comparisons=list(csv.DictReader(stream))
            self.assertEqual(len(comparisons),98)
            self.assertEqual({r['maze'] for r in comparisons},set(names.values()))
            self.assertTrue((out/'curves.csv').exists());self.assertTrue((out/'sustained-095.csv').exists())
            self.assertTrue((out/'secondary-checkpoints.csv').exists());self.assertTrue((out/'paired-differences.csv').exists())
            projection=root/'projection.json'
            with patch.object(artifacts,'registered_plan',return_value=dict(configurations=configs)):
                projected=artifacts.project(index,projection)
            self.assertEqual(projected['full_runs'],840)
            with self.assertRaisesRegex(ValueError,'projection checksum'):
                artifacts.read_projection(index,projection,'0'*64)
            for entry in entries:
                Path(entry['source']).unlink()
            projected_out=root/'from-projection'
            artifacts.analyze(index,reference,projected_out,configs=configs,names=names,
                              projection=projection,projection_sha256=projected['projection_sha256'])
            for name in ['registered-results.json','secondary-checkpoints.csv','paired-differences.csv']:
                self.assertEqual((out/name).read_bytes(),(projected_out/name).read_bytes())

    def test_analyze_rejects_an_earlier_interval_with_one_bit_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs, names, _, index, reference = self.write_analysis_inputs(root)
            earlier = json.loads(reference.read_text())
            bits, = struct.unpack('>Q', struct.pack('>d', earlier[0]['ci_low']))
            earlier[0]['ci_low'], = struct.unpack('>d', struct.pack('>Q', bits ^ 1))
            changed = root / 'changed-earlier.json'
            changed.write_text(json.dumps(earlier))
            output = root / 'output'
            with self.assertRaisesRegex(ValueError, "earlier interval differs: .*'ci_low'"):
                artifacts.analyze(index, changed, output, configs=configs, names=names)
            self.assertFalse(output.exists())

    def write_projection(self, path, index, seed):
        rows = [dict(task='example', agent=E, seed=seed, nominal_step=1000,
                     evaluation_blocks=dict(test=dict(success=.17)))]
        path.write_text(json.dumps(dict(
            plan_sha256=artifacts.APPROVED_SHA256,
            source_index_sha256=artifacts.checksum(index),
            inputs=[[['example', E, seed], rows]],
            secondary=[],
        )))
        return artifacts.checksum(path)

    def test_read_projection_rejects_another_index_for_the_same_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'index.json'
            other = root / 'other-index.json'
            entry = dict(task='example', agent=E, seed=42)
            index.write_text(json.dumps([dict(entry, source='rows.jsonl')]))
            other.write_text(json.dumps([dict(entry, source='other-rows.jsonl')]))
            projection = root / 'projection.json'
            digest = self.write_projection(projection, other, seed=42)
            with self.assertRaisesRegex(ValueError, 'another registration or source index'):
                artifacts.read_projection(index, projection, digest)

    def test_read_projection_rejects_a_different_run_with_the_same_index_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'index.json'
            index.write_text(json.dumps([dict(task='example', agent=E, seed=42)]))
            projection = root / 'projection.json'
            digest = self.write_projection(projection, index, seed=43)
            with self.assertRaisesRegex(ValueError, 'projection does not match the run index'):
                artifacts.read_projection(index, projection, digest)


if __name__=='__main__':unittest.main()
