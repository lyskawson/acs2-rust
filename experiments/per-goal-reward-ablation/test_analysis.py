import copy
import math
import statistics
import unittest
from analyze import calculate,difference,interval,sustained,verdict
from protocol import ARMS,A,A1,E,S,E1,S1,CONTRASTS,T19


class AnalysisTests(unittest.TestCase):
    def data(self):
        configs=[];inputs={};names={}
        base=[[0.03,0.04,0.02,0.05,0.06,0.07],
              [0.94,0.93,0.91,0.92,0.95,0.96],
              [0.2,0.995,0.3,0.4,0.996,0.997],
              [0.1,0.6,0.2,0.3,0.7,0.8],
              [0.6,0.1,0.7,0.8,0.2,0.3],
              [0.2,0.25,0.4,0.5,0.3,0.35],
              [0.2,0.22,0.1,0.3,0.4,0.6]]
        for i,values in enumerate(base):
            task='m'+str(i);names[task]='Thesis '+str(i)
            config=dict(task=task,targets=[1000,3000,10000],group='no_aliasing' if i<3 else 'aliasing',
                        registration=dict(blocks=dict(test=dict(random_floor=0.15 if i==0 else 0.01,reachable_within_cap=1))))
            configs.append(config)
            for ai,arm in enumerate(ARMS):
                for seed in range(42,62):
                    noise=(seed-51.5)*((ai+1)**2)*(0.000001 if i==2 else 0.000013)
                    final=values[ai]+noise
                    first=(0.94+noise if i==1 and arm==A else 0.08+ai*0.017+noise)
                    rows=[]
                    for j,(target,test) in enumerate(zip(config['targets'],[first,(first+final)/2,final])):
                        rows.append(dict(task=task,agent=arm,seed=seed,nominal_step=target,evaluation_blocks={
                            'training':dict(success=0.7-ai*0.02+j*0.01),
                            'validation':dict(success=0.6+ai*0.03-j*0.01), 'test':dict(success=test)}))
                    inputs[task,arm,seed]=rows
        return configs,inputs,names

    def test_pairs_signs_interactions_and_test_block_against_manual_nonzero_variance_calculation(self):
        configs,inputs,names=self.data();result=calculate(configs,inputs,names)
        weights={'ACS2 r(g) - ACS2':[(A1,1),(A,-1)],'E+1 - E':[(E1,1),(E,-1)],'S+1 - S':[(S1,1),(S,-1)],
                 'E+1 - ACS2 r(g)':[(E1,1),(A1,-1)],'S+1 - E+1':[(S1,1),(E1,-1)],
                 'memory x values':[(E1,1),(E,-1),(A1,-1),(A,1)],'copies x values':[(S1,1),(S,-1),(E1,-1),(E,1)]}
        for row in result['comparisons']:
            values=[]
            for seed in range(42,62):
                measures={arm:([r['evaluation_blocks']['test']['success'] for r in inputs[row['task'],arm,seed]]) for arm in ARMS}
                measure={a:statistics.mean(v) if row['measure']=='primary' else v[-1] for a,v in measures.items()}
                values.append(sum(measure[a]*w for a,w in weights[row['contrast']]))
            mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(20)
            self.assertAlmostEqual(row['mean'],mean,places=15);self.assertAlmostEqual(row['se'],se,places=15)
            self.assertAlmostEqual(row['ci_low'],mean-T19*se,places=15);self.assertAlmostEqual(row['ci_high'],mean+T19*se,places=15)
            self.assertGreater(row['se'],1e-8)
            self.assertEqual(row['maze'],names[row['task']])
            self.assertEqual(row['group'],next(c['group'] for c in configs if c['task']==row['task']))

    def test_labels_group_exclusions_and_joint_counts_are_contrast_and_measure_specific(self):
        configs,inputs,names=self.data();r=calculate(configs,inputs,names)
        labels={x['task']:x for x in r['labels']}
        self.assertTrue(labels['m0']['floor']);self.assertTrue(labels['m1']['too_easy'])
        c=lambda task,contrast,measure:next(x for x in r['comparisons'] if (x['task'],x['contrast'],x['measure'])==(task,contrast,measure))
        self.assertFalse(c('m0','E+1 - E','primary')['counted'])
        self.assertFalse(c('m1','E+1 - E','primary')['counted'])
        self.assertTrue(c('m1','E+1 - E','final')['counted'])
        self.assertTrue(c('m2','memory x values','final')['ceiling'])
        self.assertFalse(c('m2','memory x values','final')['counted'])
        self.assertFalse(c('m2','E+1 - E','final')['ceiling'])
        self.assertTrue(c('m2','memory x values','primary')['counted'])
        for x in r['grouped']:
            selected=[y for y in r['comparisons'] if y['contrast']==x['contrast'] and y['measure']==x['measure'] and y['counted'] and (x['group']=='all' or x['group']==y['group'])]
            self.assertEqual(x['n'],len(selected));self.assertEqual(x['A_plus'],sum(y['ci_low']>0 for y in selected));self.assertEqual(x['A_minus'],sum(y['ci_high']<0 for y in selected))
            self.assertEqual('verdict' in x,x['group']!='all')
        self.assertEqual(len(r['comparisons']),7*7*2)
        group=lambda g,m:next(x for x in r['grouped'] if x['group']==g and x['measure']==m and x['contrast']=='E+1 - E')
        self.assertEqual(group('no_aliasing','primary')['n'],1)
        self.assertEqual(group('aliasing','primary')['n'],4)
        self.assertEqual(group('no_aliasing','final')['n'],2)

    def test_verdict_thresholds_empty_and_touching_zero_and_interaction_language(self):
        self.assertEqual(verdict(2,0,4,'E+1 - E'),'improves')
        self.assertEqual(verdict(2,0,5,'E+1 - E'),'no clear effect')
        self.assertEqual(verdict(0,3,5,'memory x values'),'values gain less with memory')
        self.assertEqual(verdict(3,0,5,'copies x values'),'copies gain more from values')
        self.assertEqual(verdict(1,1,5,'copies x values'),'mixed')
        self.assertEqual(verdict(0,0,0,'E+1 - E'),'not assessed')
        self.assertEqual(interval([0.0]*20)['ci_low'],0)

    def test_sustained_uses_all_later_points_without_interpolation(self):
        self.assertEqual(sustained([1000,3000,10000,30000],[0.96,0.94,0.95,0.97]),10000)
        self.assertEqual(sustained([1000,3000],[0.99,0.949]),'not reached')
        configs,inputs,names=self.data();r=calculate(configs,inputs,names)
        self.assertEqual(len(r['sustained']),42)
        self.assertEqual(next(x for x in r['sustained'] if x['task']=='m1' and x['arm']==E1)['sustained_095'],10000)
        self.assertTrue(all(x['sustained_095']=='not reached' for x in r['sustained'] if x['task']=='m0'))

    def test_seed_checkpoint_and_arm_identity_cannot_be_swapped(self):
        for change in ['arm','seed','checkpoint']:
            configs,inputs,names=self.data();row=inputs['m3',E1,42][0]
            row[{'arm':'agent','seed':'seed','checkpoint':'nominal_step'}[change]]={'arm':A,'seed':43,'checkpoint':3000}[change]
            with self.assertRaises(ValueError):calculate(configs,inputs,names)
        configs,inputs,names=self.data();del inputs['m3',E1,61]
        with self.assertRaises(KeyError):calculate(configs,inputs,names)

    def test_analysis_command_reads_checksummed_rows_and_writes_thesis_names(self):
        import hashlib
        import json
        from pathlib import Path
        import runpy
        import tempfile
        from unittest.mock import patch
        import protocol
        configs,inputs,names=self.data()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'maze-names.json').write_text(json.dumps(names))
            index=[]
            for i,((task,arm,seed),rows) in enumerate(inputs.items()):
                path=root/(str(i)+'.jsonl');path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                index.append(dict(task=task,agent=arm,seed=seed,source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            path=root/'index.json';path.write_text(json.dumps(index));out=root/'output'
            with patch.object(protocol,'HERE',root),patch.object(protocol,'configurations',return_value=configs),patch('sys.argv',['analyze.py','--index',str(path),'--out',str(out)]):
                runpy.run_path(str(Path(__file__).with_name('analyze.py')),run_name='__main__')
            observed=json.loads((out/'results.json').read_text())
            self.assertEqual(observed['comparisons'],calculate(configs,inputs,names)['comparisons'])
            self.assertIn('Thesis 3',(out/'comparisons.csv').read_text())
            self.assertIn('A_plus',(out/'grouped.csv').read_text())

    def test_earlier_interval_checker_rejects_any_changed_bound_and_requires_all_36(self):
        from analyze import check_reference_intervals
        from protocol import OLD_TASKS
        configs,inputs,names=self.data();mapped={c['task']:OLD_TASKS[i] for i,c in enumerate(configs[:6])}
        for c in configs[:6]:c['task']=mapped[c['task']]
        inputs={(mapped[t],a,s):[dict(r,task=mapped[t]) for r in rows] for (t,a,s),rows in inputs.items() if t in mapped}
        names={v:v for v in mapped.values()};current=calculate(configs[:6],inputs,names)['comparisons']
        mapping={'E+1 - E':'E+1 - E','S+1 - S':'S+1 - S','copies x values':'interaction'}
        earlier=[dict(maze=r['task'],comparison=mapping[r['contrast']],metric=r['measure'],**{k:r[k] for k in ('mean','se','ci_low','ci_high')}) for r in current if r['contrast'] in mapping]
        self.assertEqual(check_reference_intervals(current,earlier),36)
        earlier[0]['ci_low']+=1e-9
        with self.assertRaises(ValueError):check_reference_intervals(current,earlier)
        with self.assertRaises(ValueError):check_reference_intervals(current,earlier[1:])
