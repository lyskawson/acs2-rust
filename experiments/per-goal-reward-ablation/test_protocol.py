import copy
from fractions import Fraction
import json
import unittest
from pathlib import Path

from protocol import HERE,ROOT,ARMS,OLD_TASKS,configurations,new_cells
from register_protocol import build,partition,configuration,exact_floor


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry=json.loads((HERE/'geometry.json').read_text())
        cls.registration=json.loads((HERE/'pair-protocol.json').read_text())

    def test_complete_registration_rebuilds_without_learning_or_randomness(self):
        self.assertEqual(build(),self.registration)
        self.assertEqual(len(configurations()),28);self.assertEqual(len(new_cells()),138)
        self.assertEqual(len(ARMS),6)
        self.assertEqual(sum(c['group']=='no_aliasing' for c in configurations()),9)
        self.assertTrue(all(c['budget']==2000000 and 3<=c['diameter']<=10 for c in self.registration['configurations']))

    def test_binding_partition_and_component_pairs_and_fallback_only_when_required(self):
        for e,c in zip([e for e in self.geometry if e['task'].endswith('_alcs')],self.registration['configurations']):
            cells=list(map(tuple,e['walkable_cells']));ids={s:i for i,s in enumerate(cells)}
            expected={(s,g) for s in cells for g in cells if s!=g and e['distances'][ids[s]][ids[g]] is not None}
            seen=[]
            for block in ('training','validation','test'):
                entries=c['blocks'][block]['goal_starts']
                self.assertEqual({tuple(x['goal']) for x in entries},set(cells))
                for x in entries:
                    self.assertTrue(x['starts']);seen.extend((tuple(s),tuple(x['goal'])) for s in x['starts'])
            self.assertEqual(len(seen),len(set(seen)));self.assertEqual(set(seen),expected)
            if c['task'] in ('woods101_5_alcs','woods102_alcs'):
                self.assertEqual(len(c['components']),2)
                self.assertTrue(all((s[0]<5)==(g[0]<5) for s,g in seen))
            if c['task'] in ('mazef1_alcs','mazef2_alcs','woods100_alcs'):
                self.assertEqual(c['split_rule_used'],'second-nearest-second-farthest')
                for gi,g in enumerate(cells):
                    ranked=sorted([s for s in cells if s!=g],key=lambda s:(e['distances'][ids[s]][gi],*s))
                    val=c['blocks']['validation']['goal_starts'][gi]['starts'];test=c['blocks']['test']['goal_starts'][gi]['starts']
                    self.assertEqual((val,test),([[*ranked[1]]],[[*ranked[-2]]]) if gi%2==0 else ([[*ranked[-2]]],[[*ranked[1]]]))
            else:self.assertEqual(c['split_rule_used'],'maze-pairs-v1')

    def test_original_split_rule_reproduces_frozen_pair_blocks_on_all_six(self):
        prior=json.loads((ROOT/'experiments/maze-baselines/pair-protocol.json').read_text())
        for e in self.geometry[:6]:
            pools,_,rule=partition(e);self.assertEqual(rule,'maze-pairs-v1')
            c=next(c for c in prior['configurations'] if c['task']==e['task'])
            for name,pool in pools.items():
                self.assertEqual([dict(goal=list(g),starts=[list(s) for s in starts]) for g,starts in pool.items()],c['blocks'][name]['goal_starts'])
                self.assertEqual(str(exact_floor(e,pool,c['cap'])),c['blocks'][name]['random_floor_exact'])

    def test_random_floor_is_independently_recomputed_with_fraction_recurrence(self):
        for task in ['mazef1_alcs','woods100_alcs','woods101_5_alcs']:
            e=next(e for e in self.geometry if e['task']==task);c=next(c for c in self.registration['configurations'] if c['task']==task)
            cells=list(map(tuple,e['walkable_cells']))
            for block in c['blocks'].values():
                total=Fraction(0);n=0
                for x in block['goal_starts']:
                    gi=cells.index(tuple(x['goal']));p=[Fraction(i==gi) for i in range(len(cells))]
                    for _ in range(c['cap']):p=[Fraction(1) if i==gi else sum((p[j] for j in row),Fraction(0))/8 for i,row in enumerate(e['successors'])]
                    for s in x['starts']:total+=p[cells.index(tuple(s))];n+=1
                self.assertEqual(total/n,Fraction(block['random_floor_exact']))
        f=next(c for c in self.registration['configurations'] if c['task']=='mazef1_alcs')
        self.assertGreater(f['blocks']['test']['random_floor'],0.2)
        self.assertFalse(f['blocks']['test']['adequacy']['three_starts_per_goal'])
