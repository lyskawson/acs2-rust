import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from protocol import ROOT,ARMS,A1,E1,S1,configurations,recorded
from schema import validate_sequence,validate_row


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.addClassCleanup(cls.tmp.cleanup);cls.cells=[]
        for task in ['maze4','mazef1_alcs','woods102_alcs']:
            c=recorded(next(c for c in configurations() if c['task']==task),42);c['targets']=[2,5]
            out=Path(cls.tmp.name)/(task+'.jsonl')
            subprocess.run([str(ROOT/'target/release/acs2-measure'),'--task',task,'--cap',str(c['cap']),
                '--protocol',c['protocol'],'--agents',','.join(ARMS),'--seeds','42','--targets','2,5','--per-start','--out',str(out)],check=True,capture_output=True)
            rows=[json.loads(l) for l in out.read_text().splitlines()]
            for r in rows:r['source_state']='clean'
            cls.cells.append((c,rows))

    def test_all_six_arms_have_correct_learning_value_and_pair_records_in_both_protocols(self):
        for c,rows in self.cells:
            for arm in ARMS:
                selected=[r for r in rows if r['agent']==arm]
                validate_sequence(selected,c,arm,42,selected[0]['commit'])
                self.assertEqual(len(selected),2)
                if arm==A1:self.assertEqual(selected[-1]['replay_diagnostics']['per_goal_rewards']['reward_updates']['relabeled'],0)

    def test_validators_reject_ga_restore_value_counters_swapped_protocols_and_unregistered_arms(self):
        c,rows=self.cells[1];r=next(r for r in rows if r['agent']==A1)
        changes=[lambda x:x['preset'].update(do_ga=True),lambda x:x['agent_parameters'].update(restore_supported=True),
            lambda x:x['replay_diagnostics']['per_goal_rewards']['reward_updates'].update(relabeled=1),
            lambda x:x['replay_diagnostics']['per_goal_rewards'].update(existing_pairs=0),
            lambda x:x['goal_protocol'].update(id='maze-pairs-v1'),lambda x:x.update(agent='acs2her'),
            lambda x:x['evaluation_blocks']['test']['starts'].pop(),lambda x:x['evaluation_blocks']['validation'].update(random_floor=0.99)]
        for mutate in changes:
            changed=copy.deepcopy(r);mutate(changed)
            with self.assertRaises(ValueError):validate_row(changed,c,changed['agent'],42,r['commit'],2)
        for name in ['acs2er','acs2her','acs2her_nontrivial',A1+'_extra']:
            changed=copy.deepcopy(r);changed['agent']=name
            with self.assertRaises(ValueError):validate_row(changed,c,name,42,r['commit'],2)

    def test_cli_rejects_ga_override_and_restore_without_starting_learning(self):
        for flag in ['--do-ga','--restore']:
            out=Path(self.tmp.name)/(flag+'.jsonl')
            run=subprocess.run([str(ROOT/'target/release/acs2-measure'),'--task','maze4','--cap','5','--agents',A1,
                '--seeds','42','--targets','2','--out',str(out),flag,'true'],capture_output=True)
            self.assertNotEqual(run.returncode,0);self.assertFalse(out.exists())
