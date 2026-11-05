import copy
import math
import unittest

from protocol import ARM, G, S, ARMS, configurations
from results import contrasts, decisions, paired, MECHANISM_METRICS
from dose import shares, check
from random_rule_schema import probability


class ResultsTests(unittest.TestCase):
    def test_paired_nonzero_variance_and_nulls_for_both_seed_counts(self):
        for n, t in [(5, 2.7764451051977987), (20, 2.093024054408263)]:
            seeds = list(range(42,42+n))
            left = {seed: 10_000+seed*100+i for i,seed in enumerate(seeds)}
            right = {seed: 10_000+seed*100 for seed in seeds}
            actual = paired(left,right,seeds)
            se = math.sqrt((n+1)/12)
            self.assertAlmostEqual(actual['mean'],(n-1)/2)
            self.assertAlmostEqual(actual['se'],se)
            self.assertAlmostEqual(actual['ci_low'],(n-1)/2-t*se)
            self.assertAlmostEqual(actual['ci_high'],(n-1)/2+t*se)
            right[42] = None
            self.assertIsNone(paired(left,right,seeds)['ci_low'])
            with self.assertRaises(ValueError): paired(left,{42:0},seeds)

    def test_contrasts_separate_arms_mazes_seeds_and_metrics(self):
        success, mechanism = {}, {}
        for m,config in enumerate(configurations()):
            for a,arm in enumerate(ARMS):
                for seed in range(42,62):
                    success[config['task'],arm,seed] = dict(primary=(a+1)**2*(seed+m),final=(a+2)**3*(seed+2*m))
                    if seed < 47:
                        mechanism[config['task'],arm,seed] = {metric:(a+1)**2*(k+3)*(seed+3*m) for k,metric in enumerate(MECHANISM_METRICS)}
        rows = contrasts(success,mechanism,configurations())
        self.assertEqual(len(rows),96)
        for row in rows:
            m = [c['task'] for c in configurations()].index(row['maze'])
            a,b = ARMS.index(row['left']),ARMS.index(row['right'])
            if row['metric']=='primary': expected=[((a+1)**2-(b+1)**2)*(seed+m) for seed in range(42,62)]
            elif row['metric']=='final': expected=[((a+2)**3-(b+2)**3)*(seed+2*m) for seed in range(42,62)]
            else: expected=[((a+1)**2-(b+1)**2)*(MECHANISM_METRICS.index(row['metric'])+3)*(seed+3*m) for seed in range(42,47)]
            self.assertEqual(row['seed_differences'],expected)
            self.assertGreater(row['se'],0)

    def test_disjoint_outcomes_ignore_woods14_and_final_and_joint_mechanism(self):
        mazes=['maze4','maze5','maze6','maze7','mazema','woods14']
        def fixture(a,b,modal=(1,1,1,0),goal=(1,1,1,0)):
            rows=[]
            for m,maze in enumerate(mazes):
                for left,right,group,metric,sign in [(G,ARM,'success','primary',a[m]),(ARM,S,'success','primary',b[m]),
                        (G,ARM,'success','final',-1),(G,ARM,'mechanism','F_mod',modal[m] if m<4 else 1),
                        (G,ARM,'mechanism','winner_goal_mass',goal[m] if m<4 else 1)]:
                    rows.append(dict(maze=maze,left=left,right=right,group=group,metric=metric,ci_low=sign-.1,ci_high=sign+.1))
            return rows
        for a,b,expected in [([1,1,1,0,0,-1],[0]*6,'rule choice'),
                ([1,1,1,0,0,-1],[1,1,1,0,0,-1],'rule choice beyond fewer updates'),
                ([1,1,0,-1,0,1],[1,1,1,0,0,0],'fewer updates'),
                ([1,1,1,-1,0,1],[1,1,1,0,0,0],'unresolved'),
                ([1,1,0,0,0,1],[1,1,0,0,0,1],'unresolved')]:
            measured=decisions(fixture(a,b))
            self.assertEqual(measured['outcome'],expected)
            self.assertTrue(measured['mechanism_separated'])
        self.assertFalse(decisions(fixture([0]*6,[0]*6,goal=(0,1,1,1)))['mechanism_separated'])
        rows=fixture([1,1,1,0,0,0],[0]*6)
        rows[0]['ci_low']=0
        self.assertEqual(decisions(rows)['A_positive'],2)
        with self.assertRaises(ValueError): decisions(rows+[rows[0]])

    def test_dose_means_seed_ratios_and_keeps_both_limits(self):
        def row(relabeled,all_original,skipped,goal_skipped=0):
            return dict(replay_diagnostics=dict(reward_updates=dict(
                original=dict(action_set_rules=all_original,skipped_without_goal=0,skipped_with_goal=0),
                relabeled=dict(action_set_rules=relabeled,skipped_without_goal=skipped,skipped_with_goal=goal_skipped))))
        self.assertEqual(shares(row(10,30,2,3)),dict(relabeled=.5,all=.125))
        config=configurations()[0]
        finals={}
        for seed in range(42,62):
            denominator=100_000*(seed-41)
            skips=round(probability('maze4')*denominator)
            finals['maze4',ARM,seed]=row(denominator,denominator,skips//2,skips-skips//2)
            finals['maze4',G,seed]=row(denominator,denominator,skips)
        actual=check(finals,[config])[0]
        self.assertFalse(actual['limited'])
        self.assertEqual(actual['all_difference'],0)
        baseline=copy.deepcopy(finals)
        for seed in range(42,62):
            denominator=100_000*(seed-41)
            skips=round((probability('maze4')+.03)*denominator)
            finals['maze4',ARM,seed]=row(denominator,denominator,skips//2,skips-skips//2)
            finals['maze4',G,seed]=row(denominator,denominator,skips)
        actual=check(finals,[config])[0]
        self.assertGreater(abs(actual['relabeled_difference']),.01)
        self.assertLess(abs(actual['relabeled_difference']),.1)
        self.assertLess(abs(actual['all_difference']),.02)
        self.assertTrue(actual['limited'])
        finals=copy.deepcopy(baseline)
        for seed in range(42,62):
            denominator=100_000*(seed-41)
            skips=round((probability('maze4')-.06)*denominator)
            finals['maze4',G,seed]=row(denominator,denominator,skips)
        actual=check(finals,[config])[0]
        self.assertLess(abs(actual['relabeled_difference']),.01)
        self.assertGreater(abs(actual['all_difference']),.02)
        self.assertTrue(actual['limited'])
