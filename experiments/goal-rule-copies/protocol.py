import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
sys.path.append(str(ROOT / 'experiments/per-pair-diagnostics'))
from grid_protocol import load_plan

EXPERIMENT = 'goal-rule-copies'
ARM = 'acs2her_shortened_goal_rules'
REFERENCES = ('acs2her_shortened', 'acs2er_episodic')
ARMS = (ARM, *REFERENCES)
BLOCKS = ['training', 'validation', 'test']
T4 = 2.7764451051977987
T19 = 2.093024054408263
ADMISSIBILITY_SHA = 'f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da'


def configurations():
    return copy.deepcopy(load_plan(ROOT / 'experiments/hindsight-admissibility/plan.json', ADMISSIBILITY_SHA)['configurations'])


def recorded(config, limit=20, per_pair=True):
    result = dict(copy.deepcopy(config), evaluation_blocks=BLOCKS, agent_settings={}, limits_minutes={ARM: limit})
    if per_pair:
        result['per_start_diagnostics'] = True
    return result


def pilot(identity):
    return dict(stage='authorized_cost_pilot', experiment=EXPERIMENT, **identity,
        agents=[ARM], seeds=[42], configurations=[recorded(config) for config in configurations()],
        partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=2048,
        concurrency=20, phase_cpu_hour_limit=2, pilot_allocation_cpu_seconds=0, accounting_since='2026-11-04')


def analysis_registration():
    return dict(success=dict(seeds=list(range(42, 62)), block='test', metrics=['primary', 'final'],
            primary='Equal-weight mean of checkpoint successes within each seed, all registered checkpoints.',
            comparisons=[[ARM, reference] for reference in REFERENCES], critical_value=T19, degrees_of_freedom=19),
        mechanism=dict(seeds=list(range(42, 47)), point='final', view='pooled',
            metrics=['F', 'F_mod', 'estimate_minus_q', 'estimate_above_v', 'within_start_slope'],
            comparisons=[[ARM, reference] for reference in REFERENCES], critical_value=T4, degrees_of_freedom=4),
        pooling='Concatenate all three blocks, preserve recorded weights, normalize total mass; equal block mass.',
        F='Chosen first action is optimal; exclude random fallbacks.',
        C='Best fixed action per start, using all pairs and their pooled weights.',
        modal='Most frequent weighted chosen action at each start among nonfallbacks; lowest action index breaks ties. F_mod is its optimal-action share on the same eligible pairs.',
        estimate_minus_q='Mean estimate minus capped Q*(start, chosen first action, goal), all pairs including fallbacks.',
        estimate_above_v='Weighted share among nonfallbacks with estimate > capped V*, with a strict binary64 comparison.',
        within_start='All pairs, including fallbacks. Demean estimate and chosen-action Q* separately within each start; sum weighted covariance and variances across starts. Slope=covariance/variance(Q*); SD=sqrt(weighted within-start variance/total mass). Zero denominators stay null.',
        tie='D uses unordered distinct goals with disjoint optimal sets at one start and nonfallbacks, product weights. D_tie_implied is the different-action mass whose winners on both sides include a goal-blind rule divided by the D universe mass; D_tie_share divides that mass by all different-action mass.',
        intervals='Paired seed differences, sample standard deviation / sqrt(n), marginal 95% t intervals, no multiplicity correction. Undefined seed slots remain null and invalidate the interval; no dropping or imputation.',
        outcome=dict(reference='acs2her_shortened', helps='Primary interval above zero on >=3 mazes and below zero on none.',
            trade_off='Final interval above zero on >=3 mazes and primary below zero on >=1, or primary above zero on >=3 and final below zero on >=1.',
            otherwise='does not help', precedence=['trade-off', 'helps', 'does not help']),
        mechanism_confirmation='On >=3 of maze4, maze5, maze6, maze7, new-shortened F_mod interval is above zero AND estimate_above_v interval is below zero.',
        supplementary=['within-start spread/correlation', 'F-C=(F_mod-C)+(F-F_mod)', 'tie-required share of D',
            'uniform ordered-pair sensitivity pooled before reweighting', 'all-checkpoint F and F_mod figures against both references'],
        labels='Only the success and final pooled mechanism contrasts above and their outcome rules are registered inferential results. Everything else is post hoc.')
