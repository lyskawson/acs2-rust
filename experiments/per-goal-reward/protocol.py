import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
sys.path.append(str(ROOT / 'experiments/per-pair-diagnostics'))
from grid_protocol import load_plan

EXPERIMENT = 'per-goal-reward'
E1 = 'acs2er_episodic_per_goal_reward'
S1 = 'acs2her_shortened_per_goal_reward'
E = 'acs2er_episodic'
S = 'acs2her_shortened'
G = 'acs2her_shortened_goal_rules'
R = 'acs2her_shortened_random_rules'
NEW_ARMS = (E1, S1)
REFERENCES = (E, S, G, R)
ARMS = (*NEW_ARMS, E, S, G)
BLOCKS = ['training', 'validation', 'test']
MAZES = ('maze4', 'maze5', 'maze6', 'maze7', 'woods14', 'mazema')
MECHANISM_MAZES = ('maze4', 'maze5', 'maze6', 'maze7')
SUCCESS_CONTRASTS = ('S+1 - S', 'E+1 - E', 'interaction', 'S+1 - G')
MECHANISM_CONTRASTS = ('S+1 - S', 'E+1 - E')
MECHANISM_METRICS = ('F', 'F_mod', 'estimate_minus_q', 'estimate_above_v',
                     'within_start_slope', 'within_start_sd_estimate')
T4 = 2.7764451051977987
T19 = 2.093024054408263
ADMISSIBILITY_SHA = 'f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da'


def configurations():
    return copy.deepcopy(load_plan(ROOT / 'experiments/hindsight-admissibility/plan.json',
                                   ADMISSIBILITY_SHA)['configurations'])


def recorded(config, limit=20, per_pair=True):
    result = dict(copy.deepcopy(config), evaluation_blocks=BLOCKS, agent_settings={},
                  limits_minutes={arm: limit for arm in NEW_ARMS})
    if per_pair:
        result['per_start_diagnostics'] = True
    return result


def pilot(identity):
    return dict(stage='authorized_cost_pilot', experiment=EXPERIMENT, **identity,
        agents=list(NEW_ARMS), seeds=[42], configurations=[recorded(config) for config in configurations()],
        partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=2048,
        concurrency=20, phase_cpu_hour_limit=4, pilot_allocation_cpu_seconds=0,
        accounting_since='2026-11-07')


def analysis_registration():
    return dict(success=dict(seeds=list(range(42, 62)), block='test', metrics=['primary', 'final'],
            primary='Equal-weight mean of all registered checkpoint success values within each seed.',
            comparisons=list(SUCCESS_CONTRASTS), interaction='(S+1-S)-(E+1-E), calculated within each seed before the interval.',
            critical_value=T19, degrees_of_freedom=19),
        mechanism=dict(seeds=list(range(42, 47)), point='final', view='pooled',
            comparisons=list(MECHANISM_CONTRASTS), metrics=list(MECHANISM_METRICS),
            critical_value=T4, degrees_of_freedom=4),
        pooling='Concatenate training, validation and test, retain their recorded pair weights and normalize by total mass. Blocks have equal mass even with unequal sizes.',
        F='Weighted optimal first-action share among nonfallback decisions.',
        F_mod='Weighted optimality of the within-start modal action among nonfallback decisions; lowest action index breaks modal ties.',
        estimate_minus_q='Mean first-action estimate minus capped Q* for the chosen action, including fallbacks.',
        estimate_above_v='Weighted share with binary64 estimate strictly greater than capped V*, among nonfallback decisions.',
        within_start_slope='Weighted within-start covariance(estimate,Q*) divided by weighted within-start variance(Q*), including fallbacks.',
        within_start_sd_estimate='Square root of weighted within-start estimate variance divided by total pair weight, including fallbacks.',
        intervals='Pair all arms by seed first. Marginal 95% t(19) and t(4), no multiplicity correction. Undefined seed metrics invalidate the interval; never drop or impute.',
        verdicts=dict(mazes=list(MAZES), measures=['primary', 'final'], separate_for_each_contrast=True,
            signs='A+ counts ci_low > 0; A- counts ci_high < 0. Touching zero counts neither.',
            effects=[dict(name='improves', predicate='A+ >= 3 and A- == 0'),
                dict(name='worsens', predicate='A- >= 3 and A+ == 0'),
                dict(name='mixed', predicate='A+ >= 1 and A- >= 1'),
                dict(name='no clear effect', predicate='otherwise')],
            interaction=[dict(name='copies gain more from per-goal values', predicate='A+ >= 3 and A- == 0'),
                dict(name='copies gain less from per-goal values', predicate='A- >= 3 and A+ == 0'),
                dict(name='mixed', predicate='A+ >= 1 and A- >= 1'),
                dict(name='no clear difference', predicate='otherwise')]),
        mechanism_rule='Separately for S+1-S and E+1-E, at least three of Maze4--7 have strictly positive intervals for BOTH within_start_slope and F_mod on the same maze.',
        post_hoc='Every other comparison, including every comparison with ACS2 or R, is post hoc.')
