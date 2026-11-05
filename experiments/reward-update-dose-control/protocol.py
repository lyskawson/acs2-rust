import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
sys.path.append(str(ROOT / 'experiments/per-pair-diagnostics'))
from grid_protocol import load_plan

EXPERIMENT = 'reward-update-dose-control'
ARM = 'acs2her_shortened_random_rules'
G = 'acs2her_shortened_goal_rules'
S = 'acs2her_shortened'
E = 'acs2er_episodic'
REFERENCES = (G, S, E)
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
        concurrency=20, phase_cpu_hour_limit=2, pilot_allocation_cpu_seconds=0, accounting_since='2026-11-05')


def analysis_registration():
    comparisons = [[G, ARM], [ARM, S]]
    return dict(success=dict(seeds=list(range(42, 62)), block='test', metrics=['primary', 'final'],
            primary='Equal-weight checkpoint mean within seed, every registered checkpoint.',
            comparisons=comparisons, critical_value=T19, degrees_of_freedom=19),
        mechanism=dict(seeds=list(range(42, 47)), point='final', view='pooled',
            metrics=['F', 'F_mod', 'estimate_minus_q', 'estimate_above_v', 'within_start_slope', 'winner_goal_mass'],
            comparisons=comparisons, critical_value=T4, degrees_of_freedom=4),
        pooling='Concatenate training, validation and test; retain recorded pair weights, giving equal mass to blocks even with unequal block sizes.',
        F='Optimal first-action share among nonfallbacks.',
        F_mod='Optimality of weighted within-start modal action among nonfallbacks; lowest action index breaks modal ties.',
        estimate_minus_q='Estimate minus capped Q* for chosen first action; all pairs including fallbacks.',
        estimate_above_v='Strict binary64 estimate > capped V* among nonfallbacks.',
        within_start_slope='Weighted within-start covariance(estimate,Q*) / within-start variance(Q*); all pairs including fallbacks.',
        winner_goal_mass='Weighted share of ALL first decisions credited to a winner specifying a goal attribute. Tied classifier winners share equally, without numerosity weighting; fallbacks contribute zero.',
        intervals='Pair by seed before computing sample SE. Marginal 95% t(19) success and t(4) mechanism intervals, no multiplicity correction. Undefined seed metrics invalidate the interval; never drop or impute.',
        dose=dict(point='final', seeds=list(range(42,62)),
            aggregation='Equal-weight mean of per-seed cumulative opportunity shares; R minus G paired by seed for the all-opportunity check.',
            relabeled='R skipped_with_goal + skipped_without_goal divided by R relabeled action_set_rules; compare absolute mean difference with registered pi.',
            all='R total skips / R all opportunities minus G total skips / G all opportunities.',
            limits=dict(relabeled_absolute_difference=0.01, all_absolute_difference=0.02),
            violation='Strictly greater than either tolerance is a stated limitation on that maze, never silently discarded.'),
        outcome=dict(mazes=['maze4','maze5','maze6','maze7','mazema'], excluded=['woods14'], measure='primary',
            counts='A+ and A- count G-R intervals strictly above and below zero; B+ counts R-S intervals strictly above zero.',
            cases=[dict(name='rule choice', predicate='A+ >= 3 and A- == 0 and B+ < 3'),
                dict(name='rule choice beyond fewer updates', predicate='A+ >= 3 and A- == 0 and B+ >= 3'),
                dict(name='fewer updates', predicate='A+ < 3 and B+ >= 3'),
                dict(name='unresolved', predicate='otherwise')]),
        mechanism_rule='On at least three of maze4, maze5, maze6, maze7, G-R intervals are strictly positive for BOTH F_mod and winner_goal_mass on the same maze.',
        interpretation='R also removes some copy rewards from goal-blind rules. R-S > 0 alone does not identify update count as the cause. G-R is the deciding contrast.',
        labels='Only these contrasts, dose checks and decision rules are registered. Everything else is post hoc.')
