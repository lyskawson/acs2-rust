import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import load_plan
from pair_schema import ARMS, MAZES

EXPERIMENT = 'per-pair-diagnostics'
BLOCKS = ['training', 'validation', 'test']
BASELINE_SHA = '913ce814484b742265ba88db09d72d4feaf8f981b200811454f9fce11b6b5a78'
ADMISSIBILITY_SHA = 'f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da'
T4 = 2.7764451051977987


def configurations():
    path = ROOT / 'experiments/hindsight-admissibility/plan.json'
    return copy.deepcopy(load_plan(path, ADMISSIBILITY_SHA)['configurations'])


def recorded(config, limit=8):
    return dict(copy.deepcopy(config), per_start_diagnostics=True, evaluation_blocks=BLOCKS,
                agent_settings={}, limits_minutes={arm: limit for arm in ARMS})


def pilot(identity):
    return dict(stage='authorized_cost_pilot', experiment=EXPERIMENT, **identity,
                agents=list(ARMS), seeds=[42], configurations=[recorded(config) for config in configurations()],
                partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=2048,
                concurrency=20, phase_cpu_hour_limit=4, pilot_allocation_cpu_seconds=0,
                accounting_since='2026-11-02')


def analysis_registration():
    return dict(seeds=list(range(42, 47)), primary_blocks=BLOCKS,
        pooling='Concatenate blocks, retain recorded pair weights, normalize by their total (equal block mass). Same full pair universe for every arm.',
        fallback='Exclude random first-decision fallbacks from F, M and both D rates; retain them in C, random reference, value errors and fallback rate. Also export eligible-set C and random reference post hoc to reveal selection bias.',
        ceiling='Within each start take the maximum weighted optimal-action coverage, then sum over starts and divide by total pair weight.',
        discrimination='Unordered distinct goals at the same start, disjoint optimal-action sets, product of the two pair weights; pool numerator and denominator across starts. Export different-action and both-optimal rates; D in registered contrasts means different-action rate.',
        winner='For the chosen action retain all matching change-anticipating classifiers tied at the global maximum of fitness*numerosity. Equal fractional credit per tied macroclassifier, the conditional expectation under the existing uniform shuffle; fallbacks have no winner. Report conditional on a nonfallback decision, plus unconditional mass, and solved/failed denominators.',
        points=[30000, 100000, 'final'], per_arm=['F - C'],
        final_policy_contrasts=[['acs2', 'acs2er_episodic'], ['acs2her', 'acs2er_episodic'],
            ['acs2her_shortened', 'acs2er_episodic'], ['acs2her_shortened', 'acs2her'], ['acs2her_nontrivial', 'acs2her']],
        final_policy_metrics=['F', 'M', 'D'],
        final_value_contrasts=[['acs2her_nontrivial', 'acs2her'], ['acs2her_shortened', 'acs2her'], ['acs2her', 'acs2er_episodic']],
        value_metric='mean estimate - Q* on all pairs at the final point, same fixed pair set and weights for every arm',
        final_winner_profile=list(ARMS), paired_interval=dict(n=5, degrees_of_freedom=4,
            critical_value=T4, coverage=0.95, multiplicity_correction='none; marginal intervals'),
        missing='Undefined rates remain null with their denominators. Never drop seeds or impute; an inferential contrast with any undefined seed is undefined with the available count.',
        post_hoc=['all-checkpoint F/M/D/winner curves', 'failed pairs', 'exact shortest-distance groups',
                  'per-block views', 'eligible-set ceilings', 'both-optimal D contrasts', 'consistency over seeds'],
        consistency='For each fixed pair at the final point count solved seeds 0..5. Descriptive plug-in Binomial(5,p), p=k/5 per pair; sum expected bin masses with pair weights. This reuses the same five outcomes to estimate p and is not an independent goodness-of-fit test.')
