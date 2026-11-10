import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.append(str(ROOT / 'experiments/grid'))
A = 'acs2'
A1 = 'acs2_per_goal_reward'
E = 'acs2er_episodic'
S = 'acs2her_shortened'
E1 = 'acs2er_episodic_per_goal_reward'
S1 = 'acs2her_shortened_per_goal_reward'
ARMS = (A, A1, E, S, E1, S1)
BASES = {A1: A, E1: E, S1: S}
OLD_TASKS = ('maze4', 'maze5', 'maze6', 'maze7', 'woods14', 'mazema')
BLOCKS = ['training', 'validation', 'test']
SEEDS = list(range(42, 62))
PER_PAIR_SEEDS = list(range(42, 47))
T19 = 2.093024054408263
CONTRASTS = {'ACS2 r(g) - ACS2': {A1: 1, A: -1}, 'E+1 - E': {E1: 1, E: -1},
    'S+1 - S': {S1: 1, S: -1}, 'E+1 - ACS2 r(g)': {E1: 1, A1: -1},
    'S+1 - E+1': {S1: 1, E1: -1},
    'memory x values': {E1: 1, E: -1, A1: -1, A: 1},
    'copies x values': {S1: 1, S: -1, E1: -1, E: 1}}
CEILING_ARMS = {name: tuple(weights) for name, weights in CONTRASTS.items()}
CEILING_ARMS.update({'memory x values': (E1, A1), 'copies x values': (S1, E1)})


def configurations():
    old = json.loads((ROOT / 'experiments/per-goal-reward/plan.json').read_text())['configurations']
    geometry = {e['task']: e for e in json.loads((HERE / 'geometry.json').read_text())}
    result = copy.deepcopy(old)
    for config in result:
        config['group'] = 'no_aliasing'
        config['aliased_share'] = 0.0
    for entry in json.loads((HERE / 'pair-protocol.json').read_text())['configurations']:
        g = geometry[entry['task']]
        result.append(dict(id=entry['id'], task=entry['task'], protocol='alcs-maze-pairs-v1', registration=entry,
            cap=entry['cap'], pool='full', encoding='coordinates', row_pool='full',
            row_schema=5, targets=entry['checkpoints'], evaluation_blocks=BLOCKS,
            knowledge_transition_goal_pairs=(len(g['walkable_cells']) - 1) * sum(sum(j != i for j in row) for i, row in enumerate(g['successors'])),
            group=entry['group'], aliased_share=entry['aliased_share']))
    for config in result:
        config['evaluation_blocks'] = BLOCKS
        config['agent_settings'] = {}
        config.pop('per_start_diagnostics', None)
        config.pop('limits_minutes', None)
    return result


def new_cells():
    return [(config, arm) for config in configurations() for arm in ((A1,) if config['task'] in OLD_TASKS else ARMS)]


def recorded(config, seed):
    result = copy.deepcopy(config)
    if seed in PER_PAIR_SEEDS:
        result['per_start_diagnostics'] = True
    return result


def analysis_registration():
    return dict(block='test', seeds=SEEDS, measures=['primary', 'final'],
        primary='Equally weighted mean of every registered checkpoint success within each seed.',
        contrasts=CONTRASTS, interactions='Calculate (E+1-E)-(ACS2 r(g)-ACS2) and (S+1-S)-(E+1-E) within each seed before the marginal 95% t(19) interval.',
        critical_value=T19, degrees_of_freedom=19,
        labels=dict(floor='No arm lower 95% final-mean bound exceeds the exact test floor.',
            too_easy='ACS2 across-seed mean at 1k >= 0.9 * test ceiling.',
            ceiling='Both compared final means >= 0.99. For interactions use their two per-goal arms.'),
        exclusions='Per group and contrast: floor always, too easy only for primary, ceiling only for final. Labels never remove table rows.',
        verdict='n counted mazes; A+ counts CI low>0, A- counts CI high<0. If n=0 not assessed; improves/worsens if its sign count >= ceil(n/2) and opposite=0; mixed if both signs occur; otherwise no clear effect/difference.',
        interaction_words={'memory x values': ['values gain more with memory', 'values gain less with memory'],
                           'copies x values': ['copies gain more from values', 'copies gain less from values']},
        all_mazes='Report A+, A- and n over all 28 without a verdict.',
        sustained_095='First registered checkpoint in environment steps whose across-seed mean and all later checkpoint means are >=0.95; no interpolation; otherwise not reached.',
        reference_check='E+1-E, S+1-S and copies x values on the original six reproduce the earlier registered intervals, without replacing earlier verdicts.',
        secondary=['knowledge','rules','per-goal entries','CPU time','peak memory'],
        post_hoc='Every other comparison, per-pair mechanism analysis, success by distance, G and R comparisons.')
