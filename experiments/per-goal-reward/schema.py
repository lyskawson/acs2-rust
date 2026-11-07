import copy
import json

from protocol import ROOT, E, E1, S, S1, NEW_ARMS, BLOCKS
from collect import validate_row as validate_base_row

BASES = {E1: E, S1: S}
VALUE_PARAMETERS = {'reward_values', 'reward_goal_count', 'reward_initialization',
                    'genetic_algorithm', 'restore_supported'}


def validate_row(row, config, agent, seed, commit, target):
    if agent not in NEW_ARMS or row.get('agent') != agent:
        raise ValueError('per-goal reward validation accepts exactly the two registered arms')
    if (config.get('protocol') != 'maze-pairs-v1' or config.get('pool') != 'full'
            or config.get('encoding') != 'coordinates' or config.get('evaluation_blocks') != BLOCKS
            or config.get('agent_settings', {})):
        raise ValueError('per-goal reward arms require the unchanged registered pair protocol and fixed settings')
    preset = json.loads((ROOT / 'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset']
    if row.get('preset') != preset or row['preset'].get('do_ga') is not False:
        raise ValueError('per-goal rewards require the thesis preset with GA disabled')
    goals = config['registration']['walkable_cells']
    parameters = row['agent_parameters']
    expected = dict(reward_values='per_classifier_per_goal', reward_goal_count=goals,
        reward_initialization='base_creation_values_alp_parent_inheritance',
        genetic_algorithm=False, restore_supported=False)
    if any(type(parameters.get(key)) is not type(value) or parameters[key] != value for key, value in expected.items()):
        raise ValueError('per-goal reward initialization, goal set, GA or restore contract differs')
    counts = row['replay_diagnostics'].get('per_goal_rewards', {})
    if (set(counts) != {'goals', 'existing_pairs', 'updated_entries', 'reward_updates'}
            or any(type(counts[key]) is not int or counts[key] < 0 for key in ('goals', 'existing_pairs', 'updated_entries'))
            or counts['goals'] != goals or counts['existing_pairs'] != goals * row['population_classifiers']
            or not 0 <= counts['updated_entries'] <= counts['existing_pairs']):
        raise ValueError('invalid live classifier-goal entry counts')
    updates = counts['reward_updates']
    if set(updates) != {'original', 'relabeled'} or any(type(value) is not int or value < 0 for value in updates.values()):
        raise ValueError('invalid per-goal reward update routes')
    if counts['updated_entries'] > sum(updates.values()):
        raise ValueError('live updated entries exceed cumulative reward writes')
    for origin in ('original', 'relabeled'):
        if row['replay_diagnostics'][origin] == 0 and updates[origin] != 0:
            raise ValueError('reward writes recorded for an absent draw origin')
    if agent == E1 and updates['relabeled']:
        raise ValueError('episodic ER cannot write hindsight values')
    adapted = copy.deepcopy(row)
    adapted['agent'] = BASES[agent]
    adapted['agent_parameters'] = {key: value for key, value in parameters.items() if key not in VALUE_PARAMETERS}
    del adapted['replay_diagnostics']['per_goal_rewards']
    validate_base_row(adapted, config, BASES[agent], seed, commit, target)


def validate_sequence(rows, config, agent, seed, commit):
    if [row['nominal_step'] for row in rows] != config['targets']:
        raise ValueError('per-goal reward run must have every registered checkpoint in order')
    previous = dict(original=0, relabeled=0)
    for row, target in zip(rows, config['targets']):
        validate_row(row, config, agent, seed, commit, target)
        updates = row['replay_diagnostics']['per_goal_rewards']['reward_updates']
        if any(updates[origin] < previous[origin] for origin in previous):
            raise ValueError('cumulative reward update count decreased')
        previous = updates
