import copy
import json
import math
from protocol import ROOT, ARMS, BASES, A, A1, E1, S1, BLOCKS
from collect import validate_row as base_validate
from pair_schema import validate_pairs

VALUE_KEYS = {'reward_values','reward_goal_count','reward_initialization','genetic_algorithm','restore_supported'}
PRESET = json.loads((ROOT / 'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset']


def per_pair_enabled(config):
    if 'per_start_diagnostics' in config and config['per_start_diagnostics'] is not True:
        raise ValueError('per-pair flag must be true or absent')
    return config.get('per_start_diagnostics', False)


def validate_row(row, config, arm, seed, commit, target):
    if arm not in ARMS or row.get('agent') != arm:
        raise ValueError('ablation accepts exactly six registered arms')
    if (config.get('protocol') not in ('maze-pairs-v1','alcs-maze-pairs-v1')
            or config.get('pool') != 'full' or config.get('encoding') != 'coordinates'
            or config.get('evaluation_blocks') != BLOCKS or config.get('agent_settings', {})):
        raise ValueError('unregistered protocol or arm settings')
    if row.get('preset') != PRESET or row['preset'].get('do_ga') is not False:
        raise ValueError('fixed thesis preset requires GA disabled')
    knowledge = row.get('knowledge')
    if (type(knowledge) not in (int,float) or not math.isfinite(knowledge) or not 0 <= knowledge <= 1
            or row.get('knowledge_transition_goal_pairs') != config['knowledge_transition_goal_pairs']):
        raise ValueError('knowledge universe differs')
    adapted = copy.deepcopy(row)
    base = BASES.get(arm,arm)
    adapted['agent'] = base
    adjusted = copy.deepcopy(config)
    adjusted.pop('per_start_diagnostics',None)
    if arm in BASES:
        goals = config['registration']['walkable_cells']
        parameters = row['agent_parameters']
        expected = dict(reward_values='per_classifier_per_goal', reward_goal_count=goals,
            reward_initialization='base_creation_values_alp_parent_inheritance', genetic_algorithm=False, restore_supported=False)
        if any(type(parameters.get(k)) is not type(v) or parameters[k] != v for k,v in expected.items()):
            raise ValueError('per-goal value contract differs')
        counts = row.get('replay_diagnostics',{}).get('per_goal_rewards',{})
        if (set(counts) != {'goals','existing_pairs','updated_entries','reward_updates'}
                or any(type(counts[k]) is not int or counts[k]<0 for k in ('goals','existing_pairs','updated_entries'))
                or counts['goals'] != goals or counts['existing_pairs'] != goals * row['population_classifiers']
                or counts['updated_entries'] > counts['existing_pairs']):
            raise ValueError('invalid live goal entries')
        updates=counts['reward_updates']
        if (set(updates) != {'original','relabeled'} or any(type(v) is not int or v<0 for v in updates.values())
                or counts['updated_entries'] > sum(updates.values()) or (arm in (A1,E1) and updates['relabeled'])):
            raise ValueError('invalid value-write counters')
        adapted['agent_parameters']={k:v for k,v in parameters.items() if k not in VALUE_KEYS}
        del adapted['replay_diagnostics']['per_goal_rewards']
        if arm == A1:
            if adapted['replay_diagnostics']:
                raise ValueError('online ACS2 cannot retain replay diagnostics')
            adapted['replay_diagnostics']=None
        else:
            for origin in ('original','relabeled'):
                if row['replay_diagnostics'][origin] == 0 and updates[origin]:
                    raise ValueError('value writes without a sampled origin')
    for name in BLOCKS:
        block=adapted['evaluation_blocks'][name]
        if per_pair_enabled(config):
            pair_config=dict(config,task='maze4',protocol='maze-pairs-v1')
            validate_pairs(adapted,pair_config,name,block)
        elif block['starts'] is not None:
            raise ValueError('compact run contains pair records')
        block['starts']=None
    base_validate(adapted,adjusted,base,seed,commit,target)


def validate_sequence(rows, config, arm, seed, commit):
    if [r['nominal_step'] for r in rows] != config['targets']:
        raise ValueError('missing, duplicate or unordered checkpoints')
    previous={'original':0,'relabeled':0}
    for row,target in zip(rows,config['targets']):
        validate_row(row,config,arm,seed,commit,target)
        if arm in BASES:
            updates=row['replay_diagnostics']['per_goal_rewards']['reward_updates']
            if any(updates[k]<previous[k] for k in previous):
                raise ValueError('cumulative value writes decreased')
            previous=updates
