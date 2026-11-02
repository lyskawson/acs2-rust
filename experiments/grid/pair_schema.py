import ast
import math
import re

ARMS = ('acs2', 'acs2er_episodic', 'acs2her', 'acs2her_nontrivial', 'acs2her_shortened')
MAZES = ('maze4', 'maze5', 'maze6', 'maze7', 'woods14', 'mazema')
PAIR_FIELDS = {'start', 'goal', 'weight', 'first_action', 'estimated_first_action_value',
               'discounted_return', 'success', 'steps', 'shortest_distance', 'first_decision'}
WINNER_FIELDS = {'specified_state_attributes', 'specified_goal_attributes', 'q', 'r', 'fitness',
                 'experience', 'numerosity', 'selection_score'}


def cell(text):
    value = ast.literal_eval(text)
    if type(value) is not tuple or len(value) != 2 or any(type(part) is not int or part < 0 for part in value):
        raise ValueError('invalid start cell')
    return value


def goal_cell(text):
    match = re.fullmatch(r'Goal \{ symbols: \[Token\((\d+)\), Token\((\d+)\)\] \}', text)
    if match is None:
        raise ValueError('invalid coordinate goal')
    return tuple(map(int, match.groups()))


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def enabled(config):
    if 'per_start_diagnostics' in config and config['per_start_diagnostics'] is not True:
        raise ValueError('per-pair recording must be explicitly true or absent')
    if config.get('per_start_diagnostics') and (
            config.get('protocol') != 'maze-pairs-v1' or config.get('encoding') != 'coordinates'
            or config.get('pool') != 'full' or config.get('row_schema') != 5
            or config.get('task') not in MAZES or config.get('evaluation_blocks') != ['training', 'validation', 'test']
            or config.get('agent_settings', {})):
        raise ValueError('per-pair recording is restricted to the unchanged registered diagnostic study')
    return config.get('per_start_diagnostics', False)


def validate_pairs(row, config, name, block):
    if (config.get('protocol') != 'maze-pairs-v1' or config.get('encoding') != 'coordinates'
            or config.get('pool') != 'full' or config.get('row_schema') != 5
            or config['task'] not in MAZES or row['agent'] not in ARMS
            or config.get('evaluation_blocks') != ['training', 'validation', 'test']
            or config.get('agent_settings', {}).get(row['agent'], {})):
        raise ValueError('per-pair records require the five unchanged registered diagnostic presets')
    records = block['starts']
    registered = config['registration']['blocks'][name]
    expected = {(tuple(start), tuple(entry['goal'])) for entry in registered['goal_starts'] for start in entry['starts']}
    if type(records) is not list or len(records) != len(expected):
        raise ValueError('missing or extra per-pair records')
    actual = set()
    for record in records:
        if set(record) != PAIR_FIELDS:
            raise ValueError('invalid per-pair record fields')
        key = cell(record['start']), goal_cell(record['goal'])
        if key in actual or key not in expected:
            raise ValueError('duplicate or unregistered evaluation pair')
        actual.add(key)
        if type(record['weight']) is not float or record['weight'] != 1.0 / len(expected):
            raise ValueError('pair weight differs from exhaustive registered weights')
        if (type(record['first_action']) is not int or not 0 <= record['first_action'] < 8
                or type(record['success']) is not bool or type(record['steps']) is not int
                or not 1 <= record['steps'] <= config['cap']
                or type(record['shortest_distance']) is not int or record['shortest_distance'] <= 0
                or any(not finite(record[key]) for key in ('estimated_first_action_value', 'discounted_return'))):
            raise ValueError('invalid per-pair outcome')
        decision = record['first_decision']
        if (set(decision) != {'random_fallback', 'winners'} or type(decision['random_fallback']) is not bool
                or type(decision['winners']) is not list or bool(decision['winners']) == decision['random_fallback']):
            raise ValueError('invalid first-decision fallback or winners')
        for winner in decision['winners']:
            if set(winner) != WINNER_FIELDS:
                raise ValueError('invalid winner profile fields')
            for key, low, high in [('specified_state_attributes', 0, 8), ('specified_goal_attributes', 0, 2),
                                   ('experience', 0, 2**32 - 1), ('numerosity', 1, 2**32 - 1)]:
                if type(winner[key]) is not int or not low <= winner[key] <= high:
                    raise ValueError('invalid winner count')
            if (any(not finite(winner[key]) for key in ('q', 'r', 'fitness', 'selection_score'))
                    or not 0 <= winner['q'] <= 1 or winner['r'] < 0
                    or winner['fitness'] != winner['q'] * winner['r']
                    or winner['selection_score'] != winner['fitness'] * winner['numerosity']):
                raise ValueError('invalid winner score')
        if len({winner['selection_score'] for winner in decision['winners']}) > 1:
            raise ValueError('winner list contains a non-tied classifier')
