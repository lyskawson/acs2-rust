import copy
import math

from proportions import BLOCKS, MEASURES_BY_ARM, S1, SHARES
from validate import validate_row as validate_measures


def close(actual, expected):
    if (type(actual) not in (int, float) or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError(f'expected {expected}, observed {actual}')


def capped_steps(success, solved_mean_steps, cap):
    return success * (solved_mean_steps or 0.0) + (1 - success) * cap


def validate_test_caps(row, config):
    cap = config['cap']
    long_cap = config['second_test_cap']
    caps = row['test_caps']
    short, long = caps['registered'], caps['extended']
    test = row['evaluation_blocks']['test']
    if (short['cap'], long['cap'], short['success']) != (cap, long_cap, test['success']):
        raise ValueError('test cap or registered success differs')
    if not short['success'] - 1e-12 <= long['success'] <= 1 + 1e-12 or long['heads'] is not None:
        raise ValueError('invalid extended test success')
    close(short['mean_steps'], capped_steps(test['success'], test['mean_success_steps'], cap))
    close(long['mean_shortest_path'], short['mean_shortest_path'])
    if not 0 < short['mean_shortest_path'] or not 0 < long['cumulative_cpu_seconds'] <= row['wall_seconds_eval']:
        raise ValueError('missing shortest path or extended evaluation CPU')
    for block in (short, long):
        if not 1 - 1e-12 <= block['mean_steps'] <= block['cap'] + 1e-12:
            raise ValueError('invalid weighted step count')
        close(block['mean_steps_over_shortest'], block['mean_steps'] / block['mean_shortest_path'])
    if not config.get('per_start_diagnostics'):
        if long['starts'] is not None:
            raise ValueError('per-pair test cap diagnostics were not registered')
        return
    if len(long['starts']) != len(test['starts']):
        raise ValueError('test pairs are missing')
    success, steps, shortest = 0.0, 0.0, 0.0
    for pair, extended in zip(test['starts'], long['starts']):
        solved, length = extended['success'], extended['steps']
        if (type(solved) is not bool or type(length) is not int or not 1 <= length <= long_cap
                or not solved and length != long_cap
                or pair['success'] and (not solved or pair['steps'] != length)
                or not pair['success'] and length <= cap
                or solved and length < pair['shortest_distance']):
            raise ValueError('the extended rollout disagrees with its registered pair')
        weight = pair['weight']
        success += weight * solved
        steps += weight * length
        shortest += weight * pair['shortest_distance']
    close(long['success'], success)
    close(long['mean_steps'], steps)
    close(short['mean_shortest_path'], shortest)


def validate_row(row, config, agent, seed, commit, target):
    if row.get('agent') != agent or agent not in (*MEASURES_BY_ARM, S1):
        raise ValueError('unregistered trajectory utility proportions arm')
    declared = config.get('trajectory_utility', {})
    share = declared.get('tu_share', 0.0 if agent == S1 else 0.5)
    if share in SHARES:
        if agent == S1 or config.get('second_test_cap') != 100 or 'test_caps' not in row:
            raise ValueError('new arms require the registered second test cap')
        if config.get('evaluation_blocks') != BLOCKS or bool(config.get('per_start_diagnostics')) != (42 <= seed <= 46):
            raise ValueError('invalid registered evaluation blocks or per-pair seed')
        if (declared.get('candidates'), declared.get('refresh_steps')) != (8, 2000):
            raise ValueError('unregistered K or F')
        parameters = row['agent_parameters']['trajectory_utility']
        probabilities = dict(original=1 / 3, hindsight=(1 - share) * (2 / 3), trajectory_utility=share * (2 / 3))
        if parameters.get('tu_share') != share or parameters.get('route_probabilities') != probabilities:
            raise ValueError('unregistered requested TU share')
        if share == 1 and row['trajectory_utility']['diagnostics']['draw_requests']['hindsight']:
            raise ValueError('TU share one has a hindsight request')
        validate_test_caps(row, config)
        adapted = copy.deepcopy(row)
        del adapted['test_caps']
        utility = adapted['agent_parameters']['trajectory_utility']
        del utility['tu_share']
        utility['route_probabilities'] = {name: 1 / 3 for name in probabilities}
        validate_measures(adapted, config, agent, seed, commit, target)
    elif share == (0.0 if agent == S1 else 0.5):
        if 'test_caps' in row or 'second_test_cap' in config:
            raise ValueError('archival references only have the registered cap')
        if agent != S1 and (declared.get('candidates'), declared.get('refresh_steps')) != (8, 2000):
            raise ValueError('unregistered archival K or F')
        validate_measures(row, config, agent, seed, commit, target)
    else:
        raise ValueError('unregistered TU share')
