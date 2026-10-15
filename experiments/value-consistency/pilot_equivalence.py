import copy
import struct

CONTROL_CHANGED_FIELDS = [
    'match_formations_train', 'classifier_perception_tests_train',
    'replay_diagnostics.priority.value_queries', 'replay_diagnostics.priority.proposal_variance',
    'replay_diagnostics.priority.accepted_variance', 'replay_diagnostics.priority.candidate_acceptance_sums',
]
NONLEARNING = {'commit', 'source_state', 'host', 'cpu_model', 'wall_seconds_train',
               'wall_seconds_eval', 'wall_seconds_total'}


def binary64(value):
    if isinstance(value, float):
        return 'f64:' + struct.pack('>d', value).hex()
    if isinstance(value, list):
        return list(map(binary64, value))
    if isinstance(value, dict):
        return {key: binary64(item) for key, item in value.items()}
    return value


def compare(pilot, revised):
    expected, actual = copy.deepcopy(pilot), copy.deepcopy(revised)
    priority = actual['replay_diagnostics']['priority']
    temperature = actual['agent_parameters']['temperature']
    heads = actual['agent_parameters']['heads']
    if priority.pop('training_variance_measured') is not (temperature > 0):
        raise ValueError('training variance measurement differs')
    if temperature == 0:
        queries = priority.pop('avoided_value_queries')
        tests = priority.pop('avoided_classifier_perception_tests')
        if (queries != expected['replay_diagnostics']['priority']['value_queries']
                or queries != heads * priority['proposals'] or tests <= 0
                or priority['value_queries'] != 0):
            raise ValueError('control removed-query accounting differs')
        expected['match_formations_train'] -= queries
        expected['classifier_perception_tests_train'] -= tests
        expected['replay_diagnostics']['priority']['value_queries'] = 0
        counts = actual['learning_update_counts']
        if actual['match_formations_train'] != actual['actual_steps'] + 3 * counts['full'] + 2 * counts['values_only']:
            raise ValueError('query-free control match identity differs')
        for field in ['proposal_variance', 'accepted_variance', 'candidate_acceptance_sums']:
            if priority[field] is not None:
                raise ValueError('control reports unmeasured proposal variance')
            expected['replay_diagnostics']['priority'][field] = None
    else:
        for field in ['proposal_variance', 'accepted_variance']:
            if priority[field].pop('logarithmic')['count'] != priority[field]['count']:
                raise ValueError('added logarithmic summary count differs')
    for block in actual['evaluation_blocks'].values():
        diagnostic = block.pop('head_disagreement')
        if diagnostic['value_queries'] != heads * diagnostic['count']:
            raise ValueError('evaluation disagreement queries differ')
    actual['evaluation_blocks'] = {'validation': actual['evaluation_blocks']['validation']}
    ensemble = actual['ensemble']
    extra = ensemble.pop('measurement_diagnostics_inline_bytes')
    if ensemble.pop('agent_and_measurement_known_bytes_lower_bound') != ensemble['agent_known_bytes_lower_bound'] + extra:
        raise ValueError('added diagnostic memory differs')
    header = ensemble['heads'][0]['head_known_bytes_lower_bound'] - ensemble['heads'][0]['population_known_bytes_lower_bound']
    agent_header = ensemble['agent_known_bytes_lower_bound'] - ensemble['heads_known_bytes_lower_bound'] - actual['trajectory_logical_bytes']
    if (header, agent_header) not in [(376, 1832), (384, 1856)]:
        raise ValueError('unmeasured target-native inline headers')
    total = 0
    for head in expected['ensemble']['heads']:
        head['head_known_bytes_lower_bound'] = head['population_known_bytes_lower_bound'] + header
        total += head['head_known_bytes_lower_bound']
    expected['ensemble']['heads_known_bytes_lower_bound'] = total
    expected['ensemble']['agent_known_bytes_lower_bound'] = total + expected['trajectory_logical_bytes'] + agent_header
    expected = {key: value for key, value in expected.items() if key not in NONLEARNING}
    actual = {key: value for key, value in actual.items() if key not in NONLEARNING}
    if binary64(expected) != binary64(actual):
        differences = [key for key in expected.keys() | actual.keys() if binary64(expected.get(key)) != binary64(actual.get(key))]
        raise ValueError('pilot learning, evaluation or knowledge fields differ: ' + ', '.join(sorted(differences)))
    return CONTROL_CHANGED_FIELDS if temperature == 0 else []
