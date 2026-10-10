import math

EVALUATION_FIELDS = ('success', 'random_floor', 'reachable_within_cap', 'reachable_after_cap',
                     'unreachable_or_ambiguous', 'evaluation_pairs', 'evaluation_distribution',
                     'evaluation_standard_error', 'evaluation_sample_seed', 'evaluation_sample_stream',
                     'mean_success_steps', 'mean_success_steps_over_shortest', 'value_diagnostics',
                     'starts', 'reference_distribution')


def evaluation_blocks(row):
    if row.get('schema') not in (2, 3, 4, 5):
        raise ValueError('unsupported measurement row schema')
    blocks = row.get('evaluation_blocks')
    if blocks is None:
        return {'full_pool': {key: row[key] for key in EVALUATION_FIELDS}}
    if row['schema'] not in (4, 5) or set(blocks) != {'training', 'validation', 'test'}:
        raise ValueError('expected three separate evaluation blocks')
    if any(key in row for key in EVALUATION_FIELDS):
        raise ValueError('split row contains ambiguous top-level evaluation fields')
    if any(set(block) != set(EVALUATION_FIELDS) for block in blocks.values()):
        raise ValueError('incomplete evaluation block')
    return blocks


def validate_blocks(row, config):
    blocks = evaluation_blocks(row)
    if row['schema'] == 5 and config.get('protocol') == 'maze-pairs-v1':
        knowledge = row.get('knowledge')
        if not isinstance(knowledge, (int, float)) or not math.isfinite(knowledge) or not 0 <= knowledge <= 1:
            raise ValueError('pair protocol requires finite knowledge in every row')
        if row.get('knowledge_transition_goal_pairs') != config['knowledge_transition_goal_pairs']:
            raise ValueError('knowledge uses a different transition-goal universe')
    if 'protocol' not in config:
        if set(blocks) != {'full_pool'}:
            raise ValueError('legacy configuration requires a full-pool row')
        return
    expected = {'id': config['protocol'], 'configuration': config['registration']}
    if row.get('goal_protocol') != expected:
        raise ValueError('goal protocol registration differs')
    for name, block in blocks.items():
        reference = config['registration']['blocks'][name]
        if block['evaluation_pairs'] != reference['evaluation_pairs']:
            raise ValueError('evaluation pair count differs from the registered block')
        if block['evaluation_distribution'] != 'exhaustive_exact_weights' or block['evaluation_standard_error'] != 0:
            raise ValueError('maze evaluation must use all pairs with exact weights')
        if block['starts'] is not None or block['evaluation_sample_seed'] is not None or block['evaluation_sample_stream'] is not None:
            raise ValueError('grid rows must be compact and exhaustive')
        for key in ('random_floor', 'reachable_within_cap'):
            if not math.isclose(block[key], reference[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'{name}: {key} differs from its own reference')
        if not math.isclose(block['reachable_after_cap'], 1 - reference['reachable_within_cap'], abs_tol=1e-12) or block['unreachable_or_ambiguous'] != 0:
            raise ValueError('incorrect reachability decomposition')
        if not math.isfinite(block['success']) or not 0 <= block['success'] <= block['reachable_within_cap'] + 1e-12:
            raise ValueError('success is outside its block ceiling')
