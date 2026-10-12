import struct

OMIT = {'commit', 'source_state', 'host', 'cpu_model', 'wall_seconds_train',
        'wall_seconds_eval', 'wall_seconds_total', 'arm'}


def field_bits(value):
    if type(value) is float:
        return 'f64:' + struct.pack('>d', value).hex()
    if isinstance(value, dict):
        return {key: field_bits(item) for key, item in value.items()}
    if isinstance(value, list):
        return [field_bits(item) for item in value]
    return value


def fields(row, blocks):
    result = {key: value for key, value in row.items() if key not in OMIT}
    result['evaluation_blocks'] = {name: row['evaluation_blocks'][name] for name in blocks}
    return field_bits(result)


def compare(current, reference, seeds, targets, blocks):
    expected = {(seed, point) for seed in seeds for point in targets}
    def index(rows):
        indexed = {}
        for row in rows:
            key = row['seed'], row['nominal_step']
            if key not in expected:
                continue
            if key in indexed:
                raise ValueError('duplicate identity row')
            indexed[key] = fields(row, blocks)
        if set(indexed) != expected:
            raise ValueError('identity check is missing a seed or a shared point')
        return indexed
    actual, pinned = index(current), index(reference)
    for key in sorted(expected):
        if actual[key] != pinned[key]:
            differing = sorted(name for name in set(actual[key]) | set(pinned[key])
                               if actual[key].get(name) != pinned[key].get(name))
            raise ValueError(f'learning, evaluation or knowledge identity changed at {key}: {differing}')
    return {'rows': len(expected), 'differences': 0, 'blocks': list(blocks),
            'excluded': sorted(OMIT), 'floating_comparison': 'IEEE 754 binary64 bits'}
