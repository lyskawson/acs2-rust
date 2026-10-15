from pilot_equivalence import compare, CONTROL_CHANGED_FIELDS


def compare_rows(current, reference, seed, targets):
    if seed != 42 or not current:
        raise ValueError('VCP pilot identity requires seed 42 and nonempty current rows')
    first = current[0]
    identity = first['task'], first['agent'], first['agent_parameters']['temperature']
    def indexed(rows):
        result = {}
        for row in rows:
            if (row['task'], row['agent'], row['agent_parameters']['temperature']) != identity or row['seed'] != seed:
                continue
            point = row['nominal_step']
            if point in result:
                raise ValueError('duplicate VCP pilot identity checkpoint')
            result[point] = row
        if set(result) != set(targets):
            raise ValueError('missing VCP pilot identity checkpoint')
        return result
    actual, pinned = indexed(current), indexed(reference)
    for point in targets:
        compare(pinned[point], actual[point])
    return dict(rows=len(targets), differences=0, blocks=['validation'],
                floating_comparison='IEEE 754 binary64 bits',
                control_changed_fields=CONTROL_CHANGED_FIELDS if identity[2] == 0 else [],
                additional_diagnostics='checked separately by the revised VCP row schema')
