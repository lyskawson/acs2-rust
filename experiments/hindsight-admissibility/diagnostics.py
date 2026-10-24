import math


def value_groups(block):
    pairs, success = block['evaluation_pairs'], block['success']
    if (type(pairs) is not int or pairs <= 0 or not math.isfinite(success)
            or block.get('evaluation_distribution') != 'exhaustive_exact_weights'):
        raise ValueError('value groups require a positive exhaustive pair count')
    count = round(success * pairs)
    if not 0 <= count <= pairs or abs(success * pairs - count) > 1e-8:
        raise ValueError('value groups require integer exhaustive successes')
    values = block['value_diagnostics']
    if count == pairs:
        failed = None
    elif count == 0:
        failed = dict(values['all'])
    else:
        if values['successful'] is None:
            raise ValueError('successful pairs have no conditional value diagnostic')
        mass = count / pairs
        failed = {key: (value - mass * values['successful'][key]) / (1 - mass)
                  for key, value in values['all'].items()}
    return dict(all=values['all'], successful=values['successful'] if count else None, failed=failed,
                successful_pairs=count, failed_pairs=pairs - count,
                failed_method='Algebraic decomposition of archived all/successful conditional means with exact exhaustive pair counts; binary64 rounding remains.')
