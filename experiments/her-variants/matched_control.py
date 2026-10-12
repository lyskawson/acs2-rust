import copy
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid'))
from grid_protocol import require_budget
from protocol import MATCHED_CONTROL, VALUES


def configurations(plan, selections):
    expected = {config['id'] for config in plan['configurations']}
    if set(selections) != expected:
        raise ValueError('matched control requires the selections for every maze')
    result = []
    for config in plan['configurations']:
        selected = selections[config['id']]
        if set(selected) != set(VALUES) or any(value not in plan['tuning']['proportions'] for value in selected.values()):
            raise ValueError('invalid values-agent selections')
        if selected['her_values'] == 2 / 3:
            continue
        control = copy.deepcopy(config)
        control['row_arm'] = MATCHED_CONTROL
        control['agent_settings'] = {'her_values': {'relabeled_proportion': 2 / 3}}
        result.append(control)
    return result


def reservation_seconds(configs):
    return sum(20 * config['limits_minutes']['her_values'] * 60 for config in configs)


def admit(pilot_seconds, states, configs, limit_hours=60):
    if not math.isfinite(pilot_seconds) or pilot_seconds < 0 or limit_hours != 60:
        raise ValueError('invalid pilot accounting or study limit')
    total = {'spent_seconds': 0, 'reserved_seconds': 0}
    for state in states:
        for field in total:
            value = state[field]
            if not math.isfinite(value) or value < 0:
                raise ValueError('invalid batch accounting')
            total[field] += value
    new = reservation_seconds(configs)
    bound = require_budget(pilot_seconds, total, new, limit_hours)
    return {'bound_seconds': bound, 'new_limit_seconds': new,
            'additional_runs': 20 * len(configs), 'states': total}
