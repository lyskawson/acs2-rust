import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import tempfile
import time

from agent_names import canonical_row

T19 = 2.093024054408263
NONLEARNING_FIELDS = {'host', 'cpu_model', 'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def load_plan(path, expected_sha256):
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError('registered plan checksum mismatch')
    return json.loads(content)


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.', delete=False) as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
        temporary = Path(stream.name)
    temporary.replace(path)


def stop_phase(path, detail):
    path = Path(path)
    event = dict(detail, unix=time.time())
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix='STOP-event-', suffix='.json', delete=False) as stream:
        json.dump(event, stream, indent=2)
    try:
        with path.open('x') as stream:
            json.dump(event, stream, indent=2)
    except FileExistsError:
        pass


def seconds(value):
    days, clock = value.split('-', 1) if '-' in value else ('0', value)
    return float(days) * 86400 + sum(float(part) * 60 ** i for i, part in enumerate(reversed(clock.split(':'))))


def allocation_rows(text):
    rows = list(csv.DictReader(io.StringIO(text), delimiter='|'))
    return [row for row in rows if '.' not in row['JobIDRaw'] and '[' not in row['JobID']]


def accounting(ledger, records):
    known = {row['JobID']: row for row in records}
    spent = reserved = 0.0
    for entry in ledger:
        if not entry.get('array_id'):
            raise ValueError('unresolved submission intent; reconcile before submitting')
        for task, run in enumerate(entry['runs']):
            job = f"{entry['array_id']}_{task}"
            row = known.get(job, run.get('terminal_accounting'))
            limit = run['limit_seconds']
            if row:
                consumed = float(row['CPUTimeRAW'])
                spent += consumed
                if row['State'].split()[0].split('+')[0] not in TERMINAL:
                    reserved += max(0, limit - consumed)
            else:
                reserved += limit
    return {'spent_seconds': spent, 'reserved_seconds': reserved}


def require_budget(pilot_seconds, state, new_seconds, limit_hours=40):
    bound = pilot_seconds + state['spent_seconds'] + state['reserved_seconds'] + new_seconds
    if bound > limit_hours * 3600:
        raise ValueError(f'phase budget would be exceeded: {bound / 3600:.6f} hours')
    return bound


def summarize(values):
    if len(values) != 20 or not all(math.isfinite(value) for value in values):
        raise ValueError('the registered analysis requires 20 finite seed values')
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(20)
    return {'n': 20, 'mean': mean, 'se': se, 'ci_low': mean - T19 * se, 'ci_high': mean + T19 * se}


def verdict(acs2_final, er_final, acs2_first, floor, ceiling):
    if max(acs2_final['ci_low'], er_final['ci_low']) <= floor:
        return 'floor_baselines_retain_for_successors'
    if acs2_first['mean'] >= 0.9 * ceiling:
        return 'too_easy_no_speed_ranking'
    return 'compare'


def learning_fields(row):
    return canonical_row({key: value for key, value in row.items() if key not in NONLEARNING_FIELDS})


def compare_pilot(grid_rows, pilot_rows):
    if len(grid_rows) != len(pilot_rows):
        raise ValueError('seed-42 pilot and grid point counts differ')
    for grid, pilot in zip(grid_rows, pilot_rows):
        if learning_fields(grid) != learning_fields(pilot):
            keys = sorted(key for key in set(grid) | set(pilot) if key not in NONLEARNING_FIELDS and grid.get(key) != pilot.get(key))
            raise ValueError(f'seed-42 learning fields differ at {grid.get("nominal_step")}: {keys}')
    return sum(len(learning_fields(row)) for row in grid_rows)
