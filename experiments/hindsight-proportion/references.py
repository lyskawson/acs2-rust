import hashlib
import json
from pathlib import Path

from protocol import ADMISSIBILITY_SHA256, SHORTENED, configurations
from grid_attempts import select_attempts
from grid_protocol import load_plan
from collect import validate_row


def shortened_rows(root, seeds=range(42, 47)):
    root = Path(root)
    plan = load_plan(root / 'approved-plan.json', ADMISSIBILITY_SHA256)
    result = {}
    for config in configurations():
        batch = root / (config['id'] + '_' + SHORTENED)
        manifest = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        registered = next(item for item in manifest['configurations'] if item['id'] == config['id'])
        if {key: value for key, value in registered.items() if key != 'limits_minutes'} != config:
            raise ValueError('archived shortened configuration differs from registered pairs')
        ledger = json.loads((batch / 'ledger.json').read_text())
        selected, _ = select_attempts(batch, ledger, registered)
        if set(selected) != {(SHORTENED, seed) for seed in range(42, 62)}:
            raise ValueError('admissibility reference archive is incomplete')
        for seed in seeds:
            directory = selected[SHORTENED, seed]['path']
            path = directory / 'rows.jsonl'
            data = path.read_bytes()
            rows = [json.loads(line) for line in data.splitlines()]
            for row, point in zip(rows, config['targets']):
                validate_row(row, config, SHORTENED, seed, plan['measurement_commit'], point)
            result[config['id'], seed] = dict(data=data, sha256=hashlib.sha256(data).hexdigest(), rows=rows)
    return result


def read_index(root):
    root = Path(root)
    index = json.loads((root / 'reference-index.json').read_text())
    result = {}
    for entry in index:
        path = root / entry['file']
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('reference outside study root')
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            raise ValueError('reference archive checksum changed')
        key = entry['configuration_id'], entry['seed']
        if key in result:
            raise ValueError('duplicate reference cell')
        result[key] = dict(entry, rows=[json.loads(line) for line in data.splitlines()])
    expected = {(config['id'], seed) for config in configurations() for seed in range(42, 47)}
    if set(result) != expected:
        raise ValueError('reference index must cover six mazes and five tuning seeds')
    return result
