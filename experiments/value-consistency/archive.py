import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from protocol import PILOT_PARTS, ROOT, draft

sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import atomic_json
from grid_attempts import select_attempts


def verifier():
    spec = importlib.util.spec_from_file_location('hindsight_archive_verifier', ROOT / 'experiments/her-variants/verify.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify


def summarize_pilot(root):
    root = Path(root)
    verify = verifier()
    parts = [part for part in PILOT_PARTS if (root / part).exists()]
    if PILOT_PARTS[0] not in parts or sum(part in parts for part in PILOT_PARTS[1:]) != 1:
        raise ValueError('pilot requires the first part and exactly one second-temperature part')
    results = {part: verify(root / part) for part in parts}
    spent = sum(result['allocation_cpu_seconds'] for result in results.values())
    if spent > 5 * 3600:
        raise ValueError('stage-1 allocated cost exceeds five CPU-hours')
    rows = []
    cells = {}
    for part, result in results.items():
        atomic_json(root / part / 'verification.json', result)
        for cell in result['cells']:
            key = cell['configuration_id'], cell['agent']
            if key not in cells or cell['body_wall_seconds'] > cells[key]['body_wall_seconds']:
                cells[key] = cell
        plan = json.loads((root / part / 'plan.json').read_text())
        ledger = json.loads((root / part / 'ledger.json').read_text())
        for config in plan['configurations']:
            selected, _ = select_attempts(root / part, ledger, config)
            for attempt in selected.values():
                rows.extend(json.loads(line) for line in (attempt['path'] / 'rows.jsonl').read_text().splitlines())
    measurement = json.loads((root / 'pilot-registration.json').read_text())['measurement']
    plan = draft(cells, rows, spent, measurement)
    atomic_json(root / 'plan-draft.json', plan)
    atomic_json(root / 'pilot-summary.json', dict(parts=results, allocation_cpu_seconds=spent,
        runs=sum(result['runs'] for result in results.values()), rows=len(rows),
        final_cells=list(cells.values()),
        second_temperature_mazes=sorted({row['task'] for row in rows if row['agent_parameters']['temperature'] == 24})))
    raw = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and ('attempts' in path.parts or path.name in ['plan.json', 'plan.sha256', 'ledger.json', 'accounting.psv', 'scope.json', 'pilot-registration.json']):
            raw[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(root / 'raw-files.sha256.json', raw)
    return dict(runs=sum(result['runs'] for result in results.values()), rows=len(rows),
                allocation_cpu_seconds=spent, raw_files=len(raw), forecast=plan['forecast'],
                requested_cpu_hour_limit=plan['requested_cpu_hour_limit'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_pilot(args.root)))
