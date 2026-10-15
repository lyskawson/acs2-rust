import hashlib
import json
from pathlib import Path
import sys

from protocol import ROOT

sys.path.append(str(ROOT / 'experiments/grid'))
from agent_names import canonical_run_name
from collect import validate_row
from grid_attempts import select_attempts
from grid_protocol import allocation_rows, load_plan


def catalog(root, approved_path, approved_sha256, names, configurations, seeds):
    root, approved_path = Path(root), Path(approved_path)
    if hashlib.sha256(approved_path.read_bytes()).hexdigest() != approved_sha256:
        raise ValueError('reused reference registration differs')
    registered = json.loads(approved_path.read_text())
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    for key in ['measurement_commit', 'binary_sha256', 'preset']:
        if plan[key] != registered[key]:
            raise ValueError('reused reference measurement identity differs: ' + key)
    ledger = json.loads((root / 'ledger.json').read_text())
    records = {row['JobID']: row for row in allocation_rows((root / 'accounting.psv').read_text())}
    entries = {}
    for config in configurations:
        archived = next(item for item in plan['configurations'] if item['id'] == config['id'])
        for key in ['task', 'cap', 'targets', 'registration', 'budget_steps', 'encoding', 'row_pool', 'protocol']:
            if archived[key] != config[key]:
                raise ValueError(f'reused reference pair protocol differs: {key}')
        selected, _ = select_attempts(root, ledger, archived)
        for (agent, seed), attempt in selected.items():
            name = canonical_run_name(agent)
            if name not in names or seed not in seeds:
                continue
            rows = [json.loads(line) for line in (attempt['path'] / 'rows.jsonl').read_text().splitlines()]
            for row, point in zip(rows, config['targets']):
                validate_row(row, archived, agent, seed, plan['measurement_commit'], point)
                if row['preset'] != plan['preset']:
                    raise ValueError('reused reference preset differs')
                if set(row['evaluation_blocks']) != {'training', 'validation', 'test'}:
                    raise ValueError('reused reference is missing an evaluation block')
            if len(rows) != len(config['targets']):
                raise ValueError('reused reference is missing a registered point')
            record = records[f"{attempt['array_id']}_{attempt['task_id']}"]
            if record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0':
                raise ValueError('reused reference allocation did not complete')
            entries[config['id'], name, seed] = dict(path=attempt['path'], targets=config['targets'], task=config['task'],
                allocation_cpu_seconds=int(record['CPUTimeRAW']))
    expected = {(config['id'], name, seed) for config in configurations for name in names for seed in seeds}
    if set(entries) != expected:
        raise ValueError('reused reference has missing or extra registered cells')
    return entries


def reused(study_plan, her, baseline):
    her, baseline = Path(her), Path(baseline)
    entries = catalog(her / 'main-grid', her / 'approved-plan.json',
                      study_plan['references']['her_plan_sha256'], ['acs2her', 'acs2er_episodic'],
                      study_plan['configurations'], study_plan['seeds'])
    entries.update(catalog(baseline, baseline / 'plan.json',
                           study_plan['references']['baseline_plan_sha256'], ['acs2', 'acs2er'],
                           study_plan['configurations'], study_plan['seeds']))
    return entries
