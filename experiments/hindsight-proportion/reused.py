import hashlib
import json
from pathlib import Path

from protocol import HER, SHORTENED, label
from agent_names import canonical_run_name
from collect import validate_row
from grid_attempts import select_attempts
from grid_protocol import allocation_rows, load_plan


def reference_catalog(root, registration, digest, names, configurations, seeds=range(42, 62)):
    root, registration = Path(root), Path(registration)
    registered = load_plan(registration, digest)
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    if any(plan[key] != registered[key] for key in ('measurement_commit', 'binary_sha256', 'preset')):
        raise ValueError('reused measurement differs from its registered source, binary or preset')
    ledger = json.loads((root / 'ledger.json').read_text())
    accounting = {row['JobID']: row for row in allocation_rows((root / 'accounting.psv').read_text())}
    entries = {}
    for config in configurations:
        archived = next(item for item in plan['configurations'] if item['id'] == config['id'])
        for key in ('task', 'cap', 'targets', 'registration', 'budget_steps', 'encoding', 'row_pool', 'protocol'):
            if archived[key] != config[key]:
                raise ValueError('reused pair configuration differs: ' + key)
        selected, _ = select_attempts(root, ledger, archived)
        for (agent, seed), attempt in selected.items():
            canonical = canonical_run_name(agent)
            if canonical not in names or seed not in seeds:
                continue
            path = attempt['path']
            rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
            for row, point in zip(rows, config['targets']):
                validate_row(row, archived, agent, seed, plan['measurement_commit'], point)
                if row['preset'] != plan['preset'] or set(row['evaluation_blocks']) != {'training', 'validation', 'test'}:
                    raise ValueError('reused row preset or evaluation blocks differ')
            if len(rows) != len(config['targets']):
                raise ValueError('incomplete reused reference points')
            record = accounting[f"{attempt['array_id']}_{attempt['task_id']}"]
            if record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0' or record['AllocCPUS'] != '1':
                raise ValueError('reused allocation did not complete on one CPU')
            arm = label(canonical, rows[0]['agent_parameters']['relabeled_proportion']) if canonical in (HER, SHORTENED) else canonical
            entries[config['id'], arm, seed] = dict(path=path, agent=canonical, arm=arm,
                allocation_cpu_seconds=int(record['CPUTimeRAW']), reused=True,
                source_rows_sha256=hashlib.sha256((path / 'rows.jsonl').read_bytes()).hexdigest())
    expected = {(config['id'], label(name, 2 / 3) if name in (HER, SHORTENED) else name, seed)
                for config in configurations for name in names for seed in seeds}
    if set(entries) != expected:
        raise ValueError('missing or extra reused reference trajectories')
    return entries


def reused(plan, admissibility, her, baseline):
    admissibility, her, baseline = Path(admissibility), Path(her), Path(baseline)
    result = reference_catalog(her / 'main-grid', her / 'approved-plan.json', plan['references']['her_plan_sha256'],
                               [HER, 'acs2er_episodic'], plan['configurations'])
    result.update(reference_catalog(baseline, baseline / 'plan.json', plan['references']['baseline_plan_sha256'],
                                   ['acs2', 'acs2er'], plan['configurations']))
    for config in plan['configurations']:
        result.update(reference_catalog(admissibility / (config['id'] + '_' + SHORTENED), admissibility / 'approved-plan.json',
            plan['references']['shortened_plan_sha256'], [SHORTENED], [config]))
    return result
