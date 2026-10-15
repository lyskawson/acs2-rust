import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from execution import authorization_manifest, pilot_measurement_identity
from protocol import AGENTS, PILOT_PARTS, PRIORITIZED, ROOT, SECOND_TEMPERATURE_ORDER, configurations
from summarize_check import metrics

sys.path.append(str(ROOT / 'experiments/grid'))
from collect import validate_row
from grid_attempts import attempts, inspect_attempt, verify_artifacts
from grid_authorization import require_derivation
from grid_protocol import TERMINAL, allocation_rows, atomic_json, learning_fields, load_plan


def signature(row):
    return json.dumps(learning_fields(row), sort_keys=True, separators=(',', ':'))


def require_agreement(observations):
    seen = {}
    for item in observations:
        for row in item['rows']:
            key = row['nominal_step']
            value = signature(row)
            if key in seen and seen[key] != value:
                raise ValueError('repeated pilot attempts disagree on learning fields')
            seen[key] = value


def require_coverage(selected, recovery_mode):
    first = {(config['id'], agent, 0 if agent.endswith('_uniform') else 1)
             for config in configurations() for agent in AGENTS}
    if not first.issubset(selected):
        raise ValueError('pilot lacks a complete first temperature or control')
    second = {key for key in selected if key[2] == 24}
    mazes = [config['task'] for config in configurations()
             if any(key[0] == config['id'] for key in second)]
    expected_second = {(config['id'], agent, 24) for config in configurations()
                       if config['task'] in mazes for agent in PRIORITIZED}
    if not mazes or second != expected_second or set(selected) != first | second:
        raise ValueError('second temperature needs both head counts on every measured maze')
    if recovery_mode and set(mazes) != set(SECOND_TEMPERATURE_ORDER[:len(mazes)]):
        raise ValueError('second-temperature maze set differs from its registered admission order')
    return sorted(mazes)


def verify_operations_copy(root, plan):
    remote = Path(plan['operations_manifest'])
    path = root / remote.parent.name / remote.name
    manifest = load_plan(path, plan['operations_manifest_sha256'])
    if manifest['commit'] != plan['operations_commit']:
        raise ValueError('operations source identity differs')
    for name, digest in manifest['files'].items():
        file = path.parent / name
        if not file.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError('operations file escapes its snapshot')
        if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError('archived operations snapshot changed')


def audit(root):
    root = Path(root)
    recovery_mode = (root / 'recovery-registration.json').exists()
    registration_name = 'recovery-registration.json' if recovery_mode else 'pilot-registration.json'
    registration = json.loads((root / registration_name).read_text())
    identity = pilot_measurement_identity()
    if registration['measurement'] != identity or registration['authorized_cpu_hours'] != 5:
        raise ValueError('pilot measurement identity or authorization limit differs')
    if hashlib.sha256((root / 'acs2-measure-musl').read_bytes()).hexdigest() != identity['binary_sha256']:
        raise ValueError('archived measurement binary differs from its frozen identity')
    if recovery_mode:
        for name, field in [('pilot-registration.json', 'original_registration_sha256'),
                            ('authorized-cancellation.json', 'cancellation_sha256')]:
            if hashlib.sha256((root / name).read_bytes()).hexdigest() != registration[field]:
                raise ValueError('recovery history checksum mismatch')
        names = ['temperature1-and-controls', *[part['pilot_part'] for part in registration['pilot_parts']]]
    else:
        names = PILOT_PARTS
    observations, inventories, allocations = {}, {}, {}
    checked_operations = set()
    for name in names:
        batch = root / name
        if not batch.exists():
            continue
        plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
        require_derivation(plan, authorization_manifest(plan))
        load_plan(root / Path(plan['authorization_plan']).name, plan['authorization_sha256'])
        scope = load_plan(root / Path(plan['budget_scope']).name, plan['budget_scope_sha256'])
        if scope['authorization_sha256'] != plan['authorization_sha256'] or scope['pilot_allocation_cpu_seconds'] != 0:
            raise ValueError('pilot accounting scope differs from its registration')
        if recovery_mode and name.startswith('recovery-'):
            scoped = [Path(path) for path in scope['batch_roots']]
            base = Path(plan['budget_scope']).parent
            if len(scoped) != len(set(scoped)) or {path.name for path in scoped} != set(names) or any(path.parent != base for path in scoped):
                raise ValueError('recovery accounting omits or duplicates a pilot batch')
        if plan['operations_manifest_sha256'] not in checked_operations:
            verify_operations_copy(root, plan)
            checked_operations.add(plan['operations_manifest_sha256'])
        ledger = json.loads((batch / 'ledger.json').read_text())
        records = allocation_rows((batch / 'accounting.psv').read_text())
        known = {record['JobID']: record for record in records}
        configs = {config['id']: config for config in plan['configurations']}
        inventory = []
        for attempt in attempts(batch, ledger):
            config = configs[attempt['configuration_id']]
            item, rows = inspect_attempt(attempt, config)
            job = f"{attempt['array_id']}_{attempt['task_id']}"
            if job in allocations:
                raise ValueError('duplicate allocation in pilot inventory')
            record = known.get(job)
            if record is None or record['State'].split()[0].split('+')[0] not in TERMINAL:
                raise ValueError('pilot has missing or nonterminal accounting')
            if record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] not in ['0', '1']:
                raise ValueError('wrong pilot allocation resources')
            allocations[job] = int(record['CPUTimeRAW'])
            if rows and record['AllocCPUS'] != '1':
                raise ValueError('measurement did not have one allocated CPU')
            completion = item.get('completion')
            if completion:
                verify_artifacts(attempt['path'], completion)
            if len(rows) > len(config['targets']):
                raise ValueError('too many pilot rows')
            for row, target in zip(rows, config['targets']):
                validate_row(row, config, attempt['agent'], attempt['seed'], plan['measurement_commit'], target)
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                    raise ValueError('pilot row preset or hardware differs')
            if item['status'] == 'complete' and (record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0'):
                raise ValueError('complete rows belong to a failed allocation')
            temperature = config.get('agent_settings', {}).get(attempt['agent'], {}).get('temperature', 0)
            key = config['id'], attempt['agent'], temperature
            item.update(configuration_id=config['id'], directory=str(attempt['path'].relative_to(root)),
                        temperature=temperature, allocation_cpu_seconds=allocations[job], allocation_state=record['State'])
            inventory.append(item)
            observations.setdefault(key, []).append(dict(rows=rows, inventory=item, completion=completion,
                order=(attempt['submitted_unix'], item['directory'])))
        inventories[name] = inventory
    selected = {}
    for key, items in observations.items():
        require_agreement(items)
        complete = [item for item in items if item['inventory']['status'] == 'complete']
        if complete:
            selected[key] = min(complete, key=lambda item: item['order'])
            selected[key]['inventory']['selected'] = True
    second_mazes = require_coverage(selected, recovery_mode)
    spent = sum(allocations.values())
    if spent > 5 * 3600:
        raise ValueError('stage-1 allocated cost exceeds five CPU-hours')
    rows, cells, selected_runs = [], {}, []
    for key, item in sorted(selected.items()):
        rows.extend(item['rows'])
        final, completion = item['rows'][-1], item['completion']
        cell = dict(configuration_id=key[0], agent=key[1], temperature=key[2],
            body_wall_seconds=final['wall_seconds_total'], allocation_cpu_seconds=item['inventory']['allocation_cpu_seconds'],
            process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
            peak_rss_kib=completion['peak_rss_kib'], final_population=final['population_classifiers'],
            maximum_population=max(row['population_classifiers'] for row in item['rows']))
        if key[:2] not in cells or cell['body_wall_seconds'] > cells[key[:2]]['body_wall_seconds']:
            cells[key[:2]] = cell
        selected_runs.append(dict(configuration_id=key[0], agent=key[1], temperature=key[2],
                                  seed=42, directory=item['inventory']['directory'], cost=cell))
    return dict(parts=inventories, allocation_cpu_seconds=spent, allocations=len(allocations),
                runs=len(selected), rows=len(rows), selected_runs=selected_runs, final_cells=list(cells.values()),
                second_temperature_mazes=second_mazes), rows, cells


def summarize_pilot(root):
    root = Path(root)
    summary, rows, cells = audit(root)
    atomic_json(root / 'pilot-summary.json', summary)
    with (root / 'pilot-rows.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    statistics = []
    for row in rows:
        priority = row['replay_diagnostics']['priority']
        detail = dict(task=row['task'], agent=row['agent'], temperature=row['agent_parameters']['temperature'],
                      nominal_step=row['nominal_step'], actual_steps=row['actual_steps'], **metrics(row))
        detail.update(priority_value_queries=priority['value_queries'], proposals=priority['proposals'],
                      accepted_draws=priority['accepted_draws'])
        for purpose, counts in row['replay_diagnostics']['objective_evaluations_by_purpose'].items():
            detail.update({purpose + '_' + key: value for key, value in counts.items()})
        for route, counts in row['replay_diagnostics']['routes'].items():
            detail.update({route + '_' + key: value for key, value in counts.items()})
        statistics.append(detail)
    columns = sorted({key for row in statistics for key in row})
    with (root / 'pilot-statistics.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(statistics)
    raw = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
           for path in sorted(root.rglob('*')) if path.is_file() and path.name != 'raw-files.sha256.json'}
    atomic_json(root / 'raw-files.sha256.json', raw)
    return dict(runs=summary['runs'], rows=summary['rows'], allocation_cpu_seconds=summary['allocation_cpu_seconds'],
                raw_files=len(raw))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize_pilot(args.root)))
