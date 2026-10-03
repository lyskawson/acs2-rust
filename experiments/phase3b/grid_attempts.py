import json
from pathlib import Path

from collect import complete_rows
from grid_protocol import learning_fields


def attempts(root, ledger, configuration_id=None):
    result = []
    for entry in ledger:
        config_id = entry['configuration_id']
        if configuration_id is not None and config_id != configuration_id:
            continue
        for index, run in enumerate(entry['runs']):
            name = f"{config_id}_{run['agent']}_s{run['seed']}"
            if 'relative_directory' in run:
                path = root / run['relative_directory']
            else:
                matches = list((root / 'batches').glob(f'*-{config_id}/runs/{name}'))
                path = matches[0] if len(matches) == 1 else root / 'missing' / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('attempt directory is outside the batch')
            result.append(dict(run, configuration_id=config_id, array_id=entry['array_id'], task_id=index,
                               attempt_id=run.get('attempt_id', f"slurm-{entry['array_id']}-{index}"),
                               submitted_unix=entry.get('submitted_unix', 0), path=path))
    identifiers = [item['attempt_id'] for item in result]
    paths = [str(item['path']) for item in result]
    if len(set(identifiers)) != len(identifiers) or len(set(paths)) != len(paths):
        raise ValueError('duplicate attempt identifier or directory')
    return result


def inspect_attempt(attempt, config):
    path = attempt['path']
    result = dict(attempt_id=attempt['attempt_id'], agent=attempt['agent'], seed=attempt['seed'],
                  array_id=attempt['array_id'], task_id=attempt['task_id'], cause=attempt.get('cause', 'registered first attempt'),
                  status='no_rows', rows=0, incomplete_final_line=False, selected=False)
    if not (path / 'rows.jsonl').exists():
        return result, []
    rows, tail = complete_rows(path / 'rows.jsonl')
    result.update(rows=len(rows), incomplete_final_line=tail, status='incomplete' if rows or tail else 'no_rows')
    if not (path / 'completion.json').exists():
        return result, rows
    completion = json.loads((path / 'completion.json').read_text())
    result['completion'] = completion
    resources, resource_tail = complete_rows(path / 'resources.jsonl') if (path / 'resources.jsonl').exists() else ([], False)
    if (not tail and not resource_tail and len(rows) == len(config['targets'])
            and [row['nominal_step'] for row in rows] == config['targets']
            and [row['nominal_step'] for row in resources] == config['targets']
            and completion['exit_code'] == 0 and not completion['failure']
            and not completion['incomplete_final_line'] and completion['rows_seen'] == len(rows)):
        result['status'] = 'complete'
    return result, rows


def select_attempts(root, ledger, config):
    inventory, complete = [], {}
    for attempt in attempts(root, ledger, config['id']):
        item, rows = inspect_attempt(attempt, config)
        item['directory'] = str(attempt['path'].relative_to(root))
        inventory.append(item)
        if item['status'] != 'complete':
            continue
        launch = json.loads((attempt['path'] / 'launch.json').read_text())
        order = (launch.get('started_unix', attempt['submitted_unix']), attempt['attempt_id'])
        complete.setdefault((attempt['agent'], attempt['seed']), []).append((order, attempt, rows, item))
    selected = {}
    for key, values in complete.items():
        values.sort(key=lambda value: value[0])
        reference = [learning_fields(row) for row in values[0][2]]
        for value in values[1:]:
            if [learning_fields(row) for row in value[2]] != reference:
                raise ValueError(f'complete attempts disagree on learning fields: {key}, {values[0][1]["attempt_id"]}, {value[1]["attempt_id"]}')
            value[3]['duplicate_of'] = values[0][1]['attempt_id']
        values[0][3]['selected'] = True
        selected[key] = values[0][1]
    return selected, inventory
