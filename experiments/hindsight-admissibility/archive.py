import argparse
import hashlib
import json
from pathlib import Path

from protocol import AGENTS, pilot
from execution import measurement_identity
from grid_attempts import attempts, select_attempts
from grid_protocol import allocation_rows, load_plan, require_budget
from grid_scope import archived_snapshot
from collect import validate_row


def audit(root):
    root = Path(root)
    batch = root / 'batch'
    plan = load_plan(batch / 'plan.json', (batch / 'plan.sha256').read_text().strip())
    registration = load_plan(root / 'pilot-registration.json', plan['authorization_sha256'])
    expected = registration['manifest']
    if expected != pilot() | measurement_identity() or any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('pilot execution differs from its fixed registration')
    operations = load_plan(root / 'operations/manifest.json', plan['operations_manifest_sha256'])
    for name, digest in operations['files'].items():
        path = root / 'operations' / name
        if not path.resolve().is_relative_to((root / 'operations').resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('pilot operations snapshot changed')
    state = archived_snapshot(root, plan)
    require_budget(0, state, 0, 2)
    if state['reserved_seconds']:
        raise ValueError('pilot still has reservations')
    ledger = json.loads((batch / 'ledger.json').read_text())
    if len(ledger) != 1 or any(run['limit_seconds'] != 600 for entry in ledger for run in entry['runs']):
        raise ValueError('pilot must use its single registered ten-minute array')
    allocated = {row['JobID']: row for row in allocation_rows((batch / 'accounting.psv').read_text())}
    inventory = attempts(batch, ledger)
    registered_cells = {(config['id'], agent, 42) for config in plan['configurations'] for agent in AGENTS}
    if len(inventory) != 12 or {(run['configuration_id'], run['agent'], run['seed']) for run in inventory} != registered_cells:
        raise ValueError('pilot allocations differ from its twelve registered cells')
    if set(allocated) != {f"{run['array_id']}_{run['task_id']}" for run in inventory}:
        raise ValueError('accounting does not match every pilot attempt')
    rows, runs, cells = [], [], {}
    for config in plan['configurations']:
        selected, _ = select_attempts(batch, ledger, config)
        if set(selected) != {(agent, 42) for agent in AGENTS}:
            raise ValueError('pilot needs both complete arms on every maze')
        for (agent, seed), run in selected.items():
            directory = run['path']
            raw = [json.loads(line) for line in (directory / 'rows.jsonl').read_text().splitlines()]
            launch = json.loads((directory / 'launch.json').read_text())
            if launch['binary_sha256'] != plan['binary_sha256'] or launch['commit'] != plan['measurement_commit']:
                raise ValueError('pilot measurement identity changed')
            manifest = launch['manifest']
            if (manifest['configuration'] != config or manifest['agent'] != agent or manifest['seed'] != seed
                    or manifest['expected_binary_sha256'] != plan['binary_sha256']
                    or manifest['expected_preset'] != plan['preset'] or manifest['plan_sha256'] != (batch / 'plan.sha256').read_text().strip()
                    or launch['partition'] != plan['partition']):
                raise ValueError('pilot launch manifest differs from its allocation')
            for row, point in zip(raw, config['targets']):
                validate_row(row, config, agent, seed, plan['measurement_commit'], point)
                if row['preset'] != plan['preset'] or row['cpu_model'] != plan['cpu_model']:
                    raise ValueError('pilot preset or hardware changed')
                if row['match_formations_train'] != 10 * row['actual_steps']:
                    raise ValueError('full-update matching volume changed')
            completion = json.loads((directory / 'completion.json').read_text())
            record = allocated[f"{run['array_id']}_{run['task_id']}"]
            if record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0' or record['Partition'] != 'lem-cpu-normal' or record['AllocCPUS'] != '1':
                raise ValueError('selected pilot allocation did not complete on its registered resources')
            cost = dict(body_wall_seconds=completion['elapsed_seconds'],
                process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                allocation_cpu_seconds=int(record['CPUTimeRAW']), peak_rss_kib=completion['peak_rss_kib'])
            cells[config['id'], agent] = cost
            final = raw[-1]
            replay = final['replay_diagnostics']
            copies, walls = replay['routes']['relabeled'], replay['wall_bumps']
            n, all_draws = copies['draws'], replay['draws']
            runs.append(dict(configuration_id=config['id'], task=config['task'], agent=agent,
                directory=str(directory.relative_to(root)), rows_sha256=hashlib.sha256((directory / 'rows.jsonl').read_bytes()).hexdigest(), cost=cost,
                diagnostics=dict(trivial_share=copies['already_reached'] / n,
                    after_end_share=copies['after_counterfactual_end'] / n,
                    fallback_share=replay['fallbacks'] / all_draws,
                    terminal_copy_share=copies['done'] / n,
                    wall_copy_share=walls['unchanged_state'] / n,
                    terminal_wall_copy_share=walls['unchanged_state_terminal'] / n,
                    wall_terminal_share=walls['unchanged_state_terminal'] / walls['unchanged_state'] if walls['unchanged_state'] else None,
                    test_value_diagnostics=final['evaluation_blocks']['test']['value_diagnostics'],
                    population=final['population_classifiers'], knowledge=final['knowledge'])))
            rows.extend(raw)
    return dict(runs=runs, allocations=len(inventory), rows=len(rows), allocation_cpu_seconds=int(state['spent_seconds']),
                measurement_commit=plan['measurement_commit'], binary_sha256=plan['binary_sha256']), rows, cells


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary, rows, _ = audit(args.root)
    (args.root / 'pilot-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.root / 'pilot-rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    print(json.dumps(dict(allocations=summary['allocations'], rows=summary['rows'], allocation_cpu_seconds=summary['allocation_cpu_seconds'])))
