import argparse
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics

from analysis import SEEDS, VIEWS, analyze_maze, diagnostic_summaries, metrics, success, summary
from approved_execution import APPROVED_SHA256, approved
from protocol import HER, SHORTENED, label, pilot_roots
from reused import reused
from verify import verify_study
from agent_names import canonical_run_name
from collect import outside_repository
from grid_attempts import attempts
from grid_protocol import allocation_rows, atomic_json


def write_csv(path, columns, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_run(entry, arm):
    path = Path(entry['file']).parent if 'file' in entry else entry['path']
    rows = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
    rows = [dict(row, agent=canonical_run_name(row['agent']), arm=arm) for row in rows]
    completion = json.loads((path / 'completion.json').read_text())
    allocated = entry['cost']['allocation_cpu_seconds'] if 'cost' in entry else entry['allocation_cpu_seconds']
    cost = dict(allocation_cpu_seconds=allocated, process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                elapsed_seconds=completion['elapsed_seconds'], peak_rss_kib=completion['peak_rss_kib'],
                train_wall_seconds=rows[-1]['wall_seconds_train'], eval_wall_seconds=rows[-1]['wall_seconds_eval'])
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ('rows.jsonl', 'completion.json')}
    return rows, cost, hashes


def allocation_inventory(root):
    root = Path(root)
    scope = json.loads((root / 'scope.json').read_text())
    result, identifiers = [], set()
    for declared in scope['batch_roots']:
        batch = root / Path(declared).name
        if not (batch / 'ledger.json').exists():
            continue
        ledger = json.loads((batch / 'ledger.json').read_text())
        records = {row['JobID']: row for row in allocation_rows((batch / 'accounting.psv').read_text())}
        for attempt in attempts(batch, ledger):
            selector = f"{attempt['array_id']}_{attempt['task_id']}"
            record = records[selector]
            if record['JobIDRaw'] in identifiers:
                raise ValueError('allocation appears twice in the scope')
            identifiers.add(record['JobIDRaw'])
            purpose = 'pilot' if batch.name in pilot_roots() else 'test' if batch.name.startswith('test-') else batch.name
            completion_path = attempt['path'] / 'completion.json'
            completion = json.loads(completion_path.read_text()) if completion_path.exists() else None
            process_cpu = completion['user_cpu_seconds'] + completion['system_cpu_seconds'] if completion else None
            result.append(dict(root=batch.name, purpose=purpose, attempt_id=attempt['attempt_id'], job_id=selector,
                job_id_raw=record['JobIDRaw'], seed=attempt['seed'], agent=attempt['agent'],
                allocation_cpu_seconds=int(record['CPUTimeRAW']), state=record['State'], exit_code=record['ExitCode'],
                limit_seconds=attempt['limit_seconds'], process_cpu_seconds=process_cpu))
    return result


def tuning_outputs(plan, tuning, selection):
    table, seed_rows, diagnostics = [], [], []
    for config in plan['configurations']:
        chosen = selection['selections'][config['id']]
        for p in plan['candidates']:
            arm = label(SHORTENED, p)
            rows = [row for (config_id, agent, seed), entry in tuning.items()
                    if config_id == config['id'] + '_' + arm for row in entry['rows']]
            targets = config['targets']
            score = next(value for value in chosen['validation_scores'] if value['proportion'] == p)
            table.append(dict(configuration_id=config['id'], task=config['task'], arm=arm, proportion=p,
                              selected=p == chosen['proportion'], n=5, mean=score['mean'], se=score['se']))
            for seed in range(42, 47):
                values = [success(row['evaluation_blocks']['validation']) for row in rows if row['seed'] == seed]
                if len(values) != len(targets):
                    raise ValueError('tuning table is missing a point')
                exact = sum(values, Fraction()) / len(values)
                seed_rows.append(dict(configuration_id=config['id'], arm=arm, proportion=p, seed=seed, score_exact=str(exact)))
            for view in VIEWS:
                summaries = diagnostic_summaries(rows, targets, view, tuple(range(42, 47)), ('validation',))
                diagnostics.extend(dict(purpose='tuning', configuration_id=config['id'], arm=arm, view=view,
                                        metric=name, **{key: value[key] for key in ('n', 'mean', 'se')}) for name, value in summaries.items())
    across = [dict(proportion=p, mazes=6, mean=statistics.mean(row['mean'] for row in table if row['proportion'] == p), descriptive=True)
              for p in plan['candidates']]
    return table, seed_rows, across, diagnostics


def analyze(root, admissibility, her, baseline, output):
    root, output = Path(root), outside_repository(output)
    new_entries, tuning, selection, execution = verify_study(root)
    plan = approved(root / 'approved-plan.json')
    archived = reused(plan, admissibility, her, baseline)
    if set(new_entries) & set(archived):
        raise ValueError('new and reused physical arms overlap')
    entries = new_entries | archived
    output.mkdir(parents=True, exist_ok=True)
    results = dict(plan_sha256=APPROVED_SHA256, seeds=list(SEEDS), views=list(VIEWS), analysis_rules=plan['analysis'],
                   execution=execution, selections=selection, mazes=[], input_checksums={},
                   reference_reuse=dict(runs=len(archived), rerun=False, charged_to_this_study=False,
                       selected_reference_allocation_cpu_seconds=sum(value['allocation_cpu_seconds'] for value in archived.values())))
    counts, scores, comparisons, thresholds, diagnostics, costs = [], [], [], [], [], []
    for config in plan['configurations']:
        p = selection['selections'][config['id']]['proportion']
        names = [label(SHORTENED, 2 / 3), label(HER, 2 / 3), 'acs2er_episodic', 'acs2', 'acs2er', label(HER, 0.5)]
        if p != 2 / 3:
            names.append(label(SHORTENED, p))
        dataset, cost_records = {}, {}
        for arm in names:
            dataset[arm], cost_records[arm] = [], []
            for seed in SEEDS:
                entry = entries[config['id'], arm, seed]
                rows, cost, hashes = load_run(entry, arm)
                dataset[arm].extend({key: value for key, value in row.items() if key not in ('goal_protocol', 'preset')} for row in rows)
                cost_records[arm].append(cost)
                results['input_checksums']['|'.join([config['id'], arm, str(seed)])] = hashes
                costs.append(dict(configuration_id=config['id'], arm=arm, agent=rows[0]['agent'], seed=seed,
                    purpose=entry.get('purpose', 'reused'), reused_reference=entry.get('reused', False), cpu_model=rows[0]['cpu_model'], **cost))
                for row in rows:
                    block = row['evaluation_blocks']['test']
                    exact = success(block) * block['evaluation_pairs']
                    counts.append(dict(configuration_id=config['id'], arm=arm, seed=seed, nominal_step=row['nominal_step'],
                                       successes=exact.numerator, evaluation_pairs=block['evaluation_pairs']))
        maze = analyze_maze(dataset, config, plan, p)
        for arm in names:
            maze['arms'][arm]['cost'] = {field: summary([value[field] for value in cost_records[arm]]) for field in cost_records[arm][0]}
            for view in VIEWS:
                arm_result = maze['arms'][arm]
                for seed, score in arm_result['success'][view]['seed_scores'].items():
                    scores.append(dict(configuration_id=config['id'], arm=arm, seed=int(seed), view=view, score_exact=score))
                for metric, value in arm_result['diagnostics'][view].items():
                    diagnostics.append(dict(purpose='final_arms', configuration_id=config['id'], arm=arm, view=view,
                                            metric=metric, **{key: value[key] for key in ('n', 'mean', 'se')}))
        for comparison in maze['comparisons']:
            value = comparison['difference']
            comparisons.append(dict(configuration_id=config['id'], view=comparison['view'], left_alias=comparison['left_alias'],
                right_alias=comparison['right_alias'], left=comparison['left'], right=comparison['right'], **value))
        for alias, crossings in maze['threshold_crossings'].items():
            for threshold, point in crossings.items():
                thresholds.append(dict(configuration_id=config['id'], alias=alias, arm=maze['aliases'][alias], threshold=threshold,
                                       threshold_exact=maze['thresholds'][threshold], first_nominal_step=point))
        results['mazes'].append(maze)
    table, tuning_seeds, across, tuning_diagnostics = tuning_outputs(plan, tuning, selection)
    results['tuning_table'], results['tuning_across_maze_means'] = table, across
    diagnostics.extend(tuning_diagnostics)
    pilot = [json.loads(line) for line in (root / 'pilot-rows.jsonl').read_text().splitlines()]
    for config in plan['configurations']:
        for arm in {row['arm'] for row in pilot if row['task'] == config['task']}:
            rows = [row for row in pilot if row['task'] == config['task'] and row['arm'] == arm]
            blocks = tuple(rows[0]['evaluation_blocks'])
            for view in VIEWS:
                values = diagnostic_summaries(rows, config['targets'], view, (42,), blocks)
                diagnostics.extend(dict(purpose='pilot', configuration_id=config['id'], arm=arm, view=view,
                    metric=metric, **{key: value[key] for key in ('n', 'mean', 'se')}) for metric, value in values.items())
    for key, entry in tuning.items():
        arm = entry['rows'][0]['arm']
        rows, cost, _ = load_run(entry, arm)
        costs.append(dict(configuration_id=key[0].removesuffix('_' + arm), arm=arm, agent=SHORTENED, seed=key[2],
            purpose='tuning', reused_reference=False, cpu_model=rows[0]['cpu_model'], **cost))
    from archive import catalog
    for name in pilot_roots():
        pilot_entries, _ = catalog(root / name)
        for key, entry in pilot_entries.items():
            arm = entry['rows'][0]['arm']
            rows, cost, _ = load_run(entry, arm)
            costs.append(dict(configuration_id=key[0].removesuffix('_' + arm), arm=arm, agent=rows[0]['agent'], seed=key[2],
                purpose='pilot', reused_reference=False, cpu_model=rows[0]['cpu_model'], **cost))
    inventory = allocation_inventory(root)
    if sum(row['allocation_cpu_seconds'] for row in inventory) != execution['allocated_cpu_seconds']:
        raise ValueError('allocation export differs from verified scope accounting')
    purpose_costs = [dict(purpose=purpose, allocation_cpu_seconds=sum(row['allocation_cpu_seconds'] for row in inventory if row['purpose'] == purpose),
                          attempts=sum(row['purpose'] == purpose for row in inventory),
                          measured_process_cpu_seconds=sum(row['process_cpu_seconds'] for row in inventory if row['purpose'] == purpose and row['process_cpu_seconds'] is not None),
                          process_unmeasured_attempts=sum(row['purpose'] == purpose and row['process_cpu_seconds'] is None for row in inventory)) for purpose in ('pilot', 'tuning', 'control', 'test')]
    if {row['purpose']: row['allocation_cpu_seconds'] for row in purpose_costs} != execution['purpose_allocation_cpu_seconds']:
        raise ValueError('allocation purposes do not reconcile')
    results['allocation_purposes'] = purpose_costs
    write_csv(output / 'success-counts.csv', list(counts[0]), counts)
    write_csv(output / 'seed-scores.csv', list(scores[0]), scores)
    write_csv(output / 'comparisons.csv', list(comparisons[0]), comparisons)
    write_csv(output / 'thresholds.csv', list(thresholds[0]), thresholds)
    write_csv(output / 'diagnostics.csv', list(diagnostics[0]), diagnostics)
    write_csv(output / 'costs.csv', list(costs[0]), costs)
    write_csv(output / 'tuning.csv', list(table[0]), table)
    write_csv(output / 'tuning-seed-scores.csv', list(tuning_seeds[0]), tuning_seeds)
    write_csv(output / 'tuning-across-mazes.csv', list(across[0]), across)
    write_csv(output / 'allocation-attempts.csv', list(inventory[0]), inventory)
    write_csv(output / 'allocation-purposes.csv', list(purpose_costs[0]), purpose_costs)
    results['analysis_sources'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(__file__).parent.glob('*.py'))}
    atomic_json(output / 'results.json', results)
    atomic_json(output / 'verification.json', execution)
    return dict(new_complete_runs=execution['new_complete_runs'], reference_runs=len(archived),
                comparisons=len(comparisons), allocated_cpu_hours=execution['allocated_cpu_seconds'] / 3600, output=str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--admissibility-root', required=True, type=Path)
    parser.add_argument('--her-root', required=True, type=Path)
    parser.add_argument('--baseline-root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.admissibility_root, args.her_root, args.baseline_root, args.out)))
