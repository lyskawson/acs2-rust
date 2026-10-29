import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

from analysis import analyze_maze, diagnostic_summaries, intervals, metrics
from approved_execution import APPROVED_SHA256, approved, catalog_cache
from execution import read_index
from protocol import BLOCKS, ER, SHORTENED, label, setting
from references import read as read_references
from verify import resolved_batch, verify_study
from agent_names import canonical_run_name
from collect import outside_repository
from grid_attempts import attempts
from grid_protocol import atomic_json
from paired_statistics import SEEDS, VIEWS, scores, success


def write_csv(path, records):
    columns = list(dict.fromkeys(key for record in records for key in record))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def load_run(entry, arm):
    path = Path(entry['file']).parent if 'file' in entry else entry['path']
    raw = [json.loads(line) for line in (path / 'rows.jsonl').read_text().splitlines()]
    rows = [dict(row, agent=canonical_run_name(row['agent']), arm=arm) for row in raw]
    completion = json.loads((path / 'completion.json').read_text())
    allocation = entry['cost']['allocation_cpu_seconds'] if 'cost' in entry else entry['allocation_cpu_seconds']
    cost = dict(allocation_cpu_seconds=allocation,
        process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
        elapsed_seconds=completion['elapsed_seconds'], peak_rss_kib=completion['peak_rss_kib'],
        train_wall_seconds=rows[-1]['wall_seconds_train'], eval_wall_seconds=rows[-1]['wall_seconds_eval'])
    hashes = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in ('rows.jsonl', 'completion.json')}
    return rows, cost, hashes


def screening_outputs(plan, screening, selection, references):
    table, seed_rows, counts, diagnostics = [], [], [], []
    for config in plan['configurations']:
        choices = selection['selections'][config['id']]
        if choices['B']['table']['base'] != choices['C']['table']['base']:
            raise ValueError('both axes must reuse exactly the same validation base')
        cells = dict(choices['B']['table'], backward=choices['C']['table']['backward'])
        for mode, cell in cells.items():
            arm = label(config, SHORTENED, mode)
            rows = [row for seed in range(42, 47) for row in references['screening-base', config['id'], seed]['rows']] if mode == 'base' else [
                row for seed in range(42, 47) for row in screening[setting(config, SHORTENED, mode, ['validation'])['id'], SHORTENED, seed]['rows']]
            table.append(dict(configuration_id=config['id'], task=config['task'], setting=mode, arm=arm,
                selected_B=choices['B']['setting'] == mode, selected_C=choices['C']['setting'] == mode,
                n=5, mean=cell['mean'], exact_mean=cell['exact_mean'], se=cell['se']))
            values = scores(rows, config['targets'], 'checkpoint_mean', 'validation', tuple(range(42, 47)), ('validation',))
            seed_rows.extend(dict(configuration_id=config['id'], setting=mode, seed=seed, score_exact=str(value)) for seed, value in values.items())
            for row in rows:
                block = row['evaluation_blocks']['validation']
                count = success(block) * block['evaluation_pairs']
                counts.append(dict(configuration_id=config['id'], setting=mode, seed=row['seed'], nominal_step=row['nominal_step'],
                    successes=count.numerator, evaluation_pairs=block['evaluation_pairs']))
            for view in VIEWS:
                measured = diagnostic_summaries(rows, config['targets'], view, tuple(range(42, 47)), ('validation',))
                diagnostics.extend(dict(purpose='screening_base' if mode == 'base' else 'screening', configuration_id=config['id'],
                    arm=arm, view=view, metric=name, **{key: value[key] for key in ('n', 'mean', 'se')}) for name, value in measured.items())
    return table, seed_rows, counts, diagnostics


def analyze(root, output):
    root, output = Path(root), outside_repository(output)
    plan = approved(root / 'approved-plan.json')
    with catalog_cache():
        tests, screening, selection, state, inventory = verify_study(root)
    reused = read_references(root)
    output.mkdir(parents=True, exist_ok=True)
    result = dict(plan_sha256=APPROVED_SHA256, seeds=list(SEEDS), views=list(VIEWS), rules=plan['analysis'],
        runtime=sys.version, execution=state, selections=selection, mazes=[], input_checksums={})
    counts, seed_scores, comparisons, differences, thresholds, diagnostics, cost_rows, point_metrics, interval_rows = [], [], [], [], [], [], [], [], []
    for config in plan['configurations']:
        selected = selection['selections'][config['id']]
        final_arms = [(label(config, SHORTENED, 'base'), SHORTENED, 'base', 'reused'),
                      (ER, ER, 'base', 'reused'), ('acs2', 'acs2', 'base', 'reused'), ('acs2er', 'acs2er', 'base', 'reused')]
        for axis in ('B', 'C'):
            mode = selected[axis]['setting']
            if mode == 'base':
                continue
            final_arms.extend((label(config, agent, mode), agent, mode, 'test') for agent in (SHORTENED, ER))
            if axis == 'C':
                final_arms.append((label(config, SHORTENED, 'random-pass'), SHORTENED, 'random-pass', 'test'))
        dataset = {}
        for arm, agent, mode, purpose in final_arms:
            dataset[arm] = []
            for seed in SEEDS:
                entry = reused[config['id'], arm, seed] if purpose == 'reused' else tests[
                    setting(config, agent, mode, BLOCKS)['id'], agent, seed]
                rows, cost, hashes = load_run(entry, arm)
                if any(row['agent'] != agent for row in rows):
                    raise ValueError('analysis alias differs from its thesis agent')
                dataset[arm].extend({key: value for key, value in row.items() if key not in ('goal_protocol', 'preset')} for row in rows)
                trajectory = f'{purpose}|{config["id"]}|{arm}|{seed}'
                result['input_checksums'][trajectory] = hashes
                cost_rows.append(dict(configuration_id=config['id'], arm=arm, agent=agent, seed=seed,
                    purpose=purpose, reused_reference=purpose == 'reused', cpu_model=rows[0]['cpu_model'], **cost))
                for row in rows:
                    measurement = dict(trajectory=trajectory, purpose=purpose, configuration_id=config['id'], arm=arm,
                        seed=seed, nominal_step=row['nominal_step'], online_updates=row['online_updates'], replay_updates=row['replay_updates'], **metrics(row))
                    for block, value in row['evaluation_blocks'].items():
                        count = success(value) * value['evaluation_pairs']
                        counts.append(dict(configuration_id=config['id'], arm=arm, seed=seed, nominal_step=row['nominal_step'],
                            block=block, successes=count.numerator, evaluation_pairs=value['evaluation_pairs']))
                        measurement[block + '_successes'] = count.numerator
                        measurement[block + '_evaluation_pairs'] = value['evaluation_pairs']
                    point_metrics.append(measurement)
                interval_rows.extend(dict(trajectory=trajectory, purpose=purpose, configuration_id=config['id'], arm=arm, seed=seed, **value)
                                     for value in intervals(rows))
        maze = analyze_maze(dataset, config, plan, selected)
        for arm, value in maze['arms'].items():
            for view in VIEWS:
                seed_scores.extend(dict(configuration_id=config['id'], arm=arm, seed=int(seed), view=view, score_exact=score)
                                   for seed, score in value['success'][view]['seed_scores'].items())
                diagnostics.extend(dict(purpose='final_arms', configuration_id=config['id'], arm=arm, view=view,
                    metric=name, **{key: measured[key] for key in ('n', 'mean', 'se')}) for name, measured in value['diagnostics'][view].items())
        for comparison in maze['comparisons']:
            difference = comparison['difference']
            comparisons.append(dict(configuration_id=config['id'], task=config['task'], axis=comparison['axis'], setting=comparison['setting'],
                view=comparison['view'], comparison=comparison['comparison'], terms=json.dumps(comparison['terms']),
                **{key: value for key, value in difference.items() if key != 'seed_differences'}))
            differences.extend(dict(configuration_id=config['id'], axis=comparison['axis'], view=comparison['view'],
                comparison=comparison['comparison'], seed=int(seed), difference_exact=value)
                for seed, value in difference['seed_differences'].items())
        for axis, values in maze['threshold_crossings'].items():
            for name, point in values.items():
                thresholds.append(dict(configuration_id=config['id'], axis=axis, arm=maze['aliases'][axis]['S_X'], threshold=name,
                    threshold_exact=maze['thresholds'][name], first_nominal_step=point))
        result['mazes'].append(maze)
    table, screening_seeds, screening_counts, screening_diagnostics = screening_outputs(plan, screening, selection, read_index(root))
    result['screening_table'] = table
    diagnostics.extend(screening_diagnostics)
    for (config_id, agent, seed), entry in screening.items():
        arm = entry['rows'][0]['arm']
        rows, cost, _ = load_run(entry, arm)
        registered = next(config for config in plan['configurations'] if rows[0]['task'] == config['task'])
        trajectory = f'screening|{config_id}|{seed}'
        cost_rows.append(dict(configuration_id=registered['id'], arm=arm, agent=agent, seed=seed,
            purpose='screening', reused_reference=False, cpu_model=rows[0]['cpu_model'], **cost))
        interval_rows.extend(dict(trajectory=trajectory, purpose='screening', configuration_id=registered['id'], arm=arm, seed=seed, **value)
                             for value in intervals(rows))
        point_metrics.extend(dict(trajectory=trajectory, purpose='screening', configuration_id=registered['id'], arm=arm, seed=seed,
            nominal_step=row['nominal_step'], online_updates=row['online_updates'], replay_updates=row['replay_updates'], **metrics(row)) for row in rows)
    value = json.loads((root / 'scope.json').read_text())
    for declared in value['batch_roots']:
        batch = resolved_batch(root, declared)
        if not batch.name.startswith('pilot-') or not (batch / 'ledger.json').exists():
            continue
        for attempt in attempts(batch, json.loads((batch / 'ledger.json').read_text())):
            path = attempt['path'] / 'rows.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
            trajectory = 'pilot|' + declared + '|' + attempt['attempt_id']
            interval_rows.extend(dict(trajectory=trajectory, purpose='pilot', configuration_id=rows[0]['task'] if rows else None,
                arm=rows[0].get('arm', rows[0]['agent']) if rows else None, seed=attempt['seed'], **measured) for measured in intervals(rows))
            point_metrics.extend(dict(trajectory=trajectory, purpose='pilot', configuration_id=row['task'], arm=row.get('arm', row['agent']),
                seed=row['seed'], nominal_step=row['nominal_step'], online_updates=row['online_updates'], replay_updates=row['replay_updates'],
                **metrics(row)) for row in rows)
    result['reference_reuse'] = dict(runs=len(reused), rerun=False, charged_to_this_study=False,
        unique_historical_allocation_cpu_seconds=sum({entry['job_id_raw']: entry['allocation_cpu_seconds'] for entry in reused.values()}.values()))
    result['cost_by_purpose'] = {purpose: dict(allocations=sum(row['purpose'] == purpose for row in inventory),
        allocation_cpu_seconds=sum(row['allocation_cpu_seconds'] for row in inventory if row['purpose'] == purpose),
        failed_allocations=sum(row['purpose'] == purpose and row['state'] != 'COMPLETED' for row in inventory))
        for purpose in ('pilot', 'screening', 'test')}
    atomic_json(output / 'results.json', result)
    exports = {'counts': counts, 'seed-scores': seed_scores, 'comparisons': comparisons,
        'paired-differences': differences, 'thresholds': thresholds, 'screening-table': table,
        'screening-seed-scores': screening_seeds, 'screening-counts': screening_counts,
        'diagnostics': diagnostics, 'run-costs': cost_rows, 'allocations': inventory,
        'checkpoint-metrics': point_metrics, 'update-intervals': interval_rows}
    for name, records in exports.items():
        write_csv(output / (name + '.csv'), records)
    return dict(mazes=len(result['mazes']), comparisons=len(comparisons), test_runs=len(tests),
                screening_runs=len(screening), reused_runs=len(reused), allocation_cpu_seconds=state['spent_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.out)))
