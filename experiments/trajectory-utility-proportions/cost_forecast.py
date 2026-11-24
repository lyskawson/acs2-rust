import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

from proportions import (ABLATION, ATTEMPT_SPREAD, MEASURES_BY_ARM, SCOUTS, checksum, configuration,
                         label, pilot_cells, read, registered)


MARGINS = dict(late_drift=1.20, wall_attempt=1.45, seed_order=1.27,
               archive_attempt=ATTEMPT_SPREAD, overhead_seconds=120)


def json_lines(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def archive_costs(archive, configs, seeds):
    cost_path = archive / 'registered-analysis/run-costs.csv'
    index_path = archive / 'registered-analysis/full-index.json'
    wanted = {(task, agent, seed) for task in configs for agent in MEASURES_BY_ARM for seed in seeds}
    with cost_path.open() as stream:
        costs = {}
        for row in csv.DictReader(stream):
            key = row['task'], row['arm'], int(row['seed'])
            if key in wanted:
                if key in costs:
                    raise ValueError('duplicate archived cost')
                allocation = float(row['measurement_step_or_allocation_cpu_seconds'])
                process = float(row['process_cpu_seconds'])
                if not math.isfinite(allocation) or not math.isfinite(process) or min(allocation, process) <= 0:
                    raise ValueError('invalid 6a CPU cost')
                costs[key] = allocation, process
    matched = {}
    for entry in read(index_path):
        key = entry['task'], entry['agent'], entry['seed']
        if key not in wanted or key[2] != 42:
            continue
        path = Path(entry['source']).resolve()
        if key[:2] in matched or archive.resolve() not in path.parents or checksum(path) != entry['sha256']:
            raise ValueError('invalid archival 6a source')
        completion = read(path.with_name('completion.json'))
        resources = path.with_name('resources.jsonl')
        if completion['sha256']['resources.jsonl'] != checksum(resources):
            raise ValueError('archived cost trace differs from its original checksum')
        values = checkpoint_cpu(resources)
        expected = configs[key[0]]['targets']
        if list(values) != expected or completion['rows_seen'] != len(expected):
            raise ValueError('incomplete archival 6a checkpoints')
        values[expected[-1]] = costs[key][1]
        matched[key[:2]] = values
    if set(costs) != wanted or set(matched) != {key[:2] for key in wanted}:
        raise ValueError('6a costs do not cover every matching maze, arm and seed')
    return costs, matched, {'full-index': checksum(index_path), 'run-costs': checksum(cost_path)}


def checkpoint_cpu(path):
    return {row['nominal_step']: row['process_cpu_seconds_at_observation'] for row in json_lines(path)}


def ceiling_minute(seconds):
    return math.ceil(seconds / 60) * 60


def late_ratios(points):
    return [row['overhead_ratio'] for row in points if row['checkpoint'] >= 100_000]


def dependent_limit(scout_seconds, seed_spread):
    return ceiling_minute(scout_seconds * seed_spread * ATTEMPT_SPREAD + MARGINS['overhead_seconds'])


def time_limits(allocations, seed_42, ratio_high, scout, full):
    estimate = max(1.0, ratio_high)
    max_seed = max(allocations)
    spread = max_seed / seed_42
    pad = MARGINS['overhead_seconds']
    prefix = MARGINS['late_drift'] * MARGINS['wall_attempt']
    ordinary_margin = ATTEMPT_SPREAD if full else prefix * MARGINS['seed_order']
    regular = ceiling_minute(max_seed * estimate * ordinary_margin + pad)
    scout_limit = ceiling_minute(seed_42 * estimate * prefix + pad) if scout else None
    envelope = dependent_limit(scout_limit, spread) if scout else None
    return dict(archive_seed_spread=spread, attempt_spread=ATTEMPT_SPREAD,
                ordinary_limit_seconds=regular if not scout else None,
                scout_limit_seconds=scout_limit, after_scout_initial_limit_seconds=envelope)


def reservation_seconds(limits, seed_count):
    return sum(
        (row['scout_limit_seconds'] + (seed_count - 1) * row['after_scout_initial_limit_seconds']
         if row['scout_limit_seconds'] else
         (seed_count - int(row['full_pilot_reused'])) * row['ordinary_limit_seconds'])
        for row in limits
    )


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verified_pilot(root, pilot_runs):
    manifest_path = root / 'files.sha256.json'
    manifest = read(manifest_path)
    for relative, digest in manifest.items():
        path = (root / relative).resolve()
        if root.resolve() not in path.parents or checksum(path) != digest:
            raise ValueError(f'sealed pilot file differs: {relative}')
    summary = read(root / 'pilot-summary.json')
    if set(summary['completed']) != {run['run_id'] for run in pilot_runs}:
        raise ValueError('the sealed pilot must cover all 168 registered cells')
    attempts = []
    for wave in range(summary['waves']):
        result = read(root / f'wave-{wave:02d}' / 'sealed.json')
        for run_id, attempt in result['attempts'].items():
            attempts.append(dict(wave=wave, run_id=run_id, job=attempt['job'], job_raw=attempt['job_raw'],
                                 state=attempt['state'], limit_seconds=attempt['limit_seconds'],
                                 charged_cpu_seconds=attempt['charged_cpu_seconds']))
    if len({row['job_raw'] for row in attempts}) != len(attempts):
        raise ValueError('a pilot allocation was counted twice')
    spent = sum(row['charged_cpu_seconds'] for row in attempts)
    if spent + summary['build_cpu_seconds'] != summary['total_charged_cpu_seconds']:
        raise ValueError('pilot charge does not reconcile to its attempts')
    return manifest, summary, attempts


def forecast(root, archive, output):
    study, _, configs, _ = registered()
    names = read(ABLATION / 'maze-names.json')
    if not configs.keys() <= names.keys():
        raise ValueError('thesis maze names do not cover the registered tasks')
    _, _, _, pilot_runs = pilot_cells()
    _, summary, attempts = verified_pilot(root, pilot_runs)
    costs, checkpoints, checksums = archive_costs(archive, configs, study['seeds'])
    pilot_rows, matched, limits, per_seed = [], [], [], []
    for run in pilot_runs:
        task, agent, seed = run['task'], run['agent'], run['seed']
        entry = summary['completed'][run['run_id']]
        directory = root / entry['relative_directory']
        records = json_lines(directory / 'rows.jsonl')
        observations = checkpoint_cpu(directory / 'resources.jsonl')
        completion = read(directory / 'completion.json')
        total_cpu = completion['user_cpu_seconds'] + completion['system_cpu_seconds']
        targets = configuration(run, configs)['targets']
        if [row['nominal_step'] for row in records] != targets or list(observations) != targets:
            raise ValueError('unexpected pilot checkpoints')
        full = targets[-1] == configs[task]['targets'][-1]
        if full:
            observations[targets[-1]] = total_cpu
        common = dict(task=task, maze=names[task], arm=label(agent, run['tu_share']),
                      agent=agent, tu_share=run['tu_share'])
        first_match = len(matched)
        prior_extended = 0.0
        for record in records:
            point = record['nominal_step']
            process = observations[point]
            base = checkpoints[task, agent][point]
            extended = record['test_caps']['extended']['cumulative_cpu_seconds']
            utility = record['trajectory_utility']['cost']['cpu_seconds']
            if (not 0 < base or not prior_extended <= extended < process
                    or not 0 < utility <= process - extended):
                raise ValueError('invalid process or evaluation CPU observation')
            prior_extended = extended
            ratio = process / base
            matched.append(dict(common, checkpoint=point, process_cpu_seconds=process,
                                base_6a_process_cpu_seconds=base, overhead_ratio=ratio,
                                overhead_excluding_second_cap_ratio=(process - extended) / base,
                                extended_test_cpu_seconds=extended, utility_cpu_seconds=utility))
        last = matched[-1]
        pilot_rows.append(dict(common, run_id=run['run_id'], pilot_checkpoint=targets[-1], full_budget=full,
                               process_cpu_seconds=total_cpu, charged_cpu_seconds=entry['charged_cpu_seconds'],
                               extended_test_cpu_seconds=last['extended_test_cpu_seconds'],
                               utility_cpu_seconds=last['utility_cpu_seconds'], peak_rss_kib=completion['peak_rss_kib']))
        ratios = late_ratios(matched[first_match:])
        ratio, low, high = last['overhead_ratio'], min(ratios), max(ratios)
        allocations = [costs[task, agent, s][0] for s in study['seeds']]
        seed_42 = costs[task, agent, seed][0]
        limit = time_limits(allocations, seed_42, high, task in SCOUTS, full)
        limits.append(dict(common, matched_checkpoint=targets[-1], cost_ratio=ratio, ratio_min=low, ratio_max=high,
                           archive_min_seed_seconds=min(allocations), archive_max_seed_seconds=max(allocations),
                           archive_seed_42_seconds=seed_42, full_pilot_reused=full, **limit))
        for grid_seed in study['seeds']:
            allocation, process = costs[task, agent, grid_seed]
            per_seed.append(dict(common, seed=grid_seed, from_full_pilot=full and grid_seed == seed,
                                 process_cpu_seconds=process * ratio, allocation_cpu_seconds=allocation * ratio,
                                 low_cpu_seconds=allocation * low, high_cpu_seconds=allocation * high))
    cost_totals = []
    for arm in dict.fromkeys(row['arm'] for row in per_seed):
        subset = [row for row in per_seed if row['arm'] == arm]
        cost_totals.append(dict(arm=arm, grid_cells=len(subset),
                                forecast_cpu_hours=sum(row['allocation_cpu_seconds'] for row in subset) / 3600,
                                forecast_remaining_cpu_hours=sum(row['allocation_cpu_seconds'] for row in subset
                                                                  if not row['from_full_pilot']) / 3600,
                                low_cpu_hours=sum(row['low_cpu_seconds'] for row in subset) / 3600,
                                high_cpu_hours=sum(row['high_cpu_seconds'] for row in subset) / 3600))
    reserved = reservation_seconds(limits, len(study['seeds']))
    result = dict(state='complete', pilot_cells=len(pilot_rows), full_pilot_cells=sum(row['full_budget'] for row in pilot_rows),
                  forecast_cells=len(per_seed), matched_checkpoints=len(matched), pilot_waves=summary['waves'],
                  attempts=len(attempts), total_pilot_cpu_hours=summary['total_charged_cpu_seconds'] / 3600,
                  completed_allocation_cpu_hours=sum(row['charged_cpu_seconds'] for row in pilot_rows) / 3600,
                  build_cpu_hours=summary['build_cpu_seconds'] / 3600,
                  process_cpu_hours=sum(row['process_cpu_seconds'] for row in pilot_rows) / 3600,
                  utility_cpu_hours=sum(row['utility_cpu_seconds'] for row in pilot_rows) / 3600,
                  second_test_cap_cpu_hours=sum(row['extended_test_cpu_seconds'] for row in pilot_rows) / 3600,
                  median_second_cap_share=median(row['extended_test_cpu_seconds'] / row['process_cpu_seconds']
                                               for row in pilot_rows),
                  peak_rss_kib=max(row['peak_rss_kib'] for row in pilot_rows),
                  forecast_cpu_hours=sum(row['forecast_cpu_hours'] for row in cost_totals),
                  forecast_remaining_cpu_hours=sum(row['forecast_remaining_cpu_hours'] for row in cost_totals),
                  forecast_cpu_hours_range=[sum(row[key] for row in cost_totals) for key in ('low_cpu_hours', 'high_cpu_hours')],
                  initial_reserved_grid_cpu_hours=reserved / 3600,
                  attempt_spread=ATTEMPT_SPREAD, margins=MARGINS, measurement_commit=summary['measurement_commit'],
                  binary_sha256=summary['binary_sha256'],
                  inputs=dict(archive=checksums, pilot_manifest=checksum(root / 'files.sha256.json')),
                  cost_model='Multiply each full 6a seed cost by the corresponding new arm-to-6a seed-42 process CPU ratio '
                  'at the last common checkpoint. Use the observed ratio extrema at >=100k as scenario bounds. '
                  'A complete pilot supplies its final process CPU; partial observations use schedstat samples.',
                  limit_model='Floor the upper ratio at one. Use the maximum 6a seed cost for ordinary runs and '
                  'the 6a seed-42 allocation cost for scouts. Multiply full-pilot ordinary cells by the archived '
                  'attempt spread 1.41. For scouts use 1.20 late drift and 1.45 wall-clock spread. For other '
                  'prefix-only cells also use 1.27 for changes in relative seed costs. Add 120s and round up to '
                  'a minute. The dependent envelope is the scout limit times the archive maximum-to-42 seed '
                  'ratio and 1.41, plus 120s, rounded up; use the actual charge after the scout.')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'pilot-costs.csv', pilot_rows)
    write_csv(output / 'pilot-attempts.csv', attempts)
    write_csv(output / 'matched-checkpoints.csv', matched)
    write_csv(output / 'forecast-runs.csv', per_seed)
    write_csv(output / 'forecast-by-arm.csv', cost_totals)
    write_csv(output / 'time-limits.csv', limits)
    (output / 'forecast.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--archive-6a', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(forecast(args.pilot, args.archive_6a, args.out), indent=2))
