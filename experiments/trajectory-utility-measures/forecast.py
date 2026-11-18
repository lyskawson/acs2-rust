import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
import statistics

from registration import ABLATION, ARMS, OLD_TASKS, S1, ablation, checksum, read
from grid_protocol import allocation_rows, load_plan
from validate import validate_row


MENU = tuple(product((4, 8), (None, 5000, 2000)))
SCOUTS = ('mazee2_alcs', 'mazee3_alcs')


@dataclass(frozen=True)
class EvaluationCosts:
    evaluations: int
    candidates: int
    candidates_at_four: int
    total_cpu: float
    scoring_cpu: float

    def seconds_at(self, candidates, evaluations):
        if not self.evaluations:
            return 0.0
        sampled = self.candidates_at_four if candidates == 4 else self.candidates
        fixed = (self.total_cpu - self.scoring_cpu) / self.evaluations
        variable = self.scoring_cpu / self.candidates * sampled / self.evaluations if self.candidates else 0.0
        return evaluations * (fixed + variable)

    def __sub__(self, earlier):
        return EvaluationCosts(
            self.evaluations - earlier.evaluations, self.candidates - earlier.candidates,
            self.candidates_at_four - earlier.candidates_at_four, self.total_cpu - earlier.total_cpu,
            self.scoring_cpu - earlier.scoring_cpu,
        )


def split_cpu(cost):
    all_costs = EvaluationCosts(
        cost['evaluations'], cost['candidates_sampled'], cost['candidate_slots_at_four'],
        cost['cpu_seconds'], cost['scoring_cpu_seconds'],
    )
    insertion = EvaluationCosts(
        cost['insertion_evaluations'], cost['insertion_candidates'], cost['insertion_candidate_slots_at_four'],
        cost['insertion_cpu_seconds'], cost['insertion_scoring_cpu_seconds'],
    )
    return insertion, all_costs - insertion


def expected_refreshes(retention_steps, refresh_steps, hazard):
    if refresh_steps is None or not hazard:
        return 0.0
    waiting_times = ((retention_steps - repetitions * refresh_steps) * hazard
                     for repetitions in range(1, (retention_steps - 1) // refresh_steps + 1))
    return sum(1.0 - math.exp(-x) * sum(x ** power / math.factorial(power) for power in range(repetitions))
               for repetitions, x in enumerate(waiting_times, 1))


def projected_utility(point, candidates, refresh_steps):
    insertion, refresh = split_cpu(point['cost'])
    retention = point['retained_steps']
    hazard = point['tu_draws'] / (retention * insertion.evaluations)
    reference_refreshes = expected_refreshes(retention, 2000, hazard)
    if refresh_steps == 2000:
        refresh_count = refresh.evaluations
    elif reference_refreshes:
        scaled = refresh.evaluations * expected_refreshes(retention, refresh_steps, hazard) / reference_refreshes
        refresh_count = min(insertion.evaluations, scaled)
    else:
        refresh_count = 0.0
    utility = insertion.seconds_at(candidates, insertion.evaluations) + refresh.seconds_at(candidates, refresh_count)
    return utility, (insertion.evaluations + refresh_count) / point['steps']


def reference_costs(archive, configs, seeds):
    directory = archive / 'grid-evidence'
    index = read(archive / 'local-full-index.json')
    archive_rows = [row for row in index if row['agent'] == S1]
    paths = {(row['task'], row['seed']): Path(row['source']) for row in archive_rows}
    expected = set(product(configs, seeds))
    if set(paths) != expected or len(paths) != len(archive_rows):
        raise ValueError('the original S+1 analysis inputs do not cover the 560 seed/task pairs')
    full = {}
    for name in ('reference', 'grid'):
        with (directory / f'{name}-run-costs.csv').open() as stream:
            for row in csv.DictReader(stream):
                if row['agent'] != S1 or row.get('state', 'COMPLETED') != 'COMPLETED':
                    continue
                key = row['task'], int(row['seed'])
                if (key in full or (row['task'] in OLD_TASKS) != (name == 'reference')
                        or row['maze'] != configs[row['task']]['maze']):
                    raise ValueError(f'wrong or duplicate S+1 cost source: {key}')
                values = {key: float(row[key]) for key in ('allocation_cpu_seconds', 'process_cpu_seconds_wait4')}
                if any(not math.isfinite(value) or value <= 0 for value in values.values()):
                    raise ValueError(f'invalid archived CPU time: {row["task"]}')
                full[key] = values
    if set(full) != expected:
        raise ValueError('missing or extra S+1 cost records')
    checkpoints = {}
    for task, config in configs.items():
        original = paths[task, 42].parent
        resources = original / 'resources.jsonl'
        completion = read(original / 'completion.json')
        if checksum(resources) != completion['sha256']['resources.jsonl']:
            raise ValueError(f'archived S+1 resources changed for {task}')
        rows = [json.loads(line) for line in resources.read_bytes().splitlines()]
        if [row['nominal_step'] for row in rows] != config['targets']:
            raise ValueError(f'archived resources do not match checkpoints for {task}')
        checkpoints[task] = {row['nominal_step']: row['process_cpu_seconds_at_observation'] for row in rows}
    return full, checkpoints


def pilot_costs(root, configurations, ablation_cost):
    summary = read(root / 'accounting-summary.json')
    ledger = read(root / 'ledger.json')
    plan = load_plan(root / 'plan.json', (root / 'plan.sha256').read_text().strip())
    if (summary['stopped'] or summary['reserved_seconds'] or summary['states'] != {'COMPLETED': 84}
            or len(plan['runs']) != 84):
        raise ValueError('a full forecast requires 84 terminal, complete pilot cells')
    if summary['total_charged_cpu_seconds'] > plan['limit_seconds']:
        raise ValueError('pilot allocation CPU exceeded its authorization')
    records = {row['JobIDRaw']: row for row in allocation_rows((root / 'accounting.psv').read_text())}
    by_job = {row['JobID']: row for row in records.values()}
    attempts = []
    cells = {}
    for batch in ledger:
        if batch['upper_bound_seconds'] > plan['limit_seconds']:
            raise ValueError('pilot admission exceeded the authorization')
        for array_index, run in enumerate(batch['runs']):
            job = f'{batch["array_id"]}_{array_index}'
            record = by_job[job]
            allocated = float(record['CPUTimeRAW'])
            config = plan['configurations'][run['configuration_id']]
            task, agent = config['task'], run['agent']
            if allocated > run['limit_seconds'] or record['AllocCPUS'] != '1' or record['State'] != 'COMPLETED':
                raise ValueError(f'pilot accounting differs from registration: {job}')
            directory = root / run['relative_directory']
            completion = read(directory / 'completion.json')
            if (completion['exit_code'] or completion['failure']
                    or any(checksum(directory / name) != digest for name, digest in completion['sha256'].items())):
                raise ValueError(f'pilot completion is invalid: {job}')
            attempts.append(dict(
                task=task, maze=configurations[task]['maze'], arm=agent, job_id=job,
                job_id_raw=record['JobIDRaw'], state=record['State'], cpu_reservation_seconds=run['limit_seconds'],
                allocated_cpu_seconds=allocated, process_cpu_seconds_wait4=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                peak_rss_kib=completion['peak_rss_kib'], last_checkpoint=config['targets'][-1],
            ))
            points = []
            with (directory / 'rows.jsonl').open() as row_file, (directory / 'resources.jsonl').open() as cost_file:
                for line, sample, target in zip(row_file, cost_file, config['targets'], strict=True):
                    row, resources = json.loads(line), json.loads(sample)
                    validate_row(row, config, agent, 42, plan['measurement_commit'], target)
                    if resources['nominal_step'] != target:
                        raise ValueError(f'pilot resources are missing checkpoint {target} for {job}')
                    cost = row['trajectory_utility']['cost']
                    process = resources['process_cpu_seconds_at_observation']
                    if not 0 < cost['cpu_seconds'] <= process:
                        raise ValueError(f'matched pilot process CPU does not contain all utility CPU for {job}')
                    points.append(dict(
                        nominal_step=target, steps=row['actual_steps'], cost=cost,
                        process_cpu=process, archive_cpu=ablation_cost[task][target],
                        retained_steps=row['replay_diagnostics']['stored_steps'],
                        tu_draws=row['trajectory_utility']['diagnostics']['draw_requests']['trajectory_utility'],
                        peak_rss_kib=resources['peak_rss_kib'],
                    ))
            if (task, agent) in cells or points[-1]['nominal_step'] < 300000:
                raise ValueError('duplicate or short pilot cell')
            cells[task, agent] = points
    if (set(cells) != set(product(configurations, ARMS))
            or len({row['job_id_raw'] for row in attempts}) != 84
            or len(records) != len(attempts)
            or sum(row['allocated_cpu_seconds'] for row in attempts) != summary['spent_seconds']):
        raise ValueError('charged allocations do not match the pilot ledger')
    return summary, attempts, cells


def csv_table(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def known_ratio(point, candidates, refresh_steps):
    utility, rate = projected_utility(point, candidates, refresh_steps)
    other = point['process_cpu'] - point['cost']['cpu_seconds']
    return (other + utility) / point['archive_cpu'], utility, rate


def forecast(root, archive, output):
    registration, inherited = ablation()
    names = read(ABLATION / 'maze-names.json')
    configs = {row['task']: dict(row, maze=names[row['task']]) for row in inherited['configurations']}
    seeds = registration['seeds']
    full, checkpoints = reference_costs(archive, configs, seeds)
    summary, attempts, pilot = pilot_costs(root, configs, checkpoints)
    output.mkdir(parents=True, exist_ok=True)
    matched, cells, totals = [], [], []
    forecasts = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0]))
    for (task, arm), points in pilot.items():
        config = configs[task]
        for point in points:
            cost = point['cost']
            matched.append(dict(
                task=task, maze=config['maze'], group=config['group'], arm=arm, nominal_step=point['nominal_step'],
                actual_steps=point['steps'], s_plus_one_process_cpu=point['archive_cpu'], process_cpu=point['process_cpu'],
                process_overhead_ratio=point['process_cpu'] / point['archive_cpu'],
                non_utility_process_cpu=point['process_cpu'] - cost['cpu_seconds'],
                evaluations_per_step=cost['evaluations'] / point['steps'],
                cpu_seconds_per_sampled_candidate=cost['cpu_seconds'] / cost['candidates_sampled'] if cost['candidates_sampled'] else None,
                scoring_cpu_seconds_per_candidate=cost['scoring_cpu_seconds'] / cost['candidates_scored'] if cost['candidates_scored'] else None,
                peak_rss_kib=point['peak_rss_kib'], retained_steps=point['retained_steps'], tu_draws=point['tu_draws'], **cost,
            ))
        recent = [point for point in points if point['nominal_step'] >= 100000]
        for k, refresh in MENU:
            ratio, utility, rate = known_ratio(points[-1], k, refresh)
            ratios = [known_ratio(point, k, refresh)[0] for point in recent]
            low, high = min(ratios), max(ratios)
            allocation_total = 0.0
            process_total = 0.0
            for seed in seeds:
                archived = full[task, seed]
                allocation = archived['allocation_cpu_seconds']
                process = archived['process_cpu_seconds_wait4']
                allocation_total += allocation * ratio
                process_total += process * ratio
                groups = [('all', 'all'), ('all', arm)]
                if task in SCOUTS:
                    groups += [(config['maze'], 'all'), (config['maze'], arm)]
                for maze, group in groups:
                    bucket = forecasts[k, refresh, maze, group][seed]
                    for i, predicted in enumerate((ratio * allocation, low * allocation, high * allocation, ratio * process)):
                        bucket[i] += predicted / 3600
            cells.append(dict(
                task=task, maze=config['maze'], group=config['group'], arm=arm, candidates=k,
                refresh_steps=refresh, matched_checkpoint=points[-1]['nominal_step'], evaluations_per_step=rate,
                utility_cpu_seconds_per_step=utility / points[-1]['steps'],
                cost_ratio=ratio, observed_checkpoint_ratio_min=low, observed_checkpoint_ratio_max=high,
                projected_allocation_cpu_hours=allocation_total / 3600, projected_process_cpu_hours=process_total / 3600,
            ))
    for (k, refresh, maze, arm), values in forecasts.items():
        estimates = [times[0] for times in values.values()]
        totals.append(dict(
            candidates=k, refresh_steps=refresh, maze=maze, arm=arm,
            projected_allocation_cpu_hours=sum(estimates),
            projected_process_cpu_hours=sum(times[3] for times in values.values()),
            checkpoint_scenario_low_cpu_hours=sum(times[1] for times in values.values()),
            checkpoint_scenario_high_cpu_hours=sum(times[2] for times in values.values()),
            minimum_paired_seed_cpu_hours=min(estimates), median_paired_seed_cpu_hours=statistics.median(estimates),
            maximum_paired_seed_cpu_hours=max(estimates),
        ))
    csv_table(output / 'pilot-attempts.csv', attempts)
    csv_table(output / 'matched-checkpoint-costs.csv', matched)
    csv_table(output / 'forecast-cells.csv', cells)
    csv_table(output / 'forecast-totals.csv', totals)
    metadata = dict(
        **summary, forecasts=len(cells), matched_checkpoints=len(matched),
        base_reference='560 complete original S+1 runs, seed 42 at all matched checkpoints',
        input_sha256={
            'pilot_files': checksum(root / 'files.sha256.json'),
            'ablation_index': checksum(archive / 'local-full-index.json'),
            'grid_run_costs': checksum(archive / 'grid-evidence/grid-run-costs.csv'),
            'reference_run_costs': checksum(archive / 'grid-evidence/reference-run-costs.csv'),
        },
        model={
            'cost_ratio': '(pilot process CPU - measured utility CPU + projected utility CPU) / S+1 process CPU',
            'matched_cpu': 'Process CPU at checkpoint observation for pilot and S+1, using scheduler samples.',
            'full_forecast': 'Apply each arm/maze ratio at its last matched checkpoint to each archived S+1 full-seed allocation/process CPU.',
            'cost_of_evaluation': 'Separate insertion and refresh. Fixed CPU per evaluation plus timed scoring CPU per candidate times observed mean candidate slots at K. The random arm uses fixed CPU only.',
            'refresh_rate': 'Keep measured insertion rate and calibrate Poisson encounter renewals to the measured F=2000 refresh rate. Hazard is TU requests over the product of retained steps and episodes inserted. Retention uses stored steps at the checkpoint. F=5000 allows at most one refresh per insertion.',
            'spread': 'Sum observed min/max ratio scenarios from matched checkpoints at 100k or later. In the point model, also report min, median and max total CPU of paired archived seed vectors. These are scenario ranges.',
            'limitations': 'Same non-utility ratio, conditional evaluation costs and retention geometry across K/F. Unmeasured training changes and late growth can move the ratios.',
        },
    )
    (output / 'cost-model.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--ablation', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(forecast(args.pilot, args.ablation, args.out), sort_keys=True))
