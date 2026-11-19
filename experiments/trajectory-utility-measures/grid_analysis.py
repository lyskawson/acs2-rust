import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import statistics

from registration import ABLATION, ARMS, MEAN, RANDOM, S1, TRIAL, read
from agent_names import canonical_run_name
from forecast import csv_table, reference_costs
from grid_limits import PILOT_SHA256, PLAN_SHA256, approved_registration, derive
from grid_protocol import load_plan
from grid_run import MEASUREMENT_COMMIT, evidence_sha


ANALYSIS_ARMS = (*ARMS, S1)
MEASURES = ('primary', 'final')
GROUPS = ('no_aliasing', 'aliasing')


def interval(values, critical):
    if len(values) != 20 or not all(math.isfinite(value) for value in values):
        raise ValueError('twenty finite paired seed values are required')
    mean = statistics.mean(values)
    se = statistics.stdev(values) / math.sqrt(len(values))
    return dict(n=20, mean=mean, se=se, ci_low=mean - critical * se, ci_high=mean + critical * se)


def verdict(positive, negative, count):
    half = (count + 1) // 2
    if not count:
        return 'not assessed'
    if positive and negative:
        return 'mixed'
    if positive >= half:
        return 'improves'
    if negative >= half:
        return 'worsens'
    return 'no clear effect'


def calculate(study, configs, names, inputs, easy):
    registered = study['analysis']
    critical = registered['critical_value']
    seeds = study['seeds']
    result = defaultdict(list)
    for task, cfg in configs.items():
        common = dict(task=task, maze=names[task], group=cfg['group'])
        final = {}
        scores = {}
        for arm in ANALYSIS_ARMS:
            vectors = []
            for seed in seeds:
                success = inputs[task, arm, seed]
                if len(success) != len(cfg['targets']) or not all(0 <= rate <= 1 + 1e-12 for rate in success):
                    raise ValueError('missing or invalid registered test checkpoint')
                vectors.append(success)
                score = dict(primary=statistics.mean(success), final=success[-1])
                scores[arm, seed] = score
                result['seed-scores'].append(dict(common, arm=arm, seed=seed, **score))
            for measure in MEASURES:
                summary = interval([scores[arm, seed][measure] for seed in seeds], critical)
                result['arm-success'].append(dict(common, arm=arm, measure=measure, **summary))
                if measure == 'final':
                    final[arm] = summary
            means = [statistics.mean(point) for point in zip(*vectors)]
            for target, mean in zip(cfg['targets'], means):
                result['curves'].append(dict(common, arm=arm, nominal_step=target, n=20, mean=mean))
            sustained = next((target for i, target in enumerate(cfg['targets']) if all(value >= .95 for value in means[i:])),
                             'not reached')
            result['sustained-095'].append(dict(common, arm=arm, sustained_095=sustained))
        floor_spec = cfg['registration']['blocks'][registered['block']]
        floor_value = Fraction(floor_spec.get('random_floor_exact', str(floor_spec['random_floor'])))
        floor = max(Fraction(final[arm]['ci_low']) for arm in ANALYSIS_ARMS) <= floor_value
        result['labels'].append(dict(common, floor=floor, too_easy=easy[task],
                                     random_floor=float(floor_value), reachable_within_cap=floor_spec['reachable_within_cap']))
        for favorable, reference in registered['contrasts']:
            ceiling = all(final[arm]['mean'] >= .99 for arm in (favorable, reference))
            contrast = f'{favorable} - {reference}'
            for measure in MEASURES:
                differences = [scores[favorable, seed][measure] - scores[reference, seed][measure] for seed in seeds]
                for seed, value in zip(seeds, differences):
                    result['paired-differences'].append(dict(common, contrast=contrast, measure=measure,
                                                              seed=seed, difference=value))
                counted = not (floor or measure == 'primary' and easy[task] or measure == 'final' and ceiling)
                result['comparisons'].append(dict(common, contrast=contrast, measure=measure, favorable=favorable,
                                                 reference=reference, **interval(differences, critical), floor=floor,
                                                 too_easy=easy[task], ceiling=ceiling, counted=counted))
    for favorable, reference in registered['contrasts']:
        contrast = f'{favorable} - {reference}'
        for measure in MEASURES:
            for group in (*GROUPS, 'all'):
                rows = [row for row in result['comparisons'] if row['contrast'] == contrast and row['measure'] == measure
                        and row['counted'] and (group == 'all' or row['group'] == group)]
                positive = sum(row['ci_low'] > 0 for row in rows)
                negative = sum(row['ci_high'] < 0 for row in rows)
                item = dict(group=group, contrast=contrast, measure=measure, A_plus=positive, A_minus=negative, n=len(rows))
                if group != 'all':
                    item['verdict'] = verdict(positive, negative, len(rows))
                result['grouped'].append(item)
    primary = [row['verdict'] for row in result['grouped'] if row['contrast'] == f'{TRIAL} - {MEAN}'
               and row['measure'] == 'primary' and row['group'] in GROUPS]
    carry = TRIAL if 'improves' in primary and 'worsens' not in primary else MEAN
    return dict(result), carry


def registered_path(pilot, grid, run, manifests):
    if 'relative_directory' in run:
        relative = run['relative_directory'] + '/rows.jsonl'
        return grid / relative, manifests['grid'][relative], 'grid'
    relative = run['pilot']['rows']
    if manifests['pilot'][relative] != run['pilot']['rows_sha256']:
        raise ValueError('full reused pilot run differs from the source manifest')
    return pilot / relative, run['pilot']['rows_sha256'], 'pilot'


def rows_from(path, digest, key, config, label, process_cpu):
    if evidence_sha(path) != digest:
        raise ValueError(f'row checksum differs: {path}')
    observations, successes = [], []
    resource_path = path.with_name('resources.jsonl')
    resource_rows = [json.loads(line) for line in resource_path.read_text().splitlines()] if resource_path.exists() else []
    checkpoints = {row['nominal_step']: row for row in resource_rows}
    targets = []
    with path.open() as stream:
        for line in stream:
            raw = json.loads(line)
            actual = raw['task'], canonical_run_name(raw['agent']), raw['seed']
            if actual != key or label != 'reference' and (raw['commit'] != MEASUREMENT_COMMIT or raw['source_state'] != 'clean'):
                raise ValueError(f'row identity differs: {path}')
            step = raw['nominal_step']
            targets.append(step)
            blocks = raw['evaluation_blocks']
            if (set(blocks) != set(config['evaluation_blocks']) or
                    label != 'reference' and any(bool(block['starts']) != (42 <= key[2] <= 46) for block in blocks.values())):
                raise ValueError('registered evaluation block or per-pair seed differs')
            successes.append(blocks['test']['success'])
            utility = raw.get('trajectory_utility', {})
            diagnostics = utility.get('diagnostics', {})
            resources = checkpoints.get(step, {})
            row = dict(task=key[0], arm=key[1], seed=key[2], nominal_step=step, source=label,
                       knowledge=raw.get('knowledge'), population_classifiers=raw['population_classifiers'],
                       population_numerosity=raw['population_numerosity'], population_logical_bytes=raw['population_logical_bytes'],
                       wall_seconds_train=raw['wall_seconds_train'], wall_seconds_eval=raw['wall_seconds_eval'],
                       process_cpu_seconds_at_observation=resources.get('process_cpu_seconds_at_observation'),
                       full_process_cpu_seconds=process_cpu if step == config['targets'][-1] else None)
            row.update({f'tu_{k}': v for k, v in utility.get('cost', {}).items()})
            for name, value in diagnostics.items():
                if isinstance(value, dict):
                    row.update({f'tu_{name}_{k}': v for k, v in value.items()})
                else:
                    row['tu_' + name] = value
            observations.append(row)
    if targets != config['targets']:
        raise ValueError(f'incomplete checkpoint sequence: {path}')
    return successes, observations


def read_labels(path, configs, names):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != len(configs) or {row['task'] for row in rows} != set(configs):
        raise ValueError('ablation ACS2 labels do not cover all registered mazes')
    labels = {}
    for row in rows:
        task = row['task']
        if row['maze'] != names[task] or row['group'] != configs[task]['group'] or row['too_easy'] not in ('True', 'False'):
            raise ValueError('invalid archival ACS2 label')
        labels[task] = row['too_easy'] == 'True'
    return labels


def measurement_step(path, manifest, grid, records):
    step_path = path.parents[2] / 'step.json'
    step = load_plan(step_path, manifest[str(step_path.relative_to(grid))])
    job_id = f"{step['job_id']}.{step['step_id']}"
    raw_id = f"{step['job_id_raw']}.{step['step_id']}"
    record = records[raw_id]
    cpu = float(record['CPUTimeRAW'])
    if (record['JobID'] != job_id or record['State'] != 'COMPLETED' or record['ExitCode'] != '0:0'
            or record['AllocCPUS'] != '1' or not 0 < cpu <= step['limit_seconds']):
        raise ValueError(f'measurement step does not match its accepted allocation: {path}')
    return job_id, raw_id, cpu


def analyze(pilot, grid, archive, labels, output, grid_manifest_sha):
    study, _, configs = approved_registration()
    names = read(ABLATION / 'maze-names.json')
    manifests = dict(pilot=load_plan(pilot / 'files.sha256.json', PILOT_SHA256),
                     grid=load_plan(grid / 'files.sha256.json', grid_manifest_sha))
    stage = load_plan(grid / 'grid-inputs.json', manifests['grid']['grid-inputs.json'])
    registered_grid = derive(stage)
    summary = load_plan(grid / 'accounting-summary.json', manifests['grid']['accounting-summary.json'])
    if (summary['submitted_runs'] != len(registered_grid['runs']) or summary['stopped']
            or summary['reserved_seconds'] or sum(summary['completed_runs'].values()) != len(registered_grid['runs'])):
        raise ValueError('analysis requires a complete successful grid')
    accounting_path = grid / 'accounting.psv'
    if evidence_sha(accounting_path) != manifests['grid']['accounting.psv']:
        raise ValueError('Slurm accounting differs from the sealed grid')
    with accounting_path.open() as stream:
        records = {row['JobIDRaw']: row for row in csv.DictReader(stream, delimiter='|')}
    pilot_accounting = load_plan(pilot / 'attempt-accounting.json', manifests['pilot']['attempt-accounting.json'])
    pilot_slots = {row['relative_directory']: row for row in pilot_accounting if row['state'] == 'COMPLETED'}
    named = {task: dict(cfg, maze=names[task]) for task, cfg in configs.items()}
    archive_costs, _ = reference_costs(archive, named, study['seeds'])
    entries = load_plan(archive / 'local-full-index.json', stage['source_sha256']['ablation_index'])
    earlier = {(e['task'], S1, e['seed']): e for e in entries if e['agent'] == S1}
    expected = {(task, arm, seed) for task in configs for arm in ANALYSIS_ARMS for seed in study['seeds']}
    if set(earlier) != {key for key in expected if key[1] == S1}:
        raise ValueError('the S+1 index differs from the registered reference')
    inputs, observations, run_costs = {}, [], []
    prefix_count = 0
    index = []
    measured_steps = set()
    for run in (*registered_grid['runs'], *registered_grid['reused']):
        key = run['task'], run['agent'], run['seed']
        path, digest, label = registered_path(pilot, grid, run, manifests)
        completion_path = path.with_name('completion.json')
        relative_completion = str(completion_path.relative_to(grid if label == 'grid' else pilot))
        completion = load_plan(completion_path, manifests[label][relative_completion])
        process_cpu = completion['user_cpu_seconds'] + completion['system_cpu_seconds']
        if (completion['exit_code'] or completion['failure'] or completion['sha256']['rows.jsonl'] != digest or key in inputs):
            raise ValueError('invalid or duplicate completed run')
        if label == 'grid' and run['pilot']:
            proof = path.with_name('prefix-identity.json')
            identity = load_plan(proof, manifests[label][str(proof.relative_to(grid))])
            if identity['pilot_sha256'] != run['pilot']['rows_sha256'] or identity['differences']:
                raise ValueError('seed42 overlap did not reproduce the pilot')
            prefix_count += identity['rows']
        inputs[key], secondary = rows_from(path, digest, key, configs[key[0]], label, process_cpu)
        observations.extend(secondary)
        if label == 'grid':
            job, raw_id, cpu = measurement_step(path, manifests['grid'], grid, records)
            if raw_id in measured_steps:
                raise ValueError('measurements share a Slurm step')
            measured_steps.add(raw_id)
        else:
            certified = pilot_slots[str(path.parent.relative_to(pilot))]
            job, raw_id, cpu = certified['job_id'], certified['job_id_raw'], certified['allocated_cpu_seconds']
        index.append(dict(task=key[0], agent=key[1], seed=key[2], source=str(path), sha256=digest, kind=label))
        run_costs.append(dict(task=key[0], arm=key[1], seed=key[2], source=label,
                              process_cpu_seconds=process_cpu, peak_rss_kib=completion['peak_rss_kib'],
                              accounting_id=job, accounting_id_raw=raw_id, measurement_step_or_allocation_cpu_seconds=cpu))
    for key, row in earlier.items():
        path, digest = Path(row['source']), row['sha256']
        process = archive_costs[key[0], key[2]]['process_cpu_seconds_wait4']
        inputs[key], secondary = rows_from(path, digest, key, configs[key[0]], 'reference', process)
        observations.extend(secondary)
        index.append(dict(row, kind='reference'))
        allocation = archive_costs[key[0], key[2]]['allocation_cpu_seconds']
        run_costs.append(dict(task=key[0], arm=key[1], seed=key[2], source='reference',
                              process_cpu_seconds=process, peak_rss_kib=read(path.with_name('completion.json'))['peak_rss_kib'],
                              accounting_id=None, accounting_id_raw=None, measurement_step_or_allocation_cpu_seconds=allocation))
    if set(inputs) != expected or prefix_count != 216 or len(measured_steps) != len(registered_grid['runs']):
        raise ValueError('registered runs or seed42 prefix comparisons are incomplete')
    results, carry = calculate(study, configs, names, inputs, read_labels(labels, configs, names))
    cpu = {arm: sum(row['process_cpu_seconds'] for row in run_costs if row['arm'] == arm)
           for arm in ANALYSIS_ARMS}
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in results.items():
        csv_table(output / (name + '.csv'), rows)
    csv_table(output / 'secondary-checkpoints.csv', observations)
    csv_table(output / 'run-costs.csv', run_costs)
    (output / 'full-index.json').write_text(json.dumps(index, indent=2) + '\n')
    summary = dict(plan_sha256=PLAN_SHA256, grid_manifest_sha256=grid_manifest_sha, full_runs=len(inputs),
                   new_runs=len(registered_grid['runs']), reused_pilot_runs=len(registered_grid['reused']),
                   reference_runs=len(earlier), overlap_checkpoints=prefix_count, carry_forward=carry,
                   trial_to_mean_process_cpu_ratio=cpu[TRIAL] / cpu[MEAN], process_cpu_seconds=cpu,
                   registered=True, labels_sha256=evidence_sha(labels), accounting=summary)
    (output / 'registered-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def figures(curves, output):
    import success_curves
    _, _, configs = approved_registration()
    names = read(ABLATION / 'maze-names.json')
    success_curves.ARMS = ANALYSIS_ARMS
    success_curves.LABELS = {MEAN: 'Mean value', TRIAL: 'Trial update', RANDOM: 'Random goal', S1: 'S+1'}
    success_curves.COLORS = {MEAN: '#2289b5', TRIAL: '#875bb6', RANDOM: '#d68b32', S1: '#cf493e'}
    success_curves.MARKERS = {MEAN: 'o', TRIAL: '^', RANDOM: 's', S1: 'P'}
    return success_curves.plot(curves, output, list(configs.values()), names)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    aggregate = actions.add_parser('analyze')
    for argument in ('pilot', 'grid', 'archive', 'labels', 'out'):
        aggregate.add_argument(f'--{argument}', type=Path, required=True)
    aggregate.add_argument('--grid-manifest-sha256', required=True)
    plot = actions.add_parser('figures')
    plot.add_argument('--curves', type=Path, required=True)
    plot.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'analyze':
        result = analyze(args.pilot, args.grid, args.archive, args.labels, args.out, args.grid_manifest_sha256)
    else:
        result = figures(args.curves, args.out)
    print(json.dumps(result, sort_keys=True))
