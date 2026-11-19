import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from registration import ABLATION, ARMS, HERE, ablation, checksum, pilot_registration, read
from forecast import SCOUTS, reference_costs
from grid_protocol import atomic_json, load_plan


PLAN_SHA256 = 'b1a174ed1b875c076decf177acadecae53a228ad5707011e425ec42db96be29d'
PILOT_SHA256 = '40ae14d66e218b3f93348ca0dc8d4f2ccdf5a7748b521f8143ece41fc037b9f7'
FIRST_SEED = 42
LIMIT_HOURS = 2000
PILOT_CPU_SECONDS = 33177
SCOUT_MARGIN = 1.30
ORDINARY_MARGIN = 1.50
SCOUT_GATE_LIMIT = 600


def approved_registration():
    study, inherited = ablation()
    if checksum(HERE / 'plan.json') != PLAN_SHA256:
        raise ValueError('final trajectory-utility registration SHA-256 differs')
    selected = study['utility']['selected']
    if (selected['K'], selected['F']) != (8, 2000):
        raise ValueError('selected utility settings differ from decision 61')
    config = {cfg['task']: dict(cfg, trajectory_utility={'candidates': selected['K'], 'refresh_steps': selected['F']})
              for cfg in inherited['configurations']}
    return study, inherited['preset'], config


def pilot_inputs(pilot, ablation_archive, costs):
    pilot_manifest = load_plan(pilot / 'files.sha256.json', PILOT_SHA256)
    summary = load_plan(pilot / 'accounting-summary.json', pilot_manifest['accounting-summary.json'])
    if (summary['total_charged_cpu_seconds'], summary['reserved_seconds'], summary['states'].get('COMPLETED')) != (
            PILOT_CPU_SECONDS, 0, 84):
        raise ValueError('sealed pilot charges or completions differ')
    study, _, configs = approved_registration()
    names = read(ABLATION / 'maze-names.json')
    named = {task: dict(cfg, maze=names[task]) for task, cfg in configs.items()}
    source_digests = read(costs / 'cost-model.json')['input_sha256']
    originals = {
        'ablation_index': ablation_archive / 'local-full-index.json',
        'grid_run_costs': ablation_archive / 'grid-evidence/grid-run-costs.csv',
        'reference_run_costs': ablation_archive / 'grid-evidence/reference-run-costs.csv',
    }
    if (source_digests['pilot_files'] != PILOT_SHA256
            or any(checksum(path) != source_digests[key] for key, path in originals.items())):
        raise ValueError('stage-one cost inputs differ')
    reference, matched = reference_costs(ablation_archive, named, study['seeds'])
    _, _, _, pilot_runs = pilot_registration()
    known = {run['attempt_id']: run for run in pilot_runs}
    pilot_accounts = load_plan(pilot / 'attempt-accounting.json', pilot_manifest['attempt-accounting.json'])
    evidence = {}
    for row in pilot_accounts:
        if row['state'] != 'COMPLETED':
            continue
        original = known[row.get('previous_attempt_id', row['attempt_id'])]
        task = next(task for task, cfg in configs.items() if original['configuration_id'] == cfg['id'])
        key = task, original['agent']
        if key in evidence:
            raise ValueError('duplicate completed pilot cell')
        rows = row['relative_directory'] + '/rows.jsonl'
        completion = rows.replace('/rows.jsonl', '/completion.json')
        complete = load_plan(pilot / completion, pilot_manifest[completion])
        if complete['sha256']['rows.jsonl'] != pilot_manifest[rows]:
            raise ValueError('pilot rows differ from the completion manifest')
        with (pilot / rows).open() as stream:
            points = {item['nominal_step']: item['trajectory_utility']['cost']['cpu_seconds']
                      for item in map(json.loads, stream)}
        resource_path = rows.replace('/rows.jsonl', '/resources.jsonl')
        if checksum(pilot / resource_path) != pilot_manifest[resource_path]:
            raise ValueError('pilot resources differ from the archived file')
        with (pilot / resource_path).open() as stream:
            profiles = []
            for item in map(json.loads, stream):
                step = item['nominal_step']
                profiles.append(dict(step=step, process=item['process_cpu_seconds_at_observation'],
                                     utility=points[step], reference=matched[task][step]))
        evidence[key] = dict(task=task, agent=original['agent'], rows=rows,
                             rows_sha256=pilot_manifest[rows], points=profiles)
        if checksum(pilot / rows) != pilot_manifest[rows]:
            raise ValueError('pilot rows differ from the archived file')
    return dict(pilot_manifest_sha256=PILOT_SHA256, final_plan_sha256=PLAN_SHA256,
                source_sha256=source_digests,
                archive={task: {str(seed): reference[task, seed]['allocation_cpu_seconds'] for seed in study['seeds']}
                         for task in configs},
                pilot=list(evidence.values()))


def ratios(points):
    latest = points[-1]
    process, utility, reference = (latest[key] for key in ('process', 'utility', 'reference'))
    if not (math.isfinite(process) and math.isfinite(utility) and math.isfinite(reference)
            and reference > 0 and 0 <= utility < process):
        raise ValueError('invalid matched CPU times')
    return process / reference, (process - utility) / reference, utility / process, utility / reference


def minute_limit(seconds):
    return 60 * math.ceil(seconds / 60)


def ordinary_limit(points, full, reference_max):
    total, nonutility, share, relative_utility = ratios(points)
    ratio = total if full else max(1, nonutility) + relative_utility
    return minute_limit(ORDINARY_MARGIN * ratio * reference_max + 120)


def scout_limit(points, reference_seed42):
    _, nonutility, utility_share, _ = ratios(points)
    return minute_limit(SCOUT_MARGIN * max(1, nonutility) * (1 + utility_share) * reference_seed42 + 180)


def conditional_limit(scout_allocation_cpu, spread):
    return minute_limit(scout_allocation_cpu * spread + 60)


def cell_key(run):
    return run['task'], run['agent']


def derive(inputs):
    study, preset, configs = approved_registration()
    settings, _, _, _ = pilot_registration()
    seeds = study['seeds']
    evidence = {cell_key(p): p for p in inputs['pilot']}
    reference = inputs['archive']
    if (inputs['final_plan_sha256'] != PLAN_SHA256 or inputs['pilot_manifest_sha256'] != PILOT_SHA256
            or len(inputs['pilot']) != 84 or set(evidence) != {(task, arm) for task in configs for arm in ARMS}
            or set(reference) != set(configs)):
        raise ValueError('grid inputs differ from the registered study')
    runs, reused = [], []
    for task, cfg in configs.items():
        if set(reference[task]) != set(map(str, seeds)) or min(reference[task].values()) <= 0:
            raise ValueError('archive lacks the registered seed costs')
        maximum = max(reference[task].values())
        seed42 = reference[task][str(FIRST_SEED)]
        spread = maximum / seed42
        last = settings['tasks'][task]['last_target']
        full = last == cfg['targets'][-1]
        for arm in ARMS:
            pilot = evidence[task, arm]
            points = pilot['points']
            if [point['step'] for point in points] != [step for step in cfg['targets'] if step <= last]:
                raise ValueError('pilot cost checkpoints differ from registration')
            if task in SCOUTS:
                limit = scout_limit(points, seed42)
                envelope = conditional_limit(limit, spread)
            else:
                limit = ordinary_limit(points, full, maximum)
            for seed in seeds:
                row = f"{cfg['id']}_{arm}_s{seed}"
                base = dict(task=task, agent=arm, seed=seed, per_pair=seed in study['per_pair_seeds'],
                            configuration_id=cfg['id'], run_id=row)
                if seed == FIRST_SEED and full:
                    reused.append(base | dict(pilot=pilot))
                    continue
                phase = ('scout' if seed == FIRST_SEED else 'conditional') if task in SCOUTS else 'ordinary'
                attempt = row + '_a01'
                runs.append(base | dict(phase=phase, attempt_id=attempt,
                                        relative_directory=f'attempts/{attempt}/measurement/{row}',
                                        limit_seconds=envelope if phase == 'conditional' else limit,
                                        archive_spread=spread if task in SCOUTS else None,
                                        pilot=pilot if seed == FIRST_SEED else None))
    counts = {phase: sum(run['phase'] == phase for run in runs) for phase in ('ordinary', 'scout', 'conditional')}
    if counts != {'ordinary': 1512, 'scout': 6, 'conditional': 114} or len(reused) != 48:
        raise ValueError('incorrect scout, full-grid or reuse coverage')
    return dict(preset=preset, configs=configs, runs=runs, reused=reused, pilot_cpu_seconds=PILOT_CPU_SECONDS)


def pack_ordinary(runs, capacity=5400):
    bundles = []
    for run in sorted(runs, key=lambda value: (-value['limit_seconds'], value['run_id'])):
        size = run['limit_seconds']
        for bundle in bundles:
            if bundle['run_limits_seconds'] + size <= capacity:
                bundle['runs'].append(run)
                bundle['run_limits_seconds'] += size
                break
        else:
            bundles.append(dict(run_limits_seconds=size, runs=[run]))
    for bundle in bundles:
        bundle['limit_seconds'] = 300 * math.ceil((bundle['run_limits_seconds'] + 60) / 300)
    return bundles


def batches(plan):
    partitions = defaultdict(list)
    for run in plan['runs']:
        partitions[run['phase']].append(run)
    result = []
    scouts = []
    for index, run in enumerate(partitions['scout']):
        name = f'scout-{index}'
        scouts.append(name)
        result.append(dict(name=name, phase='scout', limit_seconds=run['limit_seconds'], throttle=1,
                           jobs=[dict(runs=[run])], dependencies=[]))
    by_limit = defaultdict(list)
    for bundle in pack_ordinary(partitions['ordinary']):
        by_limit[bundle['limit_seconds']].append(bundle)
    previous = []
    for index, (limit, jobs) in enumerate(sorted(by_limit.items(), reverse=True)):
        name = f'ordinary-{index}'
        result.append(dict(name=name, phase='ordinary', limit_seconds=limit, throttle=min(len(jobs), 34),
                           jobs=jobs, dependencies=previous))
        previous = [name]
    result.append(dict(name='scout-gate', phase='gate', limit_seconds=SCOUT_GATE_LIMIT, throttle=1,
                       jobs=[dict(runs=[])], dependencies=scouts))
    grouped = defaultdict(list)
    for run in partitions['conditional']:
        grouped[cell_key(run)].append(run)
    lanes = [previous + ['scout-gate'], previous + ['scout-gate']]
    for index, runs in enumerate(sorted(grouped.values(), key=lambda rows: -rows[0]['limit_seconds'])):
        name = f'conditional-{index}'
        lane = index % 2
        result.append(dict(name=name, phase='conditional', limit_seconds=runs[0]['limit_seconds'], throttle=19,
                           jobs=[dict(runs=[run]) for run in runs], dependencies=lanes[lane]))
        lanes[lane] = [name]
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--ablation', type=Path, required=True)
    parser.add_argument('--costs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    inputs = pilot_inputs(args.pilot, args.ablation, args.costs)
    plan = derive(inputs)
    atomic_json(args.out, inputs)
    allocations = batches(plan)
    print(json.dumps(dict(runs=len(plan['runs']), reused=len(plan['reused']), arrays=len(allocations),
        allocation_count=sum(len(array['jobs']) for array in allocations),
        upper_bound_cpu_hours=(PILOT_CPU_SECONDS +
            sum(len(array['jobs']) * array['limit_seconds'] for array in allocations)) / 3600)))
