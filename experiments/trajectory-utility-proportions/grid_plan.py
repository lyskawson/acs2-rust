from collections import Counter, defaultdict
from pathlib import Path
import math

from proportions import CONCURRENCY, HERE, SCOUTS, checksum, label, pilot_cells, registered

MEASUREMENT_COMMIT = '8ea47c086ba6312da98bf9be26789d7420b32809'
BINARY_SHA256 = '2ee08d8f6a0378e5dad2c1c9696d9262ee164d568b6ff879e64339496aee6ee0'
PILOT_CHARGE = 53037
MAX_SECONDS = 5000 * 3600
GRANT_SECONDS = 20000 * 3600
SCOUT_GATE_SECONDS = 600


def derive(final, pilot_manifest, pilot_summary):
    study, preset, configs, _ = registered()
    settings, _, _, cells = pilot_cells()
    if (final['stage'] != 'grid_pending_approval'
            or final['draft_sha256'] != checksum(HERE / 'plan.json')
            or final['pilot_cells_sha256'] != checksum(HERE / 'pilot-cells.json')
            or final['measurement_commit'] != MEASUREMENT_COMMIT
            or pilot_summary['measurement_commit'] != MEASUREMENT_COMMIT
            or final['binary_sha256'] != BINARY_SHA256
            or pilot_summary['binary_sha256'] != BINARY_SHA256
            or pilot_summary['total_charged_cpu_seconds'] != PILOT_CHARGE
            or final['pilot']['cpu_hours'] * 3600 != PILOT_CHARGE
            or set(pilot_summary['completed']) != {cell['run_id'] for cell in cells}):
        raise ValueError('approved grid or sealed pilot identity differs')
    timing = {(row['task'], row['arm']): row for row in final['scout_gate']['limits']}
    if len(timing) != len(cells) or len(final['scout_gate']['limits']) != len(cells):
        raise ValueError('invalid registered set of limits')
    runs, reused = [], []
    for cell in cells:
        task, agent, share = (cell[field] for field in ('task', 'agent', 'tu_share'))
        cfg = configs[task]
        arm = label(agent, share)
        if final['pilot_to_grid_arm_labels'][cell['label']] != arm:
            raise ValueError('unregistered arm label')
        limit = timing[task, arm]
        full = cell['last_target'] == cfg['targets'][-1]
        if limit['reuse_full_seed_42'] != full:
            raise ValueError('registered full-budget pilot coverage differs')
        cell_record = pilot_summary['completed'][cell['run_id']]
        source = Path(cell_record['relative_directory']) / 'rows.jsonl'
        if source.is_absolute() or '..' in source.parts:
            raise ValueError('pilot row path escapes the archive')
        pilot = dict(path=str(source), sha256=pilot_manifest[str(source)],
                     checkpoints=sum(point <= cell['last_target'] for point in cfg['targets']),
                     slurm_cpu_seconds=cell_record['charged_cpu_seconds'])
        for seed in study['seeds']:
            identifier = f"{cfg['id']}_{arm}_s{seed}"
            row = dict(run_id=identifier, task=task, agent=agent, arm=arm, tu_share=share,
                       seed=seed, last_target=cfg['targets'][-1], pilot=pilot if seed == 42 else None)
            if seed == 42 and full:
                reused.append(row)
                continue
            phase = ('scout' if seed == 42 else 'conditional') if task in SCOUTS else 'ordinary'
            key = {'scout': 'scout_limit_seconds', 'conditional': 'after_scout_initial_limit_seconds',
                   'ordinary': 'ordinary_limit_seconds'}[phase]
            deadline = limit[key]
            if type(deadline) is not int or deadline < 120 or deadline % 60:
                raise ValueError('invalid registered deadline')
            runs.append(dict(row, phase=phase, limit_seconds=deadline, seed_spread=limit['seed_spread']))
    if (Counter(row['phase'] for row in runs) != dict(ordinary=3015, scout=12, conditional=228)
            or len(reused) != 105
            or sum(row['limit_seconds'] for row in runs) != round(final['costs']['sum_of_initial_allocations_cpu_hours'] * 3600)):
        raise ValueError('registered grid coverage or reserved seconds differ')
    return dict(runs=runs, reused=reused, configs=configs, preset=preset,
                cpu_model=settings['cpu_model'], memory_mib=settings['memory_mib'])


def bundle_ordinary(runs, capacity=21_600):
    bundles = []
    for run in sorted(runs, key=lambda row: (-row['limit_seconds'], row['run_id'])):
        for bundle in bundles:
            if bundle['seconds'] + run['limit_seconds'] <= capacity:
                bundle['runs'].append(run)
                bundle['seconds'] += run['limit_seconds']
                break
        else:
            bundles.append(dict(seconds=run['limit_seconds'], runs=[run]))
    return bundles


def schedule(plan):
    groups = defaultdict(list)
    for run in plan['runs']:
        groups[run['phase']].append(run)
    result = []

    def add(name, phase, limit, capacity, jobs, deps=()):
        result.append(dict(name=name, phase=phase, limit_seconds=limit, jobs=jobs,
                           throttle=min(len(jobs), capacity), dependencies=list(deps)))

    for index, run in enumerate(groups['scout']):
        add(f'scout-{index}', 'scout', run['limit_seconds'], 1, [[run]])
    scouts = [batch['name'] for batch in result]
    ordinary = defaultdict(list)
    for bundle in bundle_ordinary(groups['ordinary']):
        seconds = 300 * math.ceil((bundle['seconds'] + 60) / 300)
        ordinary[seconds].append(bundle['runs'])
    deps = []
    for index, (seconds, jobs) in enumerate(sorted(ordinary.items(), reverse=True)):
        name = f'ordinary-{index}'
        add(name, 'ordinary', seconds, CONCURRENCY - len(scouts), jobs, deps)
        deps = [name]
    add('scout-gate', 'gate', SCOUT_GATE_SECONDS, 1, [[]], scouts)
    dependents = defaultdict(list)
    for run in groups['conditional']:
        dependents[run['task'], run['arm']].append(run)
    lanes = [deps + ['scout-gate'], deps + ['scout-gate']]
    lane_time = [0, 0]
    for index, runs in enumerate(sorted(dependents.values(), key=lambda runs: -runs[0]['limit_seconds'])):
        if len(runs) != 19 or len({run['limit_seconds'] for run in runs}) != 1:
            raise ValueError('each scout must precede exactly nineteen dependent cells')
        name = f'conditional-{index}'
        lane = min(range(len(lanes)), key=lane_time.__getitem__)
        limit = runs[0]['limit_seconds']
        add(name, 'conditional', limit, 19, [[run] for run in runs], lanes[lane])
        lanes[lane] = [name]
        lane_time[lane] += limit
    return result


def schedule_totals(batches):
    return dict(arrays=len(batches), allocations=sum(len(batch['jobs']) for batch in batches),
                reserved_seconds=sum(len(batch['jobs']) * batch['limit_seconds'] for batch in batches))
