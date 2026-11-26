from collections import Counter
from pathlib import Path

from proportions import checksum
from grid_plan import BINARY_SHA256, MAX_SECONDS, MEASUREMENT_COMMIT, PILOT_CHARGE
from grid_protocol import load_plan

FIRST_MANIFEST = '0da1da2946fe5e955ac24e8a84070c36352fce9448541453bebb322dc8fbcccb'
FIRST_PLAN = 'db5c71750b28996460996d832c98cb2b8c26ff9ac6b9ca36862c2a44a3a1bdcc'
FIRST_OPERATIONS = '49cbd5c7004f5c3abab11938b09cb1ec27e34c12a1a848a46150998b9922b151'
ORIGINAL_COMPLETE = 3178
RETRY_COUNT = 77
PREFIX_COUNT = 127
CONCURRENCY = 40


def sealed_stop(root):
    manifest = load_plan(root / 'files.partial.sha256.json', FIRST_MANIFEST)
    summary = load_plan(root / 'partial-summary.json', manifest['partial-summary.json'])
    rows = load_plan(root / 'partial-measurements.json', manifest['partial-measurements.json'])
    if (summary['stage'] != 'stopped' or summary['reserved_seconds']
            or summary['final_plan_sha256'] != FIRST_PLAN
            or summary['operations_sha256'] != FIRST_OPERATIONS
            or summary['binary_sha256'] != BINARY_SHA256
            or summary['pilot_cpu_seconds'] != PILOT_CHARGE
            or summary['total_cpu_seconds'] != PILOT_CHARGE + summary['grid_cpu_seconds']
            or summary['measurement_outcomes'] != dict(full=ORIGINAL_COMPLETE, partial=20, not_started=57)
            or dict(Counter(row['outcome'] for row in rows)) != summary['measurement_outcomes']
            or summary['saved_partial_checkpoints'] != PREFIX_COUNT
            or checksum(root / 'STOP.json') != manifest['STOP.json']):
        raise ValueError('stopped grid is not sealed under the registered plan')
    return manifest, summary, rows


def registration(root, limit_seconds):
    _, summary, _ = sealed_stop(root)
    if type(limit_seconds) is not int or limit_seconds < 120 or limit_seconds % 60:
        raise ValueError('recovery needs an approved positive minute-granularity limit')
    maximum = summary['total_cpu_seconds'] + RETRY_COUNT * limit_seconds
    if maximum > MAX_SECONDS:
        raise ValueError('recovery reservations exceed the cumulative CPU-hour limit')
    return dict(stage='approved_recovery', stopped_manifest_sha256=FIRST_MANIFEST,
                final_plan_sha256=FIRST_PLAN, measurement_commit=MEASUREMENT_COMMIT,
                binary_sha256=BINARY_SHA256, run_count=RETRY_COUNT,
                wall_limit_seconds=limit_seconds, cumulative_cpu_limit_seconds=MAX_SECONDS,
                prior_cpu_seconds=summary['total_cpu_seconds'],
                maximum_reserved_and_spent_cpu_seconds=maximum)


def derive_recovery(plan, root, approved):
    if approved != registration(root, approved['wall_limit_seconds']):
        raise ValueError('recovery authorization differs from the sealed STOP')
    manifest, summary, outcomes = sealed_stop(root)
    stopped = {row['run_id']: row for row in outcomes}
    registered = {run['run_id']: run for run in plan['runs']}
    if stopped.keys() != registered.keys() or len(stopped) != len(outcomes):
        raise ValueError('stopped cells differ from the registered grid')
    retries = []
    for identifier, run in registered.items():
        row = stopped[identifier]
        fields = ('task', 'agent', 'arm', 'seed', 'tu_share', 'phase')
        if any(row[field] != run[field] for field in fields):
            raise ValueError('stopped cell identity differs from registration')
        if row['outcome'] == 'full':
            continue
        if run['phase'] != 'conditional' or run['pilot'] is not None:
            raise ValueError('only unfinished dependent cells may be submitted again')
        config = plan['configs'][run['task']]
        directory = Path('attempts') / identifier / 'measurement' / f"{config['id']}_{run['agent']}_s{run['seed']}"
        if str(directory) != row['path'] or approved['wall_limit_seconds'] <= row['prior_limit_seconds']:
            raise ValueError('recovery path or wall-clock limit differs from its stopped cell')
        previous = None
        if row['outcome'] == 'partial':
            path = str(directory / 'rows.jsonl')
            targets = config['targets']
            count = row['checkpoints']
            if (not 0 < count < len(targets) or row['last_checkpoint'] != targets[count - 1]
                    or row['rows_sha256'] != manifest[path]):
                raise ValueError('preserved retry prefix differs from its seal')
            previous = dict(path=path, sha256=manifest[path], checkpoints=count)
        elif row['checkpoints'] or row['rows_sha256'] is not None:
            raise ValueError('a cancelled slot has measurement evidence')
        retries.append(dict(run, phase='recovery', attempt_id=identifier+'_a02', previous=previous,
                            limit_seconds=approved['wall_limit_seconds'],
                            prior_limit_seconds=row['prior_limit_seconds']))
    if (len(retries) != RETRY_COUNT
            or sum(run['previous']['checkpoints'] for run in retries if run['previous']) != PREFIX_COUNT):
        raise ValueError('recovery omits a missing run or a preserved checkpoint')
    trigger = summary['trigger']['run_id']
    retries.sort(key=lambda run: (run['run_id'] != trigger, -run['prior_limit_seconds'], run['run_id']))
    return dict(plan, runs=retries, recovery=True, prior_cpu_seconds=summary['total_cpu_seconds'],
                earlier_completed_runs=ORIGINAL_COMPLETE, earlier_grid_cpu_seconds=summary['grid_cpu_seconds'])


def recovery_schedule(plan):
    result = []
    for start in range(0, len(plan['runs']), CONCURRENCY):
        runs = plan['runs'][start:start+CONCURRENCY]
        name = f"recovery-{len(result)}"
        result.append(dict(name=name, phase='recovery', limit_seconds=runs[0]['limit_seconds'],
                           jobs=[[run] for run in runs], throttle=len(runs),
                           dependencies=[result[-1]['name']] if result else []))
    return result
