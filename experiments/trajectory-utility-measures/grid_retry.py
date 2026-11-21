from collections import Counter

from grid_limits import PILOT_CPU_SECONDS, PLAN_SHA256, cell_key, derive
from grid_protocol import load_plan
from registration import checksum


STOPPED_MANIFEST_SHA256 = '7090e614b44760e664b1da652ab48aa8e9d8675ca8bd2d04a9204a1916a56687'
RETRY_LIMIT_SECONDS = 14 * 3600
RETRY_CONCURRENCY = 40
ORIGINAL_OUTCOMES = {'full': 1546, 'partial': 10, 'not_started': 76}


def stopped_evidence(path):
    manifest = load_plan(path / 'files.partial.sha256.json', STOPPED_MANIFEST_SHA256)
    summary = load_plan(path / 'partial-summary.json', manifest['partial-summary.json'])
    if (not summary['stopped'] or summary['reserved_seconds'] or summary['plan_sha256'] != PLAN_SHA256
            or summary['pilot_cpu_seconds'] != PILOT_CPU_SECONDS
            or summary['total_cpu_seconds'] != PILOT_CPU_SECONDS + summary['grid_cpu_seconds']
            or checksum(path / 'STOP.json') != manifest['STOP.json']):
        raise ValueError('the stopped grid has different or unsettled accounting')
    return manifest, summary


def retry_plan(original, inputs_sha):
    manifest, summary = stopped_evidence(original)
    if inputs_sha != manifest['grid-inputs.json']:
        raise ValueError('retry does not address the original grid inputs')
    plan = derive(load_plan(original / 'grid-inputs.json', inputs_sha))
    outcomes = load_plan(original / 'partial-measurements.json', manifest['partial-measurements.json'])
    if Counter(row['outcome'] for row in outcomes) != ORIGINAL_OUTCOMES:
        raise ValueError('the stopped grid has different run outcomes')
    by_attempt = {row['attempt_id']: row for row in outcomes}
    if set(by_attempt) != {run['attempt_id'] for run in plan['runs']}:
        raise ValueError('stopped outcomes differ from the registered grid')
    gate = load_plan(original / 'scout-gate.json', manifest['scout-gate.json'])
    previous_limits = {cell_key(row): row['limit_seconds'] for row in gate['arrays'].values()}
    retries = []
    for run in plan['runs']:
        result = by_attempt[run['attempt_id']]
        if any(result[field] != run[field] for field in ('task', 'agent', 'seed', 'phase', 'relative_directory')):
            raise ValueError('stopped outcome has a different registered run identity')
        if result['outcome'] == 'full':
            continue
        if run['phase'] != 'conditional':
            raise ValueError('only unfinished conditional runs may be retried')
        prior = None
        if result['outcome'] == 'partial':
            rows = run['relative_directory'] + '/rows.jsonl'
            count = result['rows_seen']
            if (not 0 < count < len(plan['configs'][run['task']]['targets'])
                    or result['rows_sha256'] != manifest[rows]):
                raise ValueError('partial measurement does not match its sealed prefix')
            prior = dict(rows=rows, rows_sha256=manifest[rows], checkpoints=count)
        attempt = run['run_id'] + '_a02'
        retries.append(run | dict(phase='retry', attempt_id=attempt, previous_attempt_id=run['attempt_id'],
                                  relative_directory=f"attempts/{attempt}/measurement/{run['run_id']}",
                                  limit_seconds=RETRY_LIMIT_SECONDS, prior_limit_seconds=previous_limits[cell_key(run)],
                                  previous=prior))
    return plan | dict(runs=retries, earlier_grid_cpu_seconds=summary['grid_cpu_seconds'],
                       earlier_completed_runs=ORIGINAL_OUTCOMES['full'])


def retry_batches(runs):
    ranked = sorted(runs, key=lambda run: (-run['prior_limit_seconds'], run['run_id']))
    result = []
    for start in range(0, len(ranked), RETRY_CONCURRENCY):
        group = ranked[start:start + RETRY_CONCURRENCY]
        result.append(dict(name=f'retry-{len(result)}', phase='retry', limit_seconds=RETRY_LIMIT_SECONDS,
                           throttle=len(group), jobs=[dict(runs=[run]) for run in group],
                           dependencies=[result[-1]['name']] if result else []))
    return result
