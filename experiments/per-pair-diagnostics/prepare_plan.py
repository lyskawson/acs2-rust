import argparse
import hashlib
import json
import math
from pathlib import Path

from archive import audit
from execution import measurement_identity
from protocol import ARMS, BLOCKS, EXPERIMENT, ROOT, analysis_registration, configurations
from grid_protocol import atomic_json


def forecast(summary):
    cells = []
    for run in summary['runs']:
        cost = run['cost']
        if any(not math.isfinite(cost[key]) or cost[key] <= 0 for key in ('body_wall_seconds', 'allocation_cpu_seconds', 'peak_rss_kib')):
            raise ValueError('every cost cell must be measured')
        cells.append(dict(configuration_id=run['configuration_id'], agent=run['agent'], runs=4,
            expected_cpu_seconds=4 * cost['allocation_cpu_seconds'],
            limit_minutes=max(3, math.ceil((2 * cost['body_wall_seconds'] + 60) / 60))))
    expected_keys = {(config['id'], arm) for config in configurations() for arm in ARMS}
    if {(cell['configuration_id'], cell['agent']) for cell in cells} != expected_keys or len(cells) != 30:
        raise ValueError('forecast requires exactly thirty pilot cost cells')
    for cell in cells:
        cell['array_limit_minutes'] = max(item['limit_minutes'] for item in cells if item['configuration_id'] == cell['configuration_id'])
        cell['reservation_cpu_seconds'] = 4 * cell['array_limit_minutes'] * 60
    pilot = summary['allocation_cpu_seconds']
    expected = sum(cell['expected_cpu_seconds'] for cell in cells)
    hard = sum(cell['reservation_cpu_seconds'] for cell in cells)
    return dict(cells=cells, pilot_cpu_hours=pilot / 3600, future_expected_cpu_hours=expected / 3600,
        expected_total_cpu_hours=(pilot + expected) / 3600, doubled_forecast_cpu_hours=(pilot + 2 * expected) / 3600,
        hard_reservations_cpu_hours=(pilot + hard) / 3600,
        requested_cpu_hour_limit=math.ceil(max(pilot + 2 * expected, pilot + hard) / 3600),
        memory_mib=max(512, math.ceil(3 * max(run['cost']['peak_rss_kib'] for run in summary['runs']) / 1024 / 128) * 128),
        runs=120, arrays=6, jobs_per_array=20,
        expected_rule='Four replications of each measured seed42 CPUTimeRAW; pilot cost includes all allocations.',
        hard_rule='max(3, ceil((2 * measured collector wall seconds + 60) / 60)) per arm/maze; use the maximum across the five arms for each twenty-job maze array.',
        limitation='One cost seed does not bound other seeds. All attempts, used allocations and remaining hard reservations remain charged before each submission.')


def prepare(root):
    root = Path(root)
    _, summary = audit(root)
    atomic_json(root / 'pilot-summary.json', summary)
    evidence = dict(allocation_cpu_seconds=summary['allocation_cpu_seconds'], rows=summary['rows'],
        identity_rows=summary['identity_rows'], pairs_checked=summary['pairs_checked'],
        summary_sha256=hashlib.sha256((root / 'pilot-summary.json').read_bytes()).hexdigest(),
        references_sha256=hashlib.sha256((root / 'reference-index.json').read_bytes()).hexdigest(),
        models_sha256=hashlib.sha256((root / 'model-index.json').read_bytes()).hexdigest(),
        pilot_registration_sha256=hashlib.sha256((root / 'pilot-registration.json').read_bytes()).hexdigest())
    plan = dict(stage='grid_pending_approval', experiment=EXPERIMENT, **measurement_identity(),
        agents=list(ARMS), seeds=list(range(43, 47)), configurations=configurations(), evaluation_blocks=BLOCKS,
        per_start_diagnostics=True, runs=120, cost_pilot=evidence, forecast=forecast(summary), analysis=analysis_registration(),
        identity='Every seed43--46 row must reproduce its frozen historical trajectory in all deterministic fields outside per-pair records, IEEE754 binary64 bits. Only grid_identity.OMIT provenance/time/arm fields are excluded. STOP on any mismatch.',
        ground_truth='Public acs2-envs MazeTopology transitions, including wall self-loops; reward 1000, gamma .95. Immediate arrival pays 1000; otherwise .95 times shortest-path successor value with cap-1. V*=0 beyond cap. All actions tie when no capped route is possible. Validate every recorded distance and Bellman maximum before analysis.',
        execution=dict(tooling='experiments/grid/', dispatcher='grid_dispatch.dispatch, one-shot', partition='lem-cpu-normal',
            clone='~/acs2-tu', target='x86_64-unknown-linux-musl', source='Pinned clean measurement commit and frozen static binary; immutable operations snapshots.',
            grouping='Six twenty-job arrays, one per maze: five arms times four seeds. At most two streams and forty running jobs. No dependent queue of future arrays.',
            admission='Mechanically derive exact manifests from this registration before every sbatch; read grant and shared queue; sbatch --test-only. Pilot + used + reserved + new hard limits must fit the explicitly approved cumulative CPU-hour limit.',
            scope='One accounting scope over pilot and replication, zero pilot offset, every attempt charged once.',
            failure='Node identity guards and shared STOP on real failure; diagnose before recovery and preserve every failed attempt.',
            archive='~/tu_runs/per-pair-diagnostics/ and ~/Desktop/tu-runs/per-pair-diagnostics/, two independently verified copies.'),
        authorization='Stage1 complete after registration and external report. Seeds43--46 require explicit approval of these exact plan bytes and a cumulative CPU-hour limit; no stage2 allocations are authorized yet.')
    path = Path(__file__).with_name('plan.json')
    if path.exists():
        raise ValueError('the single registered plan already exists')
    atomic_json(path, plan)
    atomic_json(root / 'registered-plan.json', plan)
    return dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), forecast=plan['forecast'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(prepare(parser.parse_args().root)))
