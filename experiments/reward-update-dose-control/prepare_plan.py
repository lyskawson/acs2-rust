import argparse
import hashlib
import json
import math
from pathlib import Path

from protocol import ARM, REFERENCES, EXPERIMENT, configurations, analysis_registration
from archive import audit
from execution import identity
from grid_protocol import atomic_json


def forecast(summary):
    expected = {config['id'] for config in configurations()}
    runs = summary['runs']
    if len(runs) != 6 or {run['configuration_id'] for run in runs} != expected:
        raise ValueError('forecast requires six measured pilot cells')
    cells = []
    for run in runs:
        cost = run['cost']
        if any(not math.isfinite(cost[key]) or cost[key] <= 0 for key in ('body_wall_seconds','allocation_cpu_seconds','peak_rss_kib')):
            raise ValueError('pilot cost cells must be measured and positive')
        cells.append(dict(configuration_id=run['configuration_id'], agent=ARM, remaining_seeds=19,
            expected_cpu_seconds=19*cost['allocation_cpu_seconds'],
            limit_minutes=max(3,math.ceil((2*cost['body_wall_seconds']+60)/60))))
    limits = {cell['configuration_id']:cell['limit_minutes'] for cell in cells}
    ids = [config['id'] for config in configurations()]
    arrays = [dict(kind='per_pair', seeds=list(range(43,47)), configurations=ids, runs=24,
                   limit_minutes=max(limits.values()))]
    for index in range(0,6,2):
        pair = ids[index:index+2]
        arrays.append(dict(kind='compact',seeds=list(range(47,62)), configurations=pair, runs=30,
                           limit_minutes=max(limits[config] for config in pair)))
    for array in arrays:
        array['reservation_cpu_seconds'] = array['runs']*array['limit_minutes']*60
    pilot = summary['allocation_cpu_seconds']
    expected_future = sum(cell['expected_cpu_seconds'] for cell in cells)
    hard = sum(array['reservation_cpu_seconds'] for array in arrays)
    return dict(cells=cells, arrays=arrays, pilot_cpu_hours=pilot/3600,
        future_expected_cpu_hours=expected_future/3600, expected_total_cpu_hours=(pilot+expected_future)/3600,
        doubled_forecast_cpu_hours=(pilot+2*expected_future)/3600, hard_reservations_cpu_hours=(pilot+hard)/3600,
        requested_cpu_hour_limit=math.ceil(max(pilot+2*expected_future,pilot+hard)/3600),
        memory_mib=max(512,math.ceil(3*max(run['cost']['peak_rss_kib'] for run in runs)/1024/128)*128),
        future_runs=114, expected_rule='Nineteen replications of each measured seed42 CPUTimeRAW; conservatively retain per-pair overhead for compact runs.',
        hard_rule='max(3,ceil((2*collector wall seconds+60)/60)) per maze; every array reserves its largest cell limit for every job.',
        limitation='One cost seed cannot bound seed variance. All attempts, spent allocations, remaining reservations and new hard limits must fit the separately approved cumulative limit.')


def prepare(root):
    root = Path(root)
    summary = audit(root)
    atomic_json(root/'pilot-summary.json',summary)
    evidence = {name+'_sha256':hashlib.sha256((root/(name+'.json')).read_bytes()).hexdigest()
                for name in ('pilot-summary','pilot-registration','reference-index','model-index')}
    plan = dict(stage='grid_pending_approval', experiment=EXPERIMENT, **identity(),
        agents=[ARM],seeds=list(range(42,62)), remaining_seeds=list(range(43,62)),
        per_pair_seeds=list(range(42,47)), configurations=configurations(), evaluation_blocks=['training','validation','test'],
        references=list(REFERENCES), probabilities=json.loads(Path(__file__).with_name('probabilities.json').read_text()), reference_policy='Reuse original archived success rows and per-pair records; never rerun reference learning.',
        arm='Shortened full-model HER at p=2/3; independent post-ALP per-rule reward skips with maze-specific pi on actual relabeled draws only, ignoring goal attributes. Originals and fallbacks update every rule. ALP, GA, q, order and full-next-match-set bootstrap stay unchanged.',
        randomness='Independent seed-derived ChaCha8 stream 0x726577617264736b. Policy stream1 and sampler stream7 retain S construction. Skip RNG state and counters participate in read-only evaluation snapshots.',
        counters='Every row records original/relabeled macroclassifier reward updates and skips by both rule kinds, post-ALP action-set opportunities, and relabeled draws containing a goal-specific rule.',
        analysis=analysis_registration(), cost_pilot=evidence, forecast=forecast(summary),
        execution=dict(clone='~/acs2-tu', target='x86_64-unknown-linux-musl', partition='lem-cpu-normal', tooling='experiments/grid/',
            source='Pinned clean measurement commit and unchanged static binary; frozen operations per stage.',
            grouping='One 24-job diagnostic array (six mazes x seeds43--46), then three 30-job compact arrays (two mazes x seeds47--61). At most two streams and forty running jobs.',
            admission='Re-derive manifests from approved registration before each sbatch; grant, expanded queue, scheduler test-only, node guards and shared STOP. Pilot + spent + reservations + new hard limits within cumulative approved CPU-hour limit.',
            accounting='Every attempt, CPUTimeRAW allocated CPU-seconds, sacct deduplicated by JobIDRaw. Two independently checksum-verified archive copies.'),
        stage2_deliverables=['seeds43--61', 'registered paired contrasts and exact outcome/mechanism decisions',
            'F and F_mod curves over training against both references', 'results section in docs/ARCHITECTURE.md'],
        authorization='Stage1 ends after this single plan with its forecast and the external report. No seed beyond42 is authorized without explicit stage2 approval of these plan bytes and a CPU-hour limit.')
    path = Path(__file__).with_name('plan.json')
    if path.exists():
        raise ValueError('the single registration already exists')
    atomic_json(path,plan)
    atomic_json(root/'registered-plan.json',plan)
    return dict(plan_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),forecast=plan['forecast'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(prepare(parser.parse_args().root)))
