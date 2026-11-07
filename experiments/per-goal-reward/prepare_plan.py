import argparse
import json
import math
from pathlib import Path

from protocol import EXPERIMENT, NEW_ARMS, REFERENCES, configurations, analysis_registration
from archive import audit
from execution import identity
from freeze_references import checksum
from grid_protocol import atomic_json


def forecast(summary):
    runs = summary['runs']
    expected = {(config['id'], arm) for config in configurations() for arm in NEW_ARMS}
    if len(runs) != 12 or {(run['configuration_id'], run['agent']) for run in runs} != expected:
        raise ValueError('forecast requires twelve measured arm-maze pilot cells')
    cells = []
    for run in runs:
        cost = run['cost']
        if any(not math.isfinite(cost[key]) or cost[key] <= 0 for key in ('body_wall_seconds', 'allocation_cpu_seconds', 'peak_rss_kib')):
            raise ValueError('pilot forecast needs measured positive costs')
        cells.append(dict(configuration_id=run['configuration_id'], agent=run['agent'], remaining_seeds=19,
            expected_cpu_seconds=19 * cost['allocation_cpu_seconds'],
            limit_minutes=max(3, math.ceil((2 * cost['body_wall_seconds'] + 60) / 60))))
    limits = {(cell['configuration_id'], cell['agent']): cell['limit_minutes'] for cell in cells}
    ids = [config['id'] for config in configurations()]
    arrays = []
    for arm in NEW_ARMS:
        arrays.append(dict(kind='per_pair', agent=arm, seeds=list(range(43, 47)), configurations=ids,
            runs=24, limit_minutes=max(limits[identifier, arm] for identifier in ids)))
        for index in range(0, 6, 2):
            pair = ids[index:index + 2]
            arrays.append(dict(kind='compact', agent=arm, seeds=list(range(47, 62)), configurations=pair,
                runs=30, limit_minutes=max(limits[identifier, arm] for identifier in pair)))
    for array in arrays:
        array['reservation_cpu_seconds'] = array['runs'] * array['limit_minutes'] * 60
    pilot = summary['allocation_cpu_seconds']
    future = sum(cell['expected_cpu_seconds'] for cell in cells)
    hard = sum(array['reservation_cpu_seconds'] for array in arrays)
    return dict(cells=cells, arrays=arrays, pilot_cpu_hours=pilot / 3600,
        future_expected_cpu_hours=future / 3600, expected_total_cpu_hours=(pilot + future) / 3600,
        doubled_forecast_cpu_hours=(pilot + 2 * future) / 3600, hard_reservations_cpu_hours=(pilot + hard) / 3600,
        requested_cpu_hour_limit=math.ceil(max(pilot + 2 * future, pilot + hard) / 3600),
        memory_mib=max(512, math.ceil(3 * max(run['cost']['peak_rss_kib'] for run in runs) / 1024 / 128) * 128),
        future_runs=228, expected_rule='Nineteen copies of each measured seed42 CPUTimeRAW; conservatively retain per-pair overhead for compact runs.',
        hard_rule='max(3,ceil((2*collector wall seconds+60)/60)) per arm-maze cell; reserve each array largest cell limit for every element.',
        limitation='One cost seed does not bound seed variance. All attempts, pilot, spent allocations, remaining reservations and new hard limits must fit the separately approved cumulative CPU-hour limit.')


def prepare(root):
    root = Path(root)
    summary = audit(root)
    atomic_json(root / 'pilot-summary.json', summary)
    evidence = {name + '_sha256': checksum(root / (name + '.json'))
                for name in ('pilot-summary', 'pilot-registration', 'reference-index', 'model-index')}
    plan = dict(stage='grid_pending_approval', experiment=EXPERIMENT, **identity(),
        agents=list(NEW_ARMS), seeds=list(range(42, 62)), remaining_seeds=list(range(43, 62)),
        per_pair_seeds=list(range(42, 47)), configurations=configurations(), evaluation_blocks=['training', 'validation', 'test'],
        references=list(REFERENCES), reference_policy='Reuse original archived E/S/G/R rows without rerunning reference learning. R comparisons are post hoc.',
        values='One r and ir for every live classifier and every maze goal. Policy q*r*num, bootstrap max q*r over change-anticipating next-match rules, rewards and first-decision profiles use the current/drawn goal. Shared r and ir have no learning role.',
        initialization='Each new entry takes the base creation values: covering r=0.0 and ir=initial_ir; general initial_r and initial_ir; ALP copies all parent r/ir entries. Update flags reset on creation. An undrawn goal keeps its start or inherited values.',
        transition_model='Core ALP primitives, q, marks, numerosity, subsumption and covering are unchanged. Every original, fallback and copy receives a full model update. GA is rejected. No restore path exists.',
        randomness='Policy stream1 and sampler stream7 retain the base construction; no additional random stream.',
        counters='Every row records cumulative macroclassifier-goal reward writes by original/relabeled origin, live entries updated at least once during their own lifetime, and live classifier-goal pairs. ALP inheritance does not itself count as a reward update; removal drops live entries.',
        scientific_scope='Departs from Butz ACS2 reinforcement learning: no value generalization across goals, with a shared transition model.',
        analysis=analysis_registration(), cost_pilot=evidence, forecast=forecast(summary),
        execution=dict(clone='~/acs2-tu', target='x86_64-unknown-linux-musl', partition='lem-cpu-normal',
            tooling='Study-local adapters compose frozen experiments/grid collection, node guards, budget, accounting and artifact checks without changing protected grid modules.',
            grouping='One 24-job per-pair array and three 30-job compact arrays per arm. At most two streams and forty running jobs.',
            admission='Re-derive manifests before every sbatch; shared grant, expanded queue, scheduler test-only, node guards and STOP. All attempts, spent allocations, reservations and new hard limits within the approved cumulative CPU-hour limit.',
            accounting='CPUTimeRAW allocation CPU-seconds, sacct deduplicated by JobIDRaw. Two independently checksum-verified archive copies.'),
        stage2_deliverables=['seeds43--61', 'registered paired contrasts, separate verdicts and joint mechanism rules',
            'F and F_mod training curves for both new arms and their bases', 'results section in docs/ARCHITECTURE.md'],
        authorization='Stage1 ends after this single plan with its forecast and the external report. No seed beyond42 without explicit stage2 approval of these plan bytes and a CPU-hour limit.')
    path = Path(__file__).with_name('plan.json')
    if path.exists():
        raise ValueError('the single registration already exists')
    atomic_json(path, plan)
    atomic_json(root / 'registered-plan.json', plan)
    return dict(plan_sha256=checksum(path), forecast=plan['forecast'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(prepare(parser.parse_args().root)))
