import argparse
import csv
import json
import math
from pathlib import Path

from protocol import HERE,ARMS,SEEDS,PER_PAIR_SEEDS,configurations,new_cells,analysis_registration
from execution import checksum,require_authorization,balance
from schema import validate_sequence


def forecast(runs,pilot_seconds):
    expected={(c['id'],a) for c,a in new_cells()}
    if len(runs)!=138 or {(r['configuration_id'],r['agent']) for r in runs}!=expected:
        raise ValueError('forecast needs all 138 measured seed42 cells')
    cells=[]
    for r in runs:
        cost=r['cost']
        if any(not math.isfinite(cost[k]) or cost[k]<=0 for k in ('allocation_cpu_seconds','body_wall_seconds','peak_rss_kib')):
            raise ValueError('positive measured CPU, wall and memory costs required')
        cells.append(dict(configuration_id=r['configuration_id'],task=r['task'],agent=r['agent'],remaining_seeds=19,
            future_expected_cpu_seconds=19*cost['allocation_cpu_seconds'],
            future_reservation_seconds=19*max(180,60*math.ceil((2*cost['body_wall_seconds']+60)/60))))
    future=sum(c['future_expected_cpu_seconds'] for c in cells)
    hard=sum(c['future_reservation_seconds'] for c in cells)
    return dict(cells=cells,pilot_cpu_hours=pilot_seconds/3600,future_expected_cpu_hours=future/3600,
        expected_total_cpu_hours=(pilot_seconds+future)/3600,doubled_forecast_cpu_hours=(pilot_seconds+2*future)/3600,
        hard_reservations_cpu_hours=(pilot_seconds+hard)/3600,
        requested_cpu_hour_limit=math.ceil(max(pilot_seconds+2*future,pilot_seconds+hard)/3600),
        future_runs=2622,pilot_runs=138,all_new_runs=2760,
        memory_mib=max(512,128*math.ceil(3*max(r['cost']['peak_rss_kib'] for r in runs)/1024/128)),
        expected_rule='Nineteen copies of each measured seed42 allocation CPUTimeRAW, retaining per-pair overhead conservatively for compact seeds.',
        hard_rule='Per cell max(180,60*ceil((2*body_wall_seconds+60)/60)); 19 future seeds; group allocations without exceeding this sum.',
        limitation='One cost seed does not bound seed variance. Approval must cover pilot, every attempt, spent allocations and remaining/new reservations.')


def summarize(root):
    root=Path(root);plan=json.loads((root/'plan.json').read_text());require_authorization(plan)
    ledger=json.loads((root/'ledger.json').read_text());state,records=balance(root,ledger)
    if state['reserved_seconds'] or state['spent_seconds']>15*3600:raise ValueError('pilot incomplete or budget exceeded')
    by_job={r['JobID']:r for r in records};runs=[]
    for batch in ledger:
        for i,r in enumerate(batch['runs']):
            account=by_job[f"{batch['array_id']}_{i}"]
            if account['State']!='COMPLETED' or account['AllocCPUS']!='1':raise ValueError('all pilot attempts must be accounted and completed')
            directory=root/r['relative_directory'];completion=json.loads((directory/'completion.json').read_text())
            if completion['exit_code'] or completion['failure'] or completion['incomplete_final_line']:raise ValueError('pilot process failed')
            for name,digest in completion['sha256'].items():
                if checksum(directory/name)!=digest:raise ValueError('pilot artifact changed')
            rows=[json.loads(l) for l in (directory/'rows.jsonl').read_text().splitlines()]
            config=next(c for c in plan['configurations'] if c['id']==r['configuration_id'])
            validate_sequence(rows,config,r['agent'],42,plan['measurement_commit'])
            runs.append(dict(configuration_id=r['configuration_id'],task=config['task'],agent=r['agent'],seed=42,
                source=str(directory/'rows.jsonl'),sha256=checksum(directory/'rows.jsonl'),
                final_entries=(rows[-1].get('replay_diagnostics') or {}).get('per_goal_rewards'),
                cost=dict(allocation_cpu_seconds=int(account['CPUTimeRAW']),body_wall_seconds=completion['elapsed_seconds'],
                    process_cpu_seconds=completion['user_cpu_seconds']+completion['system_cpu_seconds'],peak_rss_kib=completion['peak_rss_kib']),
                job_id_raw=account['JobIDRaw']))
    prediction=forecast(runs,state['spent_seconds'])
    return dict(runs=runs,allocation_cpu_seconds=state['spent_seconds'],reserved_seconds=0,forecast=prediction,
                measurement_commit=plan['measurement_commit'],binary_sha256=plan['binary_sha256'],failed_attempts=0)


def prepare(summary,reference_index_sha256):
    if not isinstance(reference_index_sha256,str) or len(reference_index_sha256)!=64:
        raise ValueError('the frozen reference index checksum is required')
    prediction=forecast(summary['runs'],summary['allocation_cpu_seconds'])
    return dict(study='per-goal-reward-ablation',stage='grid_pending_approval',arms=list(ARMS),seeds=SEEDS,
        preset=json.loads((HERE.parents[1]/'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset'],
        group_counts=dict(no_aliasing=9,aliasing=19),new_runs=2760,reference_runs=600,
        new_run_rule='All six arms on all 22 ALCS research tasks; only ACS2 r(g) on the original six tasks.',
        archive_reference_index_sha256=reference_index_sha256,
        values='Each classifier has r/ir per maze goal. Updates, bootstrap max q*r over change-anticipating rules, training and greedy q*r*num choices, and first-decision profiles use the episode goal. Shared r/ir are inert.',
        initialization='Covering r=0 and initial_ir; general initial_r/initial_ir; ALP children inherit every parent goal value and reset update flags; removal drops entries; subsumption retains owner entries.',
        transition_model='ACS2 r(g) retains the delayed ACS2 action-set update order, timing, ALP, q, marks, numerosity, subsumption and covering. GA and restore are rejected. Preset epsilon=0.8 applies only to training, evaluation stays greedy.',
        counters='Every row of each per-goal arm reports cumulative original/relabeled reward writes, live updated entries and classifier-goal pairs. Online ACS2 r(g) has no replay or copies and always zero relabeled writes.',
        execution=dict(clone='~/acs2-tu',target='x86_64-unknown-linux-musl',partition='lem-cpu-normal',max_streams=2,max_running_jobs=40,
            admission='Derive manifests from the approved plan before every submission; grant and test-only checks, source/binary/hardware/limit guards, STOP on real failure. All attempts, pilot, spent allocations and reservations fit the approved cumulative limit.',
            archives='Remote ~/tu_runs/per-goal-reward-ablation and local ~/Desktop/tu-runs/per-goal-reward-ablation; two independently checksum-verified copies.'),
        per_pair_seeds=PER_PAIR_SEEDS,remaining_seeds=list(range(43,62)),configurations=configurations(),
        analysis=analysis_registration(),forecast=prediction,measurement_commit=summary['measurement_commit'],
        registration_sha256=checksum(HERE/'registration.json'),protocol_sha256=checksum(HERE/'pair-protocol.json'),
        geometry_sha256=checksum(HERE/'geometry.json'),maze_names_sha256=checksum(HERE/'maze-names.json'),
        binary_sha256=summary['binary_sha256'],pilot_allocation_cpu_seconds=summary['allocation_cpu_seconds'],
        archive_policy='600 original archived ACS2/E/S/E+1/S+1 runs on the six original tasks; no reruns.',
        authorization='Stage1 ends here. Remaining seeds, registered result analysis and figures require explicit approval of this plan with a cumulative CPU-hour limit.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');parser.add_argument('--summary-only',action='store_true');parser.add_argument('--reference-index');args=parser.parse_args()
    root=Path(args.root);summary=summarize(root)
    (root/'pilot-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (root/'pilot-costs.csv').open('w') as stream:
        fields=['task','agent','allocation_cpu_seconds','body_wall_seconds','process_cpu_seconds','peak_rss_kib','existing_pairs','updated_entries']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for r in summary['runs']:
            entries=r['final_entries'] or {};writer.writerow(dict(task=r['task'],agent=r['agent'],**r['cost'],existing_pairs=entries.get('existing_pairs'),updated_entries=entries.get('updated_entries')))
    if not args.summary_only:
        path=HERE/'plan.json'
        if path.exists():raise ValueError('the single final plan already exists')
        if not args.reference_index:raise ValueError('reference index required for the final plan')
        path.write_text(json.dumps(prepare(summary,checksum(args.reference_index)),indent=2)+'\n')
    print(json.dumps(summary['forecast']))
