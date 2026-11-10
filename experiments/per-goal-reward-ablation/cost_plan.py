import argparse
import csv
import json
import hashlib
from pathlib import Path

from protocol import HERE,ARMS,SEEDS,PER_PAIR_SEEDS,configurations,analysis_registration
from execution import checksum
from cost_pilot import registration,verify_prior
from cost_pilot_registration import MEASUREMENT_COMMIT,BINARY_SHA256
from grid_protocol import allocation_rows,accounting
from grid_attempts import verify_artifacts
from grid_scope import verify_operations
from schema import validate_sequence
from prefix_identity import verify as verify_prefixes
from cost_evidence import batch,export_checkpoints
from cost_forecast import forecast


def verify_copy(root):
    root=Path(root);manifest=json.loads((root/'archive-checksums.json').read_text())
    for name,digest in manifest.items():
        path=root/name
        if not path.resolve().is_relative_to(root.resolve()) or checksum(path)!=digest:
            raise ValueError('archive copy differs: '+name)
    return dict(root=str(root),files=len(manifest),manifest_sha256=checksum(root/'archive-checksums.json'))


def audit(prior,pilot):
    prior,pilot=Path(prior),Path(pilot)
    copies=[verify_copy(prior),verify_copy(pilot)]
    plan=json.loads((pilot/'plan.json').read_text())
    if (checksum(pilot/'plan.json')!=(pilot/'plan.sha256').read_text().strip()
            or plan['registration']!=registration() or plan['measurement_commit']!=MEASUREMENT_COMMIT
            or plan['binary_sha256']!=BINARY_SHA256):
        raise ValueError('pilot identity or registered scope differs')
    previous=verify_prior(prior)
    if previous['files']!=plan['prior']['files']:raise ValueError('prior frozen files differ')
    remote_study=Path(plan['result_root']).parent
    operations=pilot.parent/Path(plan['operations_manifest']).relative_to(remote_study)
    verify_operations(dict(plan,operations_manifest=str(operations)))
    records=list({r['JobIDRaw']:r for r in allocation_rows((pilot/'accounting-all.psv').read_text())}.values())
    prior_ledger=json.loads((prior/'ledger.json').read_text());ledger=json.loads((pilot/'ledger.json').read_text())
    old=accounting(prior_ledger,records);new=accounting(ledger,records)
    if old!=dict(spent_seconds=9810,reserved_seconds=0) or new['reserved_seconds'] or old['spent_seconds']+new['spent_seconds']>54000:
        raise ValueError('cumulative pilot accounting incomplete or exceeds 15 CPU-hours')
    accounts={r['JobID']:r for r in records};configs={c['id']:c for c in plan['configurations']}
    seen=set()
    for allocation in ledger:
        for i,run in enumerate(allocation['runs']):
            key=run['configuration_id'],run['agent']
            if key in seen:raise ValueError('duplicate fresh cost-pilot cell')
            seen.add(key)
            account=accounts[f"{allocation['array_id']}_{i}"]
            if account['State']!='COMPLETED' or account['AllocCPUS']!='1':raise ValueError('fresh pilot allocation incomplete or failed')
            directory=pilot/run['relative_directory'];completion=json.loads((directory/'completion.json').read_text())
            if completion['exit_code'] or completion['failure'] or completion['incomplete_final_line']:raise ValueError('fresh pilot process incomplete or failed')
            verify_artifacts(directory,completion)
            rows=[json.loads(line) for line in (directory/'rows.jsonl').read_text().splitlines()]
            validate_sequence(rows,configs[run['configuration_id']],run['agent'],42,MEASUREMENT_COMMIT)
    expected={(r['configuration_id'],r['agent']) for r in plan['registration']['pilot_runs']}
    if seen!=expected or len(seen)!=138:raise ValueError('all 138 fresh cost-pilot cells required')
    identity=verify_prefixes(prior,pilot)
    return dict(verified=True,measurement_commit=MEASUREMENT_COMMIT,binary_sha256=BINARY_SHA256,
        operations_commit=plan['operations_commit'],operations_manifest_sha256=plan['operations_manifest_sha256'],prior_cpu_seconds=old['spent_seconds'],
        fresh_cpu_seconds=new['spent_seconds'],all_attempts_cpu_seconds=old['spent_seconds']+new['spent_seconds'],
        reserved_seconds=0,prior_attempts=18,fresh_attempts=138,fresh_failures=0,
        prefix_identity=identity,copies=copies,success_summary=False)


def prepare(audited,forecast_value,reference_index_sha256,cost_evidence_sha256):
    if not audited.get('verified') or forecast_value.get('new_full_runs')!=2760:
        raise ValueError('verified complete cost pilot and full cost forecast required')
    if any(not isinstance(value,str) or len(value)!=64 for value in [reference_index_sha256,cost_evidence_sha256]):
        raise ValueError('frozen source cost evidence checksums required')
    identity=[{k:check[k] for k in ('configuration_id','agent','seed','checkpoints','learning_and_per_pair_fields_bit_identical','prior_sha256','fresh_sha256')}
        for check in audited['prefix_identity']['checks']]
    identity.sort(key=lambda r:(r['configuration_id'],r['agent']))
    identity_digest=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(study='per-goal-reward-ablation',experiment='per-goal-reward-ablation',stage='grid_pending_approval',measurement_commit=MEASUREMENT_COMMIT,
        preset=json.loads((HERE.parents[1]/'fixtures/goal_measurement_regression.json').read_text())['configurations'][0]['rows'][0]['preset'],
        binary_sha256=BINARY_SHA256,arms=list(ARMS),agents=list(ARMS),seeds=SEEDS,remaining_seeds=SEEDS,remaining_full_seeds=SEEDS,per_pair_seeds=PER_PAIR_SEEDS,
        new_runs=2760,new_full_runs=2760,archived_full_runs=600,pilot_allocation_cpu_seconds=audited['all_attempts_cpu_seconds'],configurations=configurations(),analysis=analysis_registration(),
        pilot=dict(horizon=30000,fresh_cells=138,first_wave_attempts=18,first_wave_cpu_seconds=audited['prior_cpu_seconds'],
            fresh_cpu_seconds=audited['fresh_cpu_seconds'],all_attempts_cpu_seconds=audited['all_attempts_cpu_seconds'],
            cpu_hour_limit=15,reserved_seconds=0,identical_old_prefix_cells=18,success_summary=False),
        forecast=forecast_value,archive_reference_index_sha256=reference_index_sha256,cost_evidence_sha256=cost_evidence_sha256,
        prior_archive_manifest_sha256=audited['copies'][0]['manifest_sha256'],fresh_archive_manifest_sha256=audited['copies'][1]['manifest_sha256'],
        prefix_identity_sha256=identity_digest,cost_pilot_operations_commit=audited['operations_commit'],
        cost_pilot_operations_manifest_sha256=audited['operations_manifest_sha256'],
        protocol_sha256=checksum(HERE/'pair-protocol.json'),original_registration_sha256=checksum(HERE/'registration.json'),
        cost_pilot_registration_sha256=checksum(HERE/'cost-pilot.json'),geometry_sha256=checksum(HERE/'geometry.json'),maze_names_sha256=checksum(HERE/'maze-names.json'),
        values='Classifier-local r/ir per maze goal, current-goal training/greedy policy, bootstrap and per-pair profiles. Covering r=0 and initial_ir; general initial_r/initial_ir; ALP inheritance for every goal, removal drops entries, subsumption retains owner values. Shared r/ir are inert. GA and restore are rejected.',
        model='Online ACS2 r(g) retains ACS2 delayed action-set updates, order, timing, q, marks, numerosity, ALP, covering, subsumption and truncation; fixed thesis preset. Existing five arms remain unchanged.',
        archive_policy='Reuse original ACS2/E/S/E+1/S+1 full archives on the original six mazes. The fresh 30k cost pilot is not a completed full run; every new full cell still needs seeds42--61. No continuation/restore from stopped models.',
        execution=dict(clone='~/acs2-tu',target='x86_64-unknown-linux-musl',partition='lem-cpu-normal',max_streams=2,max_running_jobs=40,
            limits='Proposed per-cell limits are in forecast.cells, grouped only at equal limits so the registered hard reservation sum is preserved.',
            admission='Before every sbatch derive manifests from approved plan bytes; check grant and test-only; retain source, binary, hardware and allocation guards. STOP on real failure. Pilot, all attempts, spent allocations and remaining/new reservations fit the separately approved cumulative CPU-hour limit.',
            archives='Results outside checkout, remote ~/tu_runs/per-goal-reward-ablation and local ~/Desktop/tu-runs/per-goal-reward-ablation; two checksum-verified copies.'),
        authorization='Stage1 ends with this plan and cost-only forecast. No full-grid run or seed beyond the cost pilot without explicit approval of plan bytes and a cumulative CPU-hour limit. Full protocols and 2M/4M budgets remain unchanged.')


def write_csv(rows,path):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prior',required=True);parser.add_argument('--pilot',required=True)
    parser.add_argument('--reference-profiles',required=True);parser.add_argument('--reference-index',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--write-plan',action='store_true');args=parser.parse_args()
    audited=audit(args.prior,args.pilot)
    evidence=json.loads(Path(args.reference_profiles).read_text())
    archives=[p for p in evidence if p['kind']=='archive_full'];longer=[p for p in evidence if p['kind']=='stopped_long']
    pilots=batch(args.pilot,'pilot_30000','accounting-all.psv')
    estimates=forecast(pilots,archives,longer,audited['all_attempts_cpu_seconds'],audited['prefix_identity'])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(audited,indent=2)+'\n')
    (out/'forecast.json').write_text(json.dumps(estimates,indent=2)+'\n')
    flat=[{k:v for k,v in cell.items() if k!='growth'} for cell in estimates['cells']]
    write_csv(flat,out/'forecast-cells.csv');export_checkpoints(pilots,out/'pilot-checkpoints.csv')
    write_csv([dict(task=p['task'],agent=p['agent'],allocation_cpu_seconds=p['allocation_cpu_seconds'],
        process_cpu_seconds_wait4=p['process_cpu_seconds_wait4'],body_wall_seconds=p['body_wall_seconds'],peak_rss_kib=p['peak_rss_kib'],
        population_classifiers=p['checkpoints'][-1]['population_classifiers'],per_goal_existing_pairs=(p['checkpoints'][-1]['per_goal_entries'] or {}).get('existing_pairs'),
        per_goal_updated_entries=(p['checkpoints'][-1]['per_goal_entries'] or {}).get('updated_entries')) for p in pilots],out/'pilot-cells.csv')
    write_csv([dict(task=p['task'],agent=p['agent'],source_horizon=p['checkpoints'][-1]['step'],complete=p['complete'],
        allocation_cpu_seconds=p['allocation_cpu_seconds'],process_cpu_seconds_wait4=p['process_cpu_seconds_wait4'],peak_rss_kib=p['peak_rss_kib']) for p in evidence],out/'cost-reference-sources.csv')
    plan=prepare(audited,estimates,checksum(args.reference_index),checksum(args.reference_profiles))
    (out/'registered-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    if args.write_plan:
        path=HERE/'plan.json'
        if path.exists():raise ValueError('the single final plan already exists')
        path.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({k:v for k,v in estimates.items() if k!='cells'}))
