import json
from protocol import HERE,new_cells

TARGETS = [1000,3000,10000,30000]
PRIOR_CPU_SECONDS = 9810
MEASUREMENT_COMMIT = '41b025a0733a23d8a3d438a18cbf34b4526c9af2'
BINARY_SHA256 = '630baba8e6244e4ce38ec3cf31e59a36cdf054f67fbe488ea55a8c27da178e3d'


def configuration(config):
    import copy
    from protocol import recorded
    value=recorded(copy.deepcopy(config),42)
    if value['targets'][:4]!=TARGETS:
        raise ValueError('30k pilot must be an exact registered checkpoint prefix')
    value['targets']=TARGETS
    return value


def build():
    runs=[dict(configuration_id=c['id'],task=c['task'],agent=a,seed=42,
               limit_seconds=600 if c['task']=='mazee3_alcs' else 300,
               prior_prefix_check=c['task'] in ('mazee1_alcs','mazee2_alcs','mazee3_alcs')) for c,a in new_cells()]
    return dict(study='per-goal-reward-ablation',stage='authorized_30000_cost_pilot',
        targets=TARGETS,pilot_runs=runs,pilot_run_count=138,phase_cpu_hour_limit=15,
        prior_cpu_seconds=PRIOR_CPU_SECONDS,prior_attempts=18,prior_arrays=[6296853,6296855],
        maximum_new_reservations_cpu_seconds=sum(r['limit_seconds'] for r in runs),
        maximum_combined_cpu_hours=(PRIOR_CPU_SECONDS+sum(r['limit_seconds'] for r in runs))/3600,
        measurement_commit=MEASUREMENT_COMMIT,binary_sha256=BINARY_SHA256,
        authorization='User approval on 2026-11-10: first 30k steps in all 138 cells with registered per-pair records. The same 15 CPU-hour limit includes 9810 CPU-s already spent. Do not repeat the 18 first-wave attempts at full budget. Reuse their prefixes only after equality with fresh 30k runs. Report cost only, preserve full registered protocols and budgets, stop with plan and cost-range forecast.')


if __name__=='__main__':
    (HERE/'cost-pilot.json').write_text(json.dumps(build(),indent=2)+'\n')
