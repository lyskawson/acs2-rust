import json
from pathlib import Path
from protocol import HERE,ARMS,SEEDS,PER_PAIR_SEEDS,OLD_TASKS,new_cells,analysis_registration


def build():
    cells=[]
    for config,arm in new_cells():
        minutes=20 if config['task']=='mazee3_alcs' else 10 if config['task'] in ('mazee1_alcs','mazee2_alcs') else 5
        cells.append(dict(configuration_id=config['id'],task=config['task'],agent=arm,seed=42,
                          per_pair=True,limit_seconds=minutes*60))
    return dict(study='per-goal-reward-ablation',stage='authorized_cost_pilot',pilot_seed=42,pilot_runs=cells,
        pilot_run_count=138,pilot_cpu_hour_limit=15,pilot_hard_reservations_cpu_hours=sum(c['limit_seconds'] for c in cells)/3600,
        arms=list(ARMS),seeds=SEEDS,per_pair_seeds=PER_PAIR_SEEDS,
        new_runs=2760,archived_runs=600,archive_policy='Reuse ACS2 from verified 4b pair grid; E/S from original archives; E+1/S+1 from per-goal-reward. Never rerun for the full grid.',
        old_tasks=list(OLD_TASKS),analysis=analysis_registration(),
        authorization='Only seed42 cost pilot, all 138 new cells with per-pair records. All attempts, spent CPU time, reservations and new limits within 15 CPU-hours. No later seeds until explicit plan approval with a CPU-hour limit.')


if __name__=='__main__':
    (HERE/'registration.json').write_text(json.dumps(build(),indent=2)+'\n')
