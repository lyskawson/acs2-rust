import math
from pathlib import Path
import json

from protocol import HERE,ARMS,A,A1,configurations,new_cells


PILOT_STEPS=30000
STANDARD_STEPS=2000000


def point(profile,step):
    matches=[p for p in profile['checkpoints'] if p['step']==step]
    if len(matches)!=1:raise ValueError('cost checkpoint missing or duplicated')
    return matches[0]


def late_rate(profile):
    left,right=point(profile,10000),point(profile,PILOT_STEPS)
    width=right['actual_steps']-left['actual_steps']
    if width<=0 or right['train_cpu_seconds_delta_estimate']<=0:raise ValueError('positive late-pilot training rate required')
    return right['train_cpu_seconds_delta_estimate']/width,(left['actual_steps']+right['actual_steps'])/2


def integral(rate,midpoint,exponent,start,end):
    if not 0<midpoint or not 0<=start<=end:raise ValueError('invalid projection interval')
    return rate*(end**(exponent+1)-start**(exponent+1))/((exponent+1)*midpoint**exponent)


def growth(profile,horizon):
    p30=point(profile,PILOT_STEPS);r30,mid30=late_rate(profile)
    points=profile['checkpoints'];end=points[-1]
    eval30=p30['eval_cpu_seconds_delta_estimate']
    if eval30<=0:raise ValueError('positive evaluation CPU cost required at 30k')
    observed=[p for p in points if p['step']>PILOT_STEPS]
    if profile['kind']=='archive_full':
        duration=end['actual_steps']-p30['actual_steps']
        factor=(end['train_cpu_seconds_cumulative_estimate']-p30['train_cpu_seconds_cumulative_estimate'])/(duration*r30)
        evaluations=[p['eval_cpu_seconds_delta_estimate']/eval30 for p in observed]
        return dict(training_low=factor,training_high=factor,evaluation_low=min(evaluations),evaluation_high=max(evaluations),
            source_kind=profile['kind'],source_horizon=end['step'],post_30k_points=len(observed),rate_exponent=None)
    if not observed:
        return dict(training_low=1.0,training_high=1.0,evaluation_low=1.0,evaluation_high=1.0,
            source_kind=profile['kind'],source_horizon=end['step'],post_30k_points=0,rate_exponent=None)
    previous=points[-2];width=end['actual_steps']-previous['actual_steps']
    rate=end['train_cpu_seconds_delta_estimate']/width;mid=(end['actual_steps']+previous['actual_steps'])/2
    exponent=max(0.0,math.log(rate/r30)/math.log(mid/mid30)) if rate>0 and mid>mid30 else 0.0
    remaining=max(0,horizon-end['actual_steps'])
    known=end['train_cpu_seconds_cumulative_estimate']-p30['train_cpu_seconds_cumulative_estimate']
    denominator=(horizon-p30['actual_steps'])*r30
    low=(known+rate*remaining)/denominator
    high=(known+integral(rate,mid,exponent,end['actual_steps'],max(horizon,end['actual_steps'])))/denominator
    observed_eval=[p['eval_cpu_seconds_delta_estimate']/eval30 for p in observed]
    eval_exponent=max(0.0,math.log(observed_eval[-1])/math.log(end['actual_steps']/p30['actual_steps'])) if observed_eval[-1]>0 else 0.0
    eval_high=max(1.0,*observed_eval,(horizon/p30['actual_steps'])**eval_exponent)
    return dict(training_low=min(low,high),training_high=max(low,high),evaluation_low=min(1.0,*observed_eval),evaluation_high=eval_high,
        source_kind=profile['kind'],source_horizon=end['step'],post_30k_points=len(observed),rate_exponent=exponent)


def envelope(arm,horizon,archives,longer):
    reference_arm=A if arm==A1 else arm
    references=[p for p in archives if p['agent']==reference_arm]
    long=[p for p in longer if p['agent']==arm]
    if len(references)!=120 or len(long)!=3:raise ValueError('growth envelope requires all full references and three longer cells per arm')
    bases={p['task']:p for p in references if p['seed']==42}
    if len(bases)!=6:raise ValueError('six seed42 cost baselines required')
    estimates=[]
    train_levels=[];eval_levels=[]
    for source in references:
        estimate=growth(source,horizon)
        train_level=late_rate(source)[0]/late_rate(bases[source['task']])[0]
        eval_level=point(source,PILOT_STEPS)['eval_cpu_seconds_delta_estimate']/point(bases[source['task']],PILOT_STEPS)['eval_cpu_seconds_delta_estimate']
        for key in ('training_low','training_high'):estimate[key]*=train_level
        for key in ('evaluation_low','evaluation_high'):estimate[key]*=eval_level
        estimates.append(estimate);train_levels.append(train_level);eval_levels.append(eval_level)
    estimates.extend(growth(p,horizon) for p in long)
    return dict(training_low=min(1.0,*(p['training_low'] for p in estimates)),
        training_high=max(1.0,*(p['training_high'] for p in estimates)),
        evaluation_low=min(1.0,*(p['evaluation_low'] for p in estimates)),
        evaluation_high=max(1.0,*(p['evaluation_high'] for p in estimates)),
        reference_arm=reference_arm,archived_full_runs=len(references),longer_attempts=len(long),
        archived_seed_training_level_min=min(train_levels),archived_seed_training_level_max=max(train_levels),
        archived_seed_evaluation_level_min=min(eval_levels),archived_seed_evaluation_level_max=max(eval_levels),
        longer_with_post_30k_points=sum(e['post_30k_points']>0 for e in estimates[-3:]),
        empirical_and_projected=True)


def project(profile,targets,bounds,longer):
    p30=point(profile,PILOT_STEPS);rate,_=late_rate(profile);horizon=targets[-1]
    attributed=p30['train_cpu_seconds_cumulative_estimate']+p30['eval_cpu_seconds_cumulative_estimate']
    if attributed<=0:raise ValueError('positive attributed pilot CPU required')
    scale=profile['process_cpu_seconds_wait4']/attributed
    overhead=max(0,profile['allocation_cpu_seconds']-profile['process_cpu_seconds_wait4'])
    future=horizon-p30['actual_steps']
    missing=sum(t>PILOT_STEPS for t in targets)
    train=lambda factor:p30['train_cpu_seconds_cumulative_estimate']+rate*future*factor
    evaluate=lambda factor:p30['eval_cpu_seconds_cumulative_estimate']+missing*p30['eval_cpu_seconds_delta_estimate']*factor
    low=scale*(train(bounds['training_low'])+evaluate(bounds['evaluation_low']))+overhead
    high=scale*(train(bounds['training_high'])+evaluate(bounds['evaluation_high']))+overhead*len(targets)/4
    own=[p for p in longer if (p['task'],p['agent'])==(profile['task'],profile['agent'])]
    observed_lower=max([profile['allocation_cpu_seconds']]+[p['allocation_cpu_seconds'] for p in own])
    low=max(low,observed_lower);high=max(high,low)
    return dict(low_cpu_seconds=low,high_cpu_seconds=high,late_training_cpu_seconds_per_step=rate,
        attribution_scale=scale,observed_allocation_lower_bound_seconds=observed_lower)


def forecast(pilots,archives,longer,pilot_allocation_seconds,prefix_identity,names=None):
    if not prefix_identity.get('verified') or prefix_identity.get('cells')!=18 or prefix_identity.get('checkpoint_comparisons')!=72:
        raise ValueError('all 18 fresh prefix identity checks must pass before cost-growth reuse')
    expected={(c['task'],arm) for c,arm in new_cells()}
    if len(pilots)!=138 or {(p['task'],p['agent']) for p in pilots}!=expected:
        raise ValueError('forecast requires all 138 fresh 30k cost measurements')
    if len(archives)!=600 or len(longer)!=18:raise ValueError('all full and stopped cost evidence is required')
    authenticated={(r['configuration_id'],r['agent']):r['prior_sha256'] for r in prefix_identity['checks']}
    if len(authenticated)!=18 or any(authenticated.get((p['configuration_id'],p['agent']))!=p['source_sha256'] for p in longer):
        raise ValueError('longer cost profiles do not match the authenticated prefixes')
    names=names or json.loads((HERE/'maze-names.json').read_text())
    configs={c['task']:c for c in configurations()};bounds={};cells=[]
    for profile in pilots:
        if not profile['complete'] or [p['step'] for p in profile['checkpoints']]!=[1000,3000,10000,30000]:
            raise ValueError('cost pilot missing complete checkpoints')
        arm=profile['agent'];config=configs[profile['task']]
        for budget in {STANDARD_STEPS,config['targets'][-1]}:
            key=arm,budget
            if key not in bounds:bounds[key]=envelope(arm,budget,archives,longer)
        standard=project(profile,[1000,3000,10000,30000,100000,300000,1000000,2000000],bounds[arm,STANDARD_STEPS],longer)
        registered=project(profile,config['targets'],bounds[arm,config['targets'][-1]],longer)
        cells.append(dict(task=profile['task'],maze=names[profile['task']],agent=arm,group=config['group'],
            pilot_steps=PILOT_STEPS,pilot_allocation_cpu_seconds=profile['allocation_cpu_seconds'],
            pilot_peak_rss_kib=profile['peak_rss_kib'],registered_budget_steps=config['targets'][-1],full_runs_remaining=20,
            forecast_2m_low_cpu_seconds=standard['low_cpu_seconds'],forecast_2m_high_cpu_seconds=standard['high_cpu_seconds'],
            forecast_registered_low_cpu_seconds=registered['low_cpu_seconds'],forecast_registered_high_cpu_seconds=registered['high_cpu_seconds'],
            late_training_cpu_seconds_per_step=standard['late_training_cpu_seconds_per_step'],
            observed_lower_bound_seconds=standard['observed_allocation_lower_bound_seconds'],
            proposed_time_limit_seconds=max(180,60*math.ceil(1.25*registered['high_cpu_seconds']/60)),growth=bounds[arm,STANDARD_STEPS]))
    cells.sort(key=lambda c:(-20*c['forecast_registered_high_cpu_seconds'],c['task'],c['agent']))
    low=20*sum(c['forecast_registered_low_cpu_seconds'] for c in cells)
    high=20*sum(c['forecast_registered_high_cpu_seconds'] for c in cells)
    return dict(cells=cells,new_full_runs=2760,cost_pilot_cells=138,
        pilot_all_attempts_cpu_hours=pilot_allocation_seconds/3600,
        future_full_grid_low_cpu_hours=low/3600,future_full_grid_high_cpu_hours=high/3600,
        total_with_pilot_low_cpu_hours=(pilot_allocation_seconds+low)/3600,
        total_with_pilot_high_cpu_hours=(pilot_allocation_seconds+high)/3600,
        requested_cpu_hour_limit=math.ceil((pilot_allocation_seconds+20*sum(c['proposed_time_limit_seconds'] for c in cells))/3600),
        hard_reservations_with_pilot_cpu_hours=(pilot_allocation_seconds+20*sum(c['proposed_time_limit_seconds'] for c in cells))/3600,
        memory_mib=max(2048,128*math.ceil(3*max(p['peak_rss_kib'] for p in pilots+longer)/1024/128)),
        interval_kind='Conditional scenario range, not a confidence interval or a guaranteed cost bound.',
        training_model='Measure 10k--30k training CPU/step. Full archive factors are the observed mean post-30k rate divided by that late-pilot rate. Longer stopped attempts use measured post-30k CPU plus a constant last rate (low scenario) or continued nonnegative log-rate growth (high scenario). For each full reference, normalize the observed post-30k rate to the seed42 late-pilot rate on that same archived maze, retaining joint seed-level variation and growth. Take min and max across matching-arm full and longer evidence, including a constant-rate factor of one.',
        evaluation_model='Preserve measured first-four evaluation CPU; future checkpoints use observed minimum and maximum evaluation-cost factors from full references normalized to their same-maze seed42 30k evaluation cost and from longer attempts, with nonnegative power continuation for the latter. Normalize attributed CPU to exact wait4 process CPU and add allocation overhead; upper overhead scales with checkpoint count.',
        reference_policy='120 full archive curves per arm over the original six mazes and 20 seeds, plus three longer stopped curves per arm. ACS2 r(g) uses ACS2 full-archive growth factors and its own three stopped curves, with per-goal overhead measured by its own fresh pilot.',
        censoring='A stopped attempt with no post-30k checkpoint contributes no observed post-30k growth factor. Its exact spent allocation remains an observed lower bound for that cell. Prefix equality is required before longer evidence is admitted.',
        limitations='One fresh cost seed and cross-maze growth transfer do not bound seed variance or unobserved late changes. All seeds retain per-pair evaluation cost conservatively; compact runs receive no unmeasured discount. The 25% requested-limit margin is operational, not statistical.',
        success_summary=False,budgets_changed=False)
