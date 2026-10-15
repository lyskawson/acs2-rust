import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAIR_SHA256 = 'eb1d2c01669b4471f68cbc1f671b81fea77e71cfc8e72b75e4f17cd5dcf6b189'
AGENTS = ['acs2vcp_k2', 'acs2vcp_k4', 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform']
PRIORITIZED = AGENTS[:2]
TEMPERATURE_RANGE = list(range(1, 25))
PILOT_TEMPERATURES = [1, 24]
PILOT_PARTS = ['temperature1-and-controls', 'temperature24', 'temperature24-reduced']
REDUCED_MAZES = ['maze4', 'maze7', 'woods14']
RECOVERY_LIMITS = dict(maze4=(10, 14), maze5=(16, 24), maze6=(18, 28),
                      maze7=(20, 32), woods14=(8, 10), mazema=(8, 10))
SECOND_TEMPERATURE_ORDER = ['woods14', 'maze4', 'maze7', 'mazema', 'maze5', 'maze6']
COST_REDUCTION_THRESHOLD_HOURS = 60
RECOVERY_PARTS = [f'recovery-t{temperature}-{maze}-k{heads}'
                  for temperature in PILOT_TEMPERATURES for maze in RECOVERY_LIMITS for heads in [2, 4]
                  if (temperature, maze, heads) != (1, 'maze4', 2)]


def configurations():
    registration = ROOT / 'experiments/maze-baselines/pair-protocol.json'
    if hashlib.sha256(registration.read_bytes()).hexdigest() != PAIR_SHA256:
        raise ValueError('registered pair protocol changed')
    result = json.loads((ROOT / 'experiments/maze-baselines/pair-pilot.json').read_text())['configurations']
    for config in result:
        config.pop('limits_minutes')
    return result


def pilot(part):
    if part not in PILOT_PARTS:
        raise ValueError('unregistered VCP pilot part')
    agents = AGENTS if part == PILOT_PARTS[0] else PRIORITIZED
    temperature = 1 if part == PILOT_PARTS[0] else 24
    configs = configurations()
    if part == PILOT_PARTS[2]:
        configs = [config for config in configs if config['task'] in REDUCED_MAZES]
    limits = dict(maze4=8, maze5=12, maze6=14, maze7=16, woods14=6, mazema=4)
    for config in configs:
        config['limits_minutes'] = {agent: limits[config['task']] for agent in agents}
        config['agent_settings'] = {agent: dict(temperature=temperature) for agent in PRIORITIZED}
        config['evaluation_blocks'] = ['validation']
    return dict(stage='authorized_cost_pilot', experiment='value-consistency', pilot_part=part,
                seeds=[42], agents=agents, configurations=configs, partition='lem-cpu-normal',
                cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=512, concurrency=24,
                phase_cpu_hour_limit=5, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-15')


def recovery(part):
    if part not in RECOVERY_PARTS:
        raise ValueError('unregistered VCP recovery part')
    _, temperature, maze, heads = part.split('-')
    temperature, heads = int(temperature[1:]), int(heads[1:])
    agents = [f'acs2vcp_k{heads}']
    if temperature == 1:
        agents.append(f'acs2vcp_k{heads}_uniform')
    config = next(item for item in configurations() if item['task'] == maze)
    config['limits_minutes'] = {agent: RECOVERY_LIMITS[maze][heads // 2 - 1] for agent in agents}
    config['agent_settings'] = {agent: dict(temperature=temperature) for agent in agents if agent in PRIORITIZED}
    config['evaluation_blocks'] = ['validation']
    return dict(stage='authorized_cost_pilot', experiment='value-consistency', pilot_part=part,
                seeds=[42], agents=agents, configurations=[config], partition='lem-cpu-normal',
                cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=512, concurrency=20,
                phase_cpu_hour_limit=5, pilot_allocation_cpu_seconds=0, accounting_since='2026-10-15')


def candidates_from_priorities(rows):
    cells = {}
    for row in rows:
        agent = row['agent']
        if agent not in PRIORITIZED:
            continue
        key = (row['task'], agent, row['agent_parameters']['temperature'])
        if key not in cells or row['nominal_step'] > cells[key]['nominal_step']:
            cells[key] = row
    result = {}
    for config in configurations():
        result[config['id']] = {}
        for agent in PRIORITIZED:
            observations = [cells[key] for key in cells if key[:2] == (config['task'], agent)]
            if not observations or not any(row['agent_parameters']['temperature'] == 1 for row in observations):
                raise ValueError('cost pilot lacks the first temperature on a registered maze/head count')
            curves = []
            for row in observations:
                priority = row['replay_diagnostics']['priority']
                if priority['candidate_temperatures'] != TEMPERATURE_RANGE or priority['proposals'] <= 0:
                    raise ValueError('invalid pilot priority statistics')
                curves.append([value / priority['proposals'] for value in priority['candidate_acceptance_sums']])
            means = [sum(curve[index] for curve in curves) / len(curves) for index in range(24)]
            selected = sorted({min(TEMPERATURE_RANGE, key=lambda value: (abs(means[value - 1] - target), value))
                               for target in [0.75, 0.5, 0.25]})
            result[config['id']][agent] = dict(temperatures=selected, acceptance_curve=means,
                observed_pilot_temperatures=sorted(row['agent_parameters']['temperature'] for row in observations))
    return result


def select_temperature(scores, candidates):
    if set(scores) != set(candidates) or not scores or any(not math.isfinite(value) for value in scores.values()):
        raise ValueError('incomplete or nonfinite validation temperature scores')
    best = max(scores.values())
    return min(temperature for temperature, score in scores.items() if best - score <= 1e-12)


def reservation_batches(jobs):
    groups = [jobs[index:index + 20] for index in range(0, len(jobs), 20)]
    if len(groups) > 1 and len(groups[-1]) < 20:
        groups[-2].extend(groups.pop())
    if any(not 20 <= len(group) <= 40 for group in groups):
        raise ValueError('registered execution needs batches of twenty to forty jobs')
    return [dict(jobs=group, limit_minutes=max(job['limit_minutes'] for job in group),
                 reservation_cpu_seconds=len(group) * max(job['limit_minutes'] for job in group) * 60)
            for group in groups]


def forecast(cells, candidates, spent_seconds, agents=None):
    agents = AGENTS if agents is None else agents
    expected_keys = {(config['id'], agent) for config in configurations() for agent in AGENTS}
    if set(cells) != expected_keys:
        raise ValueError('cost pilot must cover every arm and maze')
    grid_expected = tuning_expected = 0.0
    grid_jobs, tuning_jobs = [], []
    limits = {}
    memory = 512
    for config in configurations():
        limits[config['id']] = {}
        for agent in agents:
            cell = cells[config['id'], agent]
            body = cell['body_wall_seconds']
            if body <= 0 or cell['allocation_cpu_seconds'] < 0 or cell['peak_rss_kib'] <= 0:
                raise ValueError('invalid pilot cost')
            limit = max(2, math.ceil((1.5 * body + 30) / 60))
            limits[config['id']][agent] = limit
            cost = body + max(0, cell['allocation_cpu_seconds'] - cell['process_cpu_seconds'])
            grid_expected += 20 * cost
            grid_jobs.extend(dict(configuration_id=config['id'], agent=agent, seed=seed, limit_minutes=limit)
                             for seed in range(42, 62))
            if agent in PRIORITIZED:
                count = len(candidates[config['id']][agent]['temperatures'])
                tuning_expected += 5 * count * cost
                tuning_jobs.extend(dict(configuration_id=config['id'], agent=agent, temperature=temperature,
                                        seed=seed, limit_minutes=limit)
                                   for temperature in candidates[config['id']][agent]['temperatures'] for seed in range(42, 47))
            memory = max(memory, math.ceil(3 * cell['peak_rss_kib'] / 1024 / 128) * 128)
    grid_batches, tuning_batches = reservation_batches(grid_jobs), reservation_batches(tuning_jobs)
    grid_hard = sum(batch['reservation_cpu_seconds'] for batch in grid_batches)
    tuning_hard = sum(batch['reservation_cpu_seconds'] for batch in tuning_batches)
    return dict(grid_expected_cpu_hours=grid_expected / 3600,
                tuning_expected_cpu_hours=tuning_expected / 3600,
                doubled_forecast_with_pilots_cpu_hours=(spent_seconds + 2 * (grid_expected + tuning_expected)) / 3600,
                hard_bound_with_pilots_cpu_hours=(spent_seconds + grid_hard + tuning_hard) / 3600,
                grid_reservation_cpu_seconds=grid_hard, tuning_reservation_cpu_seconds=tuning_hard,
                pilot_allocation_cpu_seconds=spent_seconds, limits_minutes=limits, memory_mib=memory,
                grid_runs=len(grid_jobs), tuning_runs=len(tuning_jobs),
                grid_batches=grid_batches, tuning_batches=tuning_batches)


def cost_reduction(cells, candidates, spent_seconds):
    candidates = copy.deepcopy(candidates)
    initial = forecast(cells, candidates, spent_seconds)
    prediction = initial
    agents = list(AGENTS)
    reductions = []
    if max(prediction['doubled_forecast_with_pilots_cpu_hours'], prediction['hard_bound_with_pilots_cpu_hours']) > COST_REDUCTION_THRESHOLD_HOURS:
        for choices in candidates.values():
            for value in choices.values():
                retained = min(value['temperatures'], key=lambda t: (abs(value['acceptance_curve'][t - 1] - 0.5), t))
                value['before_cost_reduction'] = value['temperatures']
                value['temperatures'] = [retained]
        reductions.append('Retain only the priority-derived candidate nearest mean acceptance 1/2 per maze and K.')
        prediction = forecast(cells, candidates, spent_seconds)
        if max(prediction['doubled_forecast_with_pilots_cpu_hours'], prediction['hard_bound_with_pilots_cpu_hours']) > COST_REDUCTION_THRESHOLD_HOURS:
            agents.remove('acs2vcp_k2')
            reductions.append('Remove prioritized K=2 from tuning and the final grid; retain both unprioritized controls, every maze and all twenty seeds.')
            prediction = forecast(cells, candidates, spent_seconds, agents)
    return agents, candidates, prediction, dict(trigger_cpu_hours=COST_REDUCTION_THRESHOLD_HOURS,
        rationale='Use the previous HER study approval of sixty CPU-hours as the pre-results definition of a large forecast, not as authorization for this study.',
        initial_doubled_cpu_hours=initial['doubled_forecast_with_pilots_cpu_hours'],
        initial_hard_bound_cpu_hours=initial['hard_bound_with_pilots_cpu_hours'], applied=reductions,
        remaining_rule='If the retained controls and K=4 still exceed sixty hours, request the explicit resulting limit; do not cut mazes, seeds or controls.')


def draft(cells, rows, spent_seconds, measurement):
    candidates = candidates_from_priorities(rows)
    agents, candidates, prediction, reductions = cost_reduction(cells, candidates, spent_seconds)
    configs = configurations()
    for config in configs:
        config['limits_minutes'] = prediction['limits_minutes'][config['id']]
    her = json.loads((ROOT / 'experiments/her-variants/plan.json').read_text())
    return dict(stage='draft_pending_hindsight_confirmation_and_cpu_approval', experiment='value-consistency',
        measurement_commit=measurement['measurement_commit'], binary_sha256=measurement['binary_sha256'],
        preset=measurement['preset'], pair_protocol_sha256=PAIR_SHA256, seeds=list(range(42, 62)), agents=agents, batch_size=20,
        configurations=configs, hindsight=dict(copy_update='Full', admissibility='EveryTransition',
            strategy='future', relabeled_proportion=2 / 3, relabeled_proportion_exact='2/3',
            confirmation_required_before_registration=True,
            changed_choice='If confirmed settings differ from every approved phase-5 arm, add its one-head reference on seeds 42--61 at all registered points.'),
        tuning=dict(seeds=list(range(42, 47)), evaluation_blocks=['validation'],
            candidates={maze: {agent: choice for agent, choice in choices.items() if agent in agents} for maze, choices in candidates.items()},
            candidate_rule='For each maze and K, pool each measured pilot temperature equally using its final cumulative proposal-variance acceptance curve. For target mean acceptances 3/4, 1/2 and 1/4 choose the nearest integer T in 1..24, ties to smaller T; deduplicate. No success field is read.',
            selection='Highest equally weighted checkpoint-mean validation success within each seed, averaged over five seeds; scores within 1e-12 tie to smaller T.',
            controls='T=0 fixed; not tuned', budgets='the registered final budgets and nominal points', test_access='none'),
        final_grid=dict(agents=agents, seeds=list(range(42, 62)), evaluation_blocks=['training', 'validation', 'test'],
            runs=120 * len(agents), temperature='the selected T separately for every retained maze and K; controls zero'),
        references=dict(her_plan_sha256=hashlib.sha256((ROOT / 'experiments/her-variants/plan.json').read_bytes()).hexdigest(),
            baseline_plan_sha256=hashlib.sha256((ROOT / 'experiments/maze-baselines/grid-plan.json').read_bytes()).hexdigest(),
            agents=['acs2her', 'acs2er_episodic', 'acs2', 'acs2er'],
            archived_names='experiments/grid/agent_names.json', rule='reuse verified approved archives; never rerun'),
        analysis=dict(views=['checkpoint_mean_within_seed', 'final_point'], block='test', seed_unit='seed, averaging heads inside the seed',
            summaries='mean and SE over twenty seeds', paired_interval='marginal 95% Student t(19), critical value 2.093024054408263, without multiplicity correction',
            comparisons=[['acs2vcp_k2', 'acs2vcp_k2_uniform'], ['acs2vcp_k4', 'acs2vcp_k4_uniform'],
                ['acs2vcp_k2', 'acs2her'], ['acs2vcp_k4', 'acs2her'], ['acs2vcp_k2_uniform', 'acs2her'],
                ['acs2vcp_k4_uniform', 'acs2her'], ['acs2vcp_k2', 'acs2er_episodic'],
                ['acs2vcp_k4', 'acs2er_episodic'], ['acs2vcp_k4', 'acs2vcp_k2']]
                if 'acs2vcp_k2' in agents else [['acs2vcp_k4', 'acs2vcp_k4_uniform'], ['acs2vcp_k4', 'acs2her'],
                    ['acs2vcp_k2_uniform', 'acs2her'], ['acs2vcp_k4_uniform', 'acs2her'], ['acs2vcp_k4', 'acs2er_episodic']],
            omitted_for_registered_cost_reduction=[] if 'acs2vcp_k2' in agents else
                [['acs2vcp_k2', 'acs2vcp_k2_uniform'], ['acs2vcp_k2', 'acs2her'],
                 ['acs2vcp_k2', 'acs2er_episodic'], ['acs2vcp_k4', 'acs2vcp_k2']],
            thresholds=her['analysis']['thresholds'], baseline_threshold_crossing_checks=her['analysis']['baseline_threshold_crossing_checks'],
            threshold_rule='First registered nominal point reaching each exact archived final-mean test threshold. Recover integer counts per head, divide by heads*test pairs, and use Fraction for seed means and comparisons; no interpolation.',
            secondary=['knowledge', 'head_spread', 'proposal_and_accepted_variance', 'variance_maximum', 'acceptance',
                'proposals_per_draw', 'route_composition', 'objective_cost_by_purpose', 'priority_value_queries',
                'population_per_head_and_total', 'memory_per_head_and_total', 'learning_updates', 'process_and_wall_time', 'allocation_cpu_hours'],
            other_analyses='post hoc and explicitly labelled'),
        execution=dict(partition='lem-cpu-normal', concurrency=20, batch_size=20,
            cpu_model='AMD EPYC 9554 64-Core Processor', tooling='experiments/grid/',
            guards='clean frozen musl measurement, mechanical manifests before sbatch, grant and test-only before each submission, node guards and STOP; no lease, host pinning or laptop automation',
            admission='pilots + all attempted allocation CPUTimeRAW + outstanding hard reservations + new limits <= the approved limit',
            copies='two independently checksum-verified raw copies',
            cost_reduction='Apply the registered sixty-hour forecast trigger before registration. Any later change to candidates or arms requires a revised registration and approval before tuning; keep all six mazes, twenty seeds and both T=0 controls.'),
        cost_reduction=reductions, forecast=prediction, requested_cpu_hour_limit=5 * math.ceil(max(prediction['doubled_forecast_with_pilots_cpu_hours'], prediction['hard_bound_with_pilots_cpu_hours']) / 5),
        stage2_authorized=False)
