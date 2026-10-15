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


def reservation_batches(jobs):
    groups = [jobs[index:index + 20] for index in range(0, len(jobs), 20)]
    if len(groups) > 1 and len(groups[-1]) < 20:
        groups[-2].extend(groups.pop())
    if any(not 20 <= len(group) <= 40 for group in groups):
        raise ValueError('registered execution needs batches of twenty to forty jobs')
    return [dict(jobs=group, limit_minutes=max(job['limit_minutes'] for job in group),
                 reservation_cpu_seconds=len(group) * max(job['limit_minutes'] for job in group) * 60)
            for group in groups]


def forecast(pilot_runs, calibration, spent_seconds):
    runs = {(run['configuration_id'], run['agent'], run['temperature']): run['cost'] for run in pilot_runs}
    calibrations = {(cell['configuration_id'], cell['agent']): cell for cell in calibration['cells']}
    expected_controls = {(config['id'], agent) for config in configurations() for agent in AGENTS[2:]}
    if set(calibrations) != expected_controls:
        raise ValueError('control calibration must cover every maze and both head counts')
    ratios = {}
    for agent in PRIORITIZED:
        measured = [(config['id'], runs[config['id'], agent, 24]['allocation_cpu_seconds'] /
                     runs[config['id'], agent, 1]['allocation_cpu_seconds'])
                    for config in configurations() if (config['id'], agent, 24) in runs]
        if len(measured) != 5:
            raise ValueError('cost pilot must retain the five measured second-temperature mazes')
        ratios[agent] = max(measured, key=lambda item: (item[1], item[0]))
    cells, jobs, limits = [], [], {}
    expected_seconds, memory = 0.0, 512
    for config in configurations():
        limits[config['id']] = {}
        reference = calibrations[config['id'], AGENTS[2]]
        extra_tests_per_classifier = reference['all_block_disagreement_transitions'] + config['cap'] * reference['additional_policy_pairs']
        for agent in AGENTS:
            temperature = 0 if agent.endswith('_uniform') else 24
            key = config['id'], agent, temperature
            missing = key not in runs
            if missing and (config['task'] != 'maze6' or agent not in PRIORITIZED):
                raise ValueError('cost pilot lacks a required measured arm')
            pilot_cost = runs[config['id'], agent, 1] if missing else runs[key]
            if (pilot_cost['body_wall_seconds'] <= 0 or pilot_cost['allocation_cpu_seconds'] <= 0
                    or pilot_cost['peak_rss_kib'] <= 0):
                raise ValueError('invalid pilot cost')
            scale = ratios[agent][1] if missing else 1.0
            basis = 'measured T=24' if temperature else 'measured T=0 scaled by deterministic classifier-perception work'
            calibration_cell = None
            if temperature == 0:
                calibration_cell = calibrations[config['id'], agent]
                old, new = calibration_cell['pilot_classifier_tests'], calibration_cell['revised_classifier_tests']
                if not 0 < new < old or old - new != calibration_cell['avoided_classifier_tests']:
                    raise ValueError('control calibration does not account exactly for the removed value queries')
                scale = new / old
            if missing:
                basis = 'Maze6 T=1 scaled by the largest measured T=24/T=1 allocation CPUTimeRAW ratio for the same K'
            extra_evaluation = (pilot_cost['pilot_population_sum_over_points'] * extra_tests_per_classifier *
                                pilot_cost['training_wall_seconds'] / pilot_cost['classifier_perception_tests']) * (ratios[agent][1] if missing else 1.0)
            body = scale * pilot_cost['body_wall_seconds'] + extra_evaluation
            overhead = max(0, pilot_cost['allocation_cpu_seconds'] - pilot_cost['process_cpu_seconds'])
            cost = body + overhead
            pilot_limit = max(2, math.ceil((1.5 * pilot_cost['body_wall_seconds'] + 30) / 60))
            limit = max(2, math.ceil((1.5 * body + 30) / 60), pilot_limit if temperature == 0 else 0)
            limits[config['id']][agent] = limit
            expected_seconds += 20 * cost
            jobs.extend(dict(configuration_id=config['id'], agent=agent, seed=seed, limit_minutes=limit)
                        for seed in range(42, 62))
            memory = max(memory, math.ceil(3 * pilot_cost['peak_rss_kib'] / 1024 / 128) * 128)
            cells.append(dict(configuration_id=config['id'], task=config['task'], agent=agent, temperature=temperature,
                basis=basis, pilot_body_wall_seconds=pilot_cost['body_wall_seconds'], pilot_allocation_cpu_seconds=pilot_cost['allocation_cpu_seconds'],
                scale=scale, ratio_source_configuration=ratios[agent][0] if missing else None,
                additional_evaluation_seconds=extra_evaluation, expected_body_seconds=body,
                expected_allocation_seconds=cost, pilot_derived_limit_minutes=pilot_limit, limit_minutes=limit,
                calibration=calibration_cell))
    batches = reservation_batches(jobs)
    hard = sum(batch['reservation_cpu_seconds'] for batch in batches)
    return dict(grid_expected_cpu_hours=expected_seconds / 3600,
                doubled_forecast_with_pilots_cpu_hours=(spent_seconds + 2 * expected_seconds) / 3600,
                hard_bound_with_pilots_cpu_hours=(spent_seconds + hard) / 3600,
                grid_reservation_cpu_seconds=hard, pilot_allocation_cpu_seconds=spent_seconds,
                limits_minutes=limits, memory_mib=memory, grid_runs=len(jobs), grid_batches=batches,
                cells=cells, maze6_ratio_metric='allocation CPUTimeRAW T=24 divided by T=1; maximum over the five measured mazes separately for K=2 and K=4',
                control_scale='classifier-perception tests at the same registered seed-42 checkpoint; no timing ratio',
                added_evaluation_rule='Additional training/test policies charged at every pair times cap, plus disagreement queries for all three blocks, times the pilot training wall seconds per classifier-perception test. Sum the pilot head populations at every registered point; scale the missing Maze6 T=24 estimate by its same-K cost ratio.',
                limitations='One seed, short control calibration and a missing Maze6 T=24 pilot are forecasts, not guaranteed costs; actual limits and accounting govern admission.')


def draft(pilot_runs, rows, calibration, spent_seconds, measurement):
    evidence = candidates_from_priorities(rows)
    prediction = forecast(pilot_runs, calibration, spent_seconds)
    configs = configurations()
    for config in configs:
        config['limits_minutes'] = prediction['limits_minutes'][config['id']]
        config['agent_settings'] = {agent: dict(temperature=24) for agent in PRIORITIZED}
        config['evaluation_blocks'] = ['training', 'validation', 'test']
        config['vcp_diagnostics'] = 'evaluation_disagreement'
    her = json.loads((ROOT / 'experiments/her-variants/plan.json').read_text())
    return dict(stage='draft_pending_hindsight_confirmation_and_cpu_approval', experiment='value-consistency',
        measurement_commit=measurement['measurement_commit'], binary_sha256=measurement['binary_sha256'],
        preset=measurement['preset'], pair_protocol_sha256=PAIR_SHA256, seeds=list(range(42, 62)), agents=list(AGENTS), batch_size=20,
        configurations=configs, hindsight=dict(copy_update='Full', admissibility='EveryTransition',
            strategy='future', relabeled_proportion=2 / 3, relabeled_proportion_exact='2/3',
            confirmation_required_before_registration=True,
            changed_choice='If confirmed settings differ from every approved phase-5 arm, add its one-head reference on seeds 42--61 at all registered points and revise the forecast before approval.'),
        temperature=dict(prioritized=24, controls=0, fixed_before_results=True, tuning=False,
            rationale='Across the representable integer range 1..24, the priority-statistics rule cannot reach mean-acceptance targets 3/4, 1/2 or 1/4: the pooled pilot curves remain at least 0.90 on every maze and K. T=1, the predecessor setting, accepted 99.69--99.97% of proposals and is practically the T=0 control already in the grid. T=24 is the strongest prioritization available in the registered range; tuning between them would mostly repeat the registered VCP-minus-control comparison.',
            pilot_priority_evidence=evidence),
        final_grid=dict(agents=list(AGENTS), seeds=list(range(42, 62)), evaluation_blocks=['training', 'validation', 'test'],
            runs=480, temperature='T=24 fixed for both prioritized arms; T=0 for both controls; no tuning stage'),
        references=dict(her_plan_sha256=hashlib.sha256((ROOT / 'experiments/her-variants/plan.json').read_bytes()).hexdigest(),
            baseline_plan_sha256=hashlib.sha256((ROOT / 'experiments/maze-baselines/grid-plan.json').read_bytes()).hexdigest(),
            agents=['acs2her', 'acs2er_episodic', 'acs2', 'acs2er'],
            archived_names='experiments/grid/agent_names.json', rule='reuse verified approved archives; never rerun'),
        identity_checks=dict(seed=42, source_pilot_commit='79648424076144c77a3a87acf87418bb61aa7c42',
            rule='Before accepting grid results, all seed-42 pilot-overlapping nominal points reproduce every learning, evaluation and knowledge field of each matching arm and temperature. Compare the validation block, since the pilot evaluated validation only. Prioritized arms retain every existing portable counter and priority field; controls remove exactly the independently counted value queries and classifier tests, satisfy formations=steps+3*full+2*values_only with zero priority queries, and report no training variance. Additive diagnostics and target-native inline headers are checked separately.',
            missing_scope='Maze6 T=24 has no pilot identity oracle; the unchanged core/HER pins and joint-volume identities still apply. T=1 pilot rows are not a T=24 oracle.'),
        analysis=dict(views=['checkpoint_mean_within_seed', 'final_point'], block='test', seed_unit='seed, averaging heads inside the seed',
            summaries='mean and SE over twenty seeds', paired_interval='marginal 95% Student t(19), critical value 2.093024054408263, without multiplicity correction',
            comparisons=[['acs2vcp_k2', 'acs2vcp_k2_uniform'], ['acs2vcp_k4', 'acs2vcp_k4_uniform'],
                ['acs2vcp_k2', 'acs2her'], ['acs2vcp_k4', 'acs2her'], ['acs2vcp_k2_uniform', 'acs2her'],
                ['acs2vcp_k4_uniform', 'acs2her'], ['acs2vcp_k2', 'acs2er_episodic'],
                ['acs2vcp_k4', 'acs2er_episodic'], ['acs2vcp_k4', 'acs2vcp_k2']],
            thresholds=her['analysis']['thresholds'], baseline_threshold_crossing_checks=her['analysis']['baseline_threshold_crossing_checks'],
            threshold_rule='First registered nominal point reaching each exact archived final-mean test threshold. Recover integer counts per head, divide by heads*test pairs, and use Fraction for seed means and comparisons; no interpolation.',
            secondary=['knowledge', 'head_spread', 'evaluation_head_disagreement_on_fixed_block_transitions',
                'proposal_and_accepted_variance_with_logarithmic_bins_and_exact_zeros', 'variance_maximum', 'acceptance',
                'proposals_per_draw', 'route_composition', 'objective_cost_by_purpose', 'priority_value_queries',
                'population_per_head_and_total', 'memory_per_head_and_total_including_measurement_diagnostics',
                'learning_updates', 'process_and_wall_time', 'allocation_cpu_hours'], other_analyses='post hoc and explicitly labelled'),
        execution=dict(partition='lem-cpu-normal', concurrency=20, batch_size=20,
            cpu_model='AMD EPYC 9554 64-Core Processor', tooling='experiments/grid/',
            guards='clean frozen musl measurement, mechanical manifests before sbatch, grant and test-only before each submission, node guards and STOP; no lease, host pinning or laptop automation',
            admission='pilots + all attempted allocation CPUTimeRAW + outstanding hard reservations + new limits <= the approved limit',
            copies='two independently checksum-verified raw copies',
            scope='All four arms, six mazes, twenty seeds and nine comparisons are fixed. The user sets the CPU-hour limit at approval; no automatic sixty-hour trigger or arm reduction.'),
        forecast=prediction, requested_cpu_hour_limit=5 * math.ceil(max(prediction['doubled_forecast_with_pilots_cpu_hours'], prediction['hard_bound_with_pilots_cpu_hours']) / 5),
        stage2_authorized=False)
