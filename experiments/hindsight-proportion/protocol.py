import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import load_plan

PAIR_SHA256 = 'eb1d2c01669b4471f68cbc1f671b81fea77e71cfc8e72b75e4f17cd5dcf6b189'
HER_SHA256 = 'e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444'
ADMISSIBILITY_SHA256 = 'f7dcf93ce08fa6eee9093736c5210e3c532cecfee5541e5f5475258aff91d2da'
BASELINE_SHA256 = '913ce814484b742265ba88db09d72d4feaf8f981b200811454f9fce11b6b5a78'
SHORTENED = 'acs2her_shortened'
HER = 'acs2her'
CANDIDATES = [1 / 3, 1 / 2, 2 / 3, 4 / 5]
BLOCKS = ['training', 'validation', 'test']
PILOT_SETTINGS = [(SHORTENED, 1 / 3, BLOCKS), (SHORTENED, 1 / 2, BLOCKS),
                  (SHORTENED, 4 / 5, BLOCKS), (HER, 1 / 2, BLOCKS),
                  (SHORTENED, 2 / 3, ['validation'])]


def valid_proportion(value):
    if type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('proportion must be a finite binary64 value in [0, 1]')
    return value


def candidates(values):
    values = [valid_proportion(value) for value in values]
    if not values or len(set(values)) != len(values) or 2 / 3 not in values:
        raise ValueError('distinct candidates including the measured 2/3 reference are required')
    return sorted(values)


def label(agent, proportion):
    if agent not in (SHORTENED, HER):
        raise ValueError('this study parameterizes only full HER and shortened HER')
    value = valid_proportion(proportion)
    fraction = Fraction(value).limit_denominator(10 ** 9)
    if float(fraction) != value:
        fraction = Fraction(value)
    return f'{agent}_p{fraction.numerator}of{fraction.denominator}'


def configurations():
    registration = load_plan(ROOT / 'experiments/maze-baselines/pair-protocol.json', PAIR_SHA256)
    source = json.loads((ROOT / 'experiments/maze-baselines/pair-pilot.json').read_text())
    configs = copy.deepcopy(source['configurations'])
    for config in configs:
        config.pop('limits_minutes')
        if config['registration'] not in registration['configurations']:
            raise ValueError('pair configuration changed')
    return configs


def setting(config, agent, proportion, blocks, limit=10):
    arm = label(agent, proportion)
    return dict(copy.deepcopy(config), id=config['id'] + '_' + arm,
                registered_id=config['id'], row_arm=arm, evaluation_blocks=list(blocks),
                agent_settings={agent: dict(relabeled_proportion=proportion)},
                limits_minutes={agent: limit})


def pilot_roots():
    return ['pilot-' + label(agent, proportion) for agent, proportion, _ in PILOT_SETTINGS]


def scope_roots():
    return pilot_roots() + ['tuning', 'control'] + ['test-' + config['id'] for config in configurations()]


def pilot(index):
    if type(index) is not int or not 0 <= index < len(PILOT_SETTINGS):
        raise ValueError('unregistered cost-pilot arm')
    agent, proportion, blocks = PILOT_SETTINGS[index]
    return dict(stage='authorized_cost_pilot', experiment='hindsight-proportion', pilot_index=index,
                seeds=[42], agents=[agent], configurations=[setting(config, agent, proportion, blocks) for config in configurations()],
                partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor',
                memory_mib=2048, concurrency=20, phase_cpu_hour_limit=5,
                pilot_allocation_cpu_seconds=0, accounting_since='2026-10-28',
                registration_sha256=PAIR_SHA256)


def tuning_groups(items):
    groups = [items[index:index + 4] for index in range(0, len(items), 4)]
    if len(groups) > 1 and len(groups[-1]) < 4:
        groups[-2].extend(groups.pop())
    if any(not 4 <= len(group) <= 8 for group in groups):
        raise ValueError('tuning must partition into groups of 20--40 jobs')
    return groups


def forecast(cells, pilot_seconds, proportions=CANDIDATES):
    proportions = candidates(proportions)
    if not 0 < pilot_seconds <= 5 * 3600:
        raise ValueError('cost pilot must fit five allocated CPU-hours')
    estimates = []
    memory = 512
    for config in configurations():
        costs = {}
        for agent, proportion, blocks in PILOT_SETTINGS:
            key = config['id'], label(agent, proportion)
            if key not in cells:
                raise ValueError('missing pilot cost cell')
            cost = cells[key]
            if any(not math.isfinite(cost[field]) or cost[field] <= 0 for field in
                   ('body_wall_seconds', 'process_cpu_seconds', 'allocation_cpu_seconds', 'peak_rss_kib')):
                raise ValueError('unmeasured pilot cost')
            costs[agent, proportion, tuple(blocks)] = cost
            memory = max(memory, math.ceil(3 * cost['peak_rss_kib'] / 1024 / 128) * 128)
        for purpose, agent, points, seeds in [('tuning', SHORTENED, proportions, 5),
                                              ('test', SHORTENED, [p for p in proportions if p != 2 / 3], 20),
                                              ('control', HER, [0.5], 20)]:
            purpose_cells = []
            for p in points:
                mode = tuple(['validation'] if purpose == 'tuning' else BLOCKS)
                observed = costs.get((agent, p, mode))
                if observed is None:
                    observed = costs.get((agent, p, tuple(BLOCKS)))
                if observed is None:
                    observed = max((cost for (name, _, _), cost in costs.items() if name == agent),
                                   key=lambda item: item['allocation_cpu_seconds'])
                limit = max(3, math.ceil((2 * observed['body_wall_seconds'] + 60) / 60))
                purpose_cells.append(dict(purpose=purpose, configuration_id=config['id'], agent=agent,
                    proportion=p, runs=seeds, expected_cpu_seconds=seeds * observed['allocation_cpu_seconds'],
                    limit_minutes=limit, reservation_cpu_seconds=seeds * limit * 60))
            if purpose == 'test' and purpose_cells:
                chosen = max(purpose_cells, key=lambda cell: cell['expected_cpu_seconds'])
                reserve = max(cell['reservation_cpu_seconds'] for cell in purpose_cells)
                estimates.append(dict(chosen, reservation_cpu_seconds=reserve, limit_minutes=reserve // (20 * 60),
                    rule='Worst selected non-reference candidate; only one candidate runs per maze.'))
            else:
                estimates.extend(purpose_cells)
    arrays = []
    for group in tuning_groups([cell for cell in estimates if cell['purpose'] == 'tuning']):
        limit = max(cell['limit_minutes'] for cell in group)
        group_id = group[0]['configuration_id'] + '_' + label(SHORTENED, group[0]['proportion'])
        for cell in group:
            cell.update(limit_minutes=limit, reservation_cpu_seconds=cell['runs'] * limit * 60, batch_group=group_id)
        arrays.append(dict(purpose='tuning', batch_group=group_id, runs=sum(cell['runs'] for cell in group),
                           limit_minutes=limit, reservation_cpu_seconds=sum(cell['reservation_cpu_seconds'] for cell in group)))
    for cell in estimates:
        if cell['purpose'] != 'tuning':
            arrays.append({key: cell[key] for key in ('purpose', 'configuration_id', 'runs', 'limit_minutes', 'reservation_cpu_seconds')})
    expected = sum(cell['expected_cpu_seconds'] for cell in estimates)
    reserved = sum(cell['reservation_cpu_seconds'] for cell in estimates)
    doubled, hard = pilot_seconds + 2 * expected, pilot_seconds + reserved
    return dict(candidates=proportions, cells=estimates, pilot_allocation_cpu_seconds=pilot_seconds,
        expected_total_cpu_hours=(pilot_seconds + expected) / 3600,
        doubled_forecast_cpu_hours=doubled / 3600, hard_reservations_cpu_hours=hard / 3600,
        requested_cpu_hour_limit=5 * math.ceil(max(doubled, hard) / 3600 / 5),
        array_reservations=arrays, memory_mib=memory,
        reservation_rule='Actual tuning groups of 20--40 jobs use their own maximum limit; control and conditional test use twenty-seed per-maze arrays. Test reserves the worst candidate per maze.',
        estimate_rule='Allocation CPUTimeRAW at seed 42; full-block cost conservatively proxies validation-only cost when unmeasured. Unpiloted candidates use the most expensive pilot setting of that agent on that maze.',
        limit_rule='max(3, ceil((2 * body wall seconds + 60) / 60)) minutes; shared node watchdog keeps 30 seconds.',
        limitation='One-seed forecast, no guarantee about new seeds. Every attempt and outstanding hard limit remains charged.')


def study(cells, pilot_seconds, identity, evidence, proportions=CANDIDATES):
    proportions = candidates(proportions)
    her = load_plan(ROOT / 'experiments/her-variants/plan.json', HER_SHA256)
    forecasts = dict(with_four_fifths=forecast(cells, pilot_seconds, proportions),
        without_four_fifths=forecast(cells, pilot_seconds, [p for p in proportions if p != 0.8]))
    return dict(stage='grid_pending_approval', experiment='hindsight-proportion', **identity,
        candidates=proportions, configurations=configurations(), pair_protocol_sha256=PAIR_SHA256,
        tuning=dict(agent=SHORTENED, proportions=proportions, seeds=list(range(42, 47)), evaluation_blocks=['validation'],
            runs=6 * len(proportions) * 5, selection='Per maze: mean over registered points within seed, then mean over seeds 42--46; maximize validation success, within 1e-12 of maximum choose smaller p.'),
        test=dict(agent=SHORTENED, seeds=list(range(42, 62)), evaluation_blocks=BLOCKS, maximum_runs=120,
            derivation='Recompute saved selection from complete verified tuning rows at derivation. Run twenty seeds only where p* differs from 2/3; otherwise reuse admissibility rows.'),
        control=dict(agent=HER, proportion=0.5, seeds=list(range(42, 62)), evaluation_blocks=BLOCKS, runs=120,
            independence='May run beside tuning; no selection dependency.'),
        references=dict(shortened_plan_sha256=ADMISSIBILITY_SHA256, her_plan_sha256=HER_SHA256,
            baseline_plan_sha256=BASELINE_SHA256, names='experiments/grid/agent_names.json',
            arms=[label(SHORTENED, 2 / 3), label(HER, 2 / 3), 'acs2er_episodic', 'acs2', 'acs2er'],
            policy='Reuse verified archived rows and registrations; never rerun a reference arm.'),
        cost_pilot=evidence, forecasts=forecasts,
        analysis=dict(block='test', views=['checkpoint_mean', 'final_point'], seeds=list(range(42, 62)),
            aliases=dict(S=SHORTENED, H=HER, E='acs2er_episodic'),
            primary='Mean over registered points within each seed, then mean and SE over twenty seeds.',
            final='Final registered point per seed, then mean and SE over twenty seeds.',
            comparisons=[['S@selected', 'S@2/3'], ['S@selected', 'E'], ['S@selected', 'H@2/3'], ['S@2/3', 'H@1/2'], ['H@1/2', 'H@2/3']],
            coincident_arms='Omit S@selected minus S@2/3 where selection is 2/3; these arms coincide.',
            paired_interval=dict(degrees_of_freedom=19, critical_value=2.093024054408263, coverage=0.95, multiplicity_correction='none; marginal intervals'),
            dose_interpretation='H@1/2 approximately matches realized copies of S@2/3. Report actual shares; this is a rule contrast at similar dose, not exact dose matching.',
            decomposition='At each seed and point, and both aggregate views, verify exactly with Fraction: (S@2/3 - H@1/2) + (H@1/2 - H@2/3) = S@2/3 - H@2/3.',
            thresholds=her['analysis']['thresholds'], baseline_threshold_crossing_checks=her['analysis']['baseline_threshold_crossing_checks'],
            threshold_arms=['S@selected', 'H@1/2'], threshold_arithmetic='Integer exhaustive successes over fixed test pair counts; Fraction for mean curves and >= both baseline final mean test thresholds. First nominal point, no interpolation, null if unreached.',
            tuning_table='Each maze/candidate: mean and SE of five seed checkpoint-mean validation scores; descriptive equally weighted across-maze mean per candidate.',
            diagnostics=['realized copies / all draws', 'trivial copies / copies', 'copies after counterfactual end / copies', 'fallbacks / all draws', 'rewarded copies / copies',
                'wall bumps / copies where measured; historical missing values remain null', 'first-action value minus actual return on test: all, successful and failed pairs',
                'knowledge', 'population', 'memory', 'time', 'allocation CPUTimeRAW for every attempt', 'objective calls by scoring, admissibility selection and provenance'],
            diagnostic_scope='Every new/reused arm and tuning setting in both registered views; tuning has validation-only value diagnostics, test value diagnostics are unmeasured until a three-block run. Empty conditional groups stay null with contributing seed counts.',
            diagnostic_aggregation='Compute route shares from counters within each row. In checkpoint_mean average rows equally within seed, then report mean/SE across seeds; final_point uses the last row. Values, knowledge, population and memory follow the same views. Also report cumulative final cost totals. Empty groups are null with contributing seed counts.',
            failed_value_groups='Recover failed-pair conditional means from exhaustive integer success counts and the archived all/successful means by algebraic decomposition; empty groups null, binary64 rounding remains. No extra evaluation calls.',
            costs='Report pilot, tuning, conditional test and control separately; process CPU=user+system, allocated CPU from sacct. Reused historical costs separate, never charged again.',
            post_hoc='Every unregistered comparison, including all success comparisons against ACS2, is labelled post hoc.'),
        identity_checks=dict(reference_two_thirds='Pilot seed42 and all tuning seeds42--46 at p=2/3 reproduce every admissibility validation, learning and knowledge field in IEEE754 binary64 bits at every point. STOP before selection on mismatch.',
            selected='Test seeds42--46 reproduce validation-only tuning learning/knowledge and validation; seed42 reproduces all pilot blocks at the same setting.',
            control='Every seed42 point and all three blocks reproduce pilot.',
            exclusions='Only grid_identity.OMIT: source/host/time provenance and arm annotation; no learning or knowledge field excluded.'),
        execution=dict(partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', concurrency=20, batch_size=20,
            tooling='experiments/grid/', clone='~/acs2-tu', target='x86_64-unknown-linux-musl',
            dispatch='Lean stage-2 dispatcher: one registered 20--40-job array per active stream, at most two streams and forty running jobs. Wait for a stream to become terminal before its next array; do not enqueue the entire multi-array manifest. Re-derive and check admission before each sbatch.',
            scope_roots=scope_roots(), accounting='One scope containing all pilot, tuning, test and control roots; pilot offset zero to avoid double charging.',
            admission='Before each sbatch mechanically rederive approved manifest; check grant, sbatch --test-only and pilot + used + reserved + new hard limits <= approved CPU-hours.',
            guards='Immutable clean source/binary/operations, hardware/preset/row/identity guards, shared STOP on real failure; 20--40 jobs at a time.',
            archive='~/tu_runs/hindsight-proportion/ and ~/Desktop/tu-runs/hindsight-proportion/; independently verify both copies before analysis.'),
        authorization='Stage 1 pilot only is authorized, at most 5 CPU-hours including reservations. Stop after registration; tuning, control and test require explicit approval of plan bytes and CPU-hour limit.')
