import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import load_plan

EXPERIMENT = 'replay-schedule-and-order'
PROPORTION_SHA256 = '894596d2f792a2327ff5fd061c4344a17bcb96b01b9a96dd4f58a61a1e3a2888'
SHORTENED = 'acs2her_shortened'
ER = 'acs2er_episodic'
BLOCKS = ['training', 'validation', 'test']
SCHEDULES = ['3to1at30000', '3to1at100000', '3to1at300000']
SCREEN_SETTINGS = SCHEDULES + ['backward']
PILOT_SETTINGS = [(SHORTENED, mode) for mode in SCREEN_SETTINGS + ['random-pass']] + [(ER, mode) for mode in SCREEN_SETTINGS] + [(SHORTENED, 'base')]
CORRECTED_PILOT_INDICES = [3, 4, 8, 9]
BASE_PROPORTIONS = dict(maze4=0.5, maze5=0.8, maze6=0.8, maze7=0.8, woods14=0.8, mazema=1 / 3)


def configurations():
    return copy.deepcopy(load_plan(ROOT / 'experiments/hindsight-proportion/plan.json', PROPORTION_SHA256)['configurations'])


def proportion_label(p):
    return {0.5: 'p1of2', 0.8: 'p4of5', 1 / 3: 'p1of3'}[p]


def label(config, agent, mode):
    if (agent, mode) not in PILOT_SETTINGS:
        raise ValueError('unregistered agent and replay setting')
    base = agent + ('_' + proportion_label(BASE_PROPORTIONS[config['task']]) if agent == SHORTENED else '')
    if mode == 'base':
        return base
    suffix = 'replay' + mode if mode in SCHEDULES else 'backward' if mode == 'backward' else 'pass_random'
    return base + '_' + suffix


def setting(config, agent, mode, blocks, limit=10):
    arm = label(config, agent, mode)
    settings = dict(relabeled_proportion=BASE_PROPORTIONS[config['task']]) if agent == SHORTENED else {}
    if mode != 'base':
        settings.update(replay_schedule=mode if mode in SCHEDULES else 'constant',
                        replay_order='independent' if mode in SCHEDULES else mode)
    return dict(copy.deepcopy(config), id=config['id'] + '_' + arm, registered_id=config['id'],
        row_arm=arm, evaluation_blocks=list(blocks), agent_settings={agent: settings}, limits_minutes={agent: limit})


def pilot_roots():
    return ['pilot-' + agent + '-' + mode for agent, mode in PILOT_SETTINGS]


def scope_roots(previous=None, failed=None):
    pilots = pilot_roots() if previous is None else [str(Path(previous) / name) for name in pilot_roots()] + [pilot_roots()[index] for index in CORRECTED_PILOT_INDICES]
    if failed is not None:
        pilots.append(str(Path(failed) / pilot_roots()[-1]))
    return pilots + ['screening'] + ['test-' + axis + '-' + config['id'] + '-' + slot
        for axis in ('B', 'C') for config in configurations()
        for slot in (['shortened', 'er'] if axis == 'B' else ['shortened', 'er', 'order-control'])]


def pilot(index):
    if type(index) is not int or not 0 <= index < len(PILOT_SETTINGS):
        raise ValueError('unregistered pilot arm')
    agent, mode = PILOT_SETTINGS[index]
    return dict(stage='authorized_cost_pilot', experiment=EXPERIMENT, pilot_index=index, seeds=[42], agents=[agent],
        configurations=[setting(config, agent, mode, BLOCKS) for config in configurations()],
        partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=2048,
        concurrency=20, phase_cpu_hour_limit=12, pilot_allocation_cpu_seconds=0,
        accounting_since='2026-10-28', proportion_plan_sha256=PROPORTION_SHA256)


def grouped(cells):
    ordered = sorted(cells, key=lambda cell: (cell['limit_minutes'], cell['expected_cpu_seconds'],
                                             cell['configuration_id'], cell['mode']))
    best = {0: (0, [])}
    for end in range(4, len(ordered) + 1):
        choices = []
        for size in range(4, 9):
            begin = end - size
            if begin in best:
                cost, spans = best[begin]
                choices.append((cost + size * ordered[end - 1]['limit_minutes'], spans + [(begin, end)]))
        if choices:
            best[end] = min(choices, key=lambda choice: (choice[0], len(choice[1]), choice[1]))
    if len(ordered) not in best or not ordered:
        raise ValueError('screening must group twenty to forty jobs')
    return [ordered[begin:end] for begin, end in best[len(ordered)][1]]


def forecast(cells, pilot_seconds):
    if not 0 < pilot_seconds <= 12 * 3600:
        raise ValueError('pilot must fit its twelve-hour allocation cap')
    estimates = []
    memory = 512
    for config in configurations():
        for agent, mode in PILOT_SETTINGS:
            cost = cells[config['id'], label(config, agent, mode)]
            if any(not math.isfinite(cost[key]) or cost[key] <= 0 for key in
                   ('body_wall_seconds', 'allocation_cpu_seconds', 'process_cpu_seconds', 'peak_rss_kib')):
                raise ValueError('unmeasured pilot cell')
            memory = max(memory, math.ceil(3 * cost['peak_rss_kib'] / 1024 / 128) * 128)
            if mode == 'base':
                continue
            purposes = ['screening', 'test'] if agent == SHORTENED and mode in SCREEN_SETTINGS else ['test']
            for purpose in purposes:
                seeds = 5 if purpose == 'screening' else 20
                limit = max(3, math.ceil((2 * cost['body_wall_seconds'] + 60) / 60))
                estimates.append(dict(purpose=purpose, configuration_id=config['id'], agent=agent, mode=mode,
                    runs=seeds, expected_cpu_seconds=seeds * cost['allocation_cpu_seconds'],
                    limit_minutes=limit, reservation_cpu_seconds=seeds * limit * 60))
    screening = [cell for cell in estimates if cell['purpose'] == 'screening']
    for group in grouped(screening):
        limit = max(cell['limit_minutes'] for cell in group)
        group_id = 'screening-' + group[0]['configuration_id'] + '-' + group[0]['mode']
        for cell in group:
            cell.update(batch_group=group_id, limit_minutes=limit,
                        reservation_cpu_seconds=cell['runs'] * limit * 60)
    worst = []
    for config in configurations():
        tests = [cell for cell in estimates if cell['purpose'] == 'test' and cell['configuration_id'] == config['id']]
        for axis in ('B', 'C'):
            modes = SCHEDULES if axis == 'B' else ['backward']
            alternatives = []
            for mode in modes:
                group = [cell for cell in tests if cell['mode'] == mode or (axis == 'C' and cell['mode'] == 'random-pass')]
                alternatives.append(dict(mode=mode, expected_cpu_seconds=sum(cell['expected_cpu_seconds'] for cell in group),
                    reservation_cpu_seconds=sum(cell['reservation_cpu_seconds'] for cell in group)))
            worst.append(dict(configuration_id=config['id'], axis=axis,
                expected_cpu_seconds=max(item['expected_cpu_seconds'] for item in alternatives),
                reservation_cpu_seconds=max(item['reservation_cpu_seconds'] for item in alternatives), alternatives=alternatives))
    expected = sum(cell['expected_cpu_seconds'] for cell in screening + worst)
    reserve = sum(cell['reservation_cpu_seconds'] for cell in screening + worst)
    doubled, hard = pilot_seconds + 2 * expected, pilot_seconds + reserve
    return dict(cells=estimates, worst_case_tests=worst, memory_mib=memory,
        expected_total_cpu_hours=(pilot_seconds + expected) / 3600, doubled_forecast_cpu_hours=doubled / 3600,
        hard_reservations_cpu_hours=hard / 3600, requested_cpu_hour_limit=5 * math.ceil(max(doubled, hard) / 18000),
        screening_runs=120, maximum_test_runs=600,
        estimate_rule='Seed42 allocation CPUTimeRAW; three-block pilots conservatively proxy validation-only screening. Worst selected B schedule and all three C arms per maze, both axes tested everywhere.',
        screening_group_rule='Sort cells by hard limit and pilot allocation cost, then choose contiguous groups of four to eight five-seed cells minimizing total hard reservations; ties use fewer arrays, then earlier boundaries. No success values enter grouping.',
        limit_rule='max(3, ceil((2 * body wall seconds + 60) / 60)) minutes, common maximum within each screening array; one twenty-seed array per test arm.',
        limitation='One seed measures cost, not seed variability or scientific benefit. Every attempt and outstanding reservation remains charged.')


def study(cells, pilot_seconds, identity, evidence):
    previous = load_plan(ROOT / 'experiments/hindsight-proportion/plan.json', PROPORTION_SHA256)
    return dict(stage='grid_pending_approval', experiment=EXPERIMENT, **identity, configurations=configurations(),
        base_proportions=BASE_PROPORTIONS, proportion_plan_sha256=PROPORTION_SHA256, cost_pilot=evidence,
        screening=dict(agent=SHORTENED, settings=SCREEN_SETTINGS, seeds=list(range(42, 47)),
            evaluation_blocks=['validation'], runs=120,
            base='Reuse approved proportion-study validation-only tuning rows at the saved per-maze p*. Pilot default identity permits reuse.',
            selection='Per maze and axis: mean over registered points within seed, then mean over five validation seeds; maximize; candidates within 1e-12 of maximum tie, base wins ties, otherwise earlier registered setting.'),
        test=dict(seeds=list(range(42, 62)), evaluation_blocks=BLOCKS, maximum_runs=600,
            B=dict(settings=SCHEDULES, arms=[SHORTENED, ER]),
            C=dict(settings=['backward'], arms=[SHORTENED, ER, SHORTENED + '_pass_random']),
            derivation='Recompute saved selection from complete screening and frozen validation-only base rows. Run only selected non-base axes and mazes; base selections cause no allocations.'),
        references=dict(proportion_plan_sha256=PROPORTION_SHA256, her_plan_sha256=previous['references']['her_plan_sha256'],
            baseline_plan_sha256=previous['references']['baseline_plan_sha256'], names='experiments/grid/agent_names.json',
            arms=['S at p* from proportion test', 'acs2er_episodic from HER study', 'acs2 and acs2er from pair baseline'],
            policy='Reuse twenty-seed verified references without rerunning them. The six default pilot trajectories are identity controls, not new reference estimates.'),
        forecast=forecast(cells, pilot_seconds),
        analysis=dict(block='test', seeds=list(range(42, 62)), views=['checkpoint_mean', 'final_point'],
            primary='Average all registered points equally within seed, then mean and SE across twenty seeds.',
            final='Use last registered point per seed, then mean and SE across twenty seeds.',
            comparisons=['S_X - S', 'E_X - E', '(S_X - S) - (E_X - E)', 'S_X - E_X'],
            C_additional='S_backward - S_order_control',
            aliases=dict(S='shortened HER at saved p* with constant independent replay', E='episodic ER with constant independent replay'),
            paired_interval=dict(degrees_of_freedom=19, critical_value=2.093024054408263, coverage=0.95, multiplicity_correction='none; marginal intervals'),
            thresholds=previous['analysis']['thresholds'], baseline_threshold_crossing_checks=previous['analysis']['baseline_threshold_crossing_checks'],
            threshold_arithmetic='For S_X only: exact exhaustive integer counts and Fraction means; inclusive >= both registered baseline final means; first nominal point, no interpolation, null if unreached.',
            screening_table='Mean and SE of five validation checkpoint-mean seed scores for every maze and setting, including reused base; no screening test-block access.',
            diagnostics=['realized updates / interactions for every checkpoint interval from cumulative differences',
                'requested pass share, realized copy share, fallbacks, trivial copies, post-end copies, rewarded copies, wall copies',
                'first-action value minus return for all, successful and failed test pairs', 'knowledge', 'population and numerosity',
                'memory lower bounds and peak RSS', 'training and evaluation times', 'objective calls by scoring, selection and provenance',
                'allocated CPUTimeRAW, process user+system CPU and cost by pilot, screening, test and reused historical purpose'],
            diagnostic_aggregation='Within each row use appropriate route denominators; checkpoint_mean averages equally within seed; final_point uses only last row; then mean/SE over seeds. Empty groups stay null with counts.',
            interpretation='B deliberately changes update volume. Same settings equalize the rule and prescribed volume between S_X and E_X; realized totals may differ with episode overshoot and episode-end switch rounding. C compares full passes against independent sampling; random-pass control isolates within-pass order.',
            limitations=['Uniform retained-episode selection gives equal expected exposure per retained transition for complete original-goal passes. Counterfactual prefixes and the final budget cut add goal- and position-dependent weighting.',
                'One-goal passes change copy weighting: p is per pass, realized per-update share need not equal p. Report pass and draw counts separately.',
                'Backward and random control share pass samples and goals conditional on identical stores and sampler state; learned policies and subsequent real trajectories can diverge.',
                'Intervals condition on selected settings and do not include selection uncertainty. Finite training horizons do not establish convergence.'],
            post_hoc='Every other comparison, including every success comparison against ACS2, is labelled post hoc.'),
        identity_checks=dict(base='Every default seed42 row reproduces proportion-study selected-p test learning, knowledge and all three blocks in binary64 bits. STOP on mismatch.',
            screening='Seed42 validation and shared learning fields reproduce the full-block pilot.',
            test='Seed42 all blocks reproduce pilot; seeds42--46 reproduce screening on validation and every learning/knowledge field for S_X.',
            exclusions='Only grid_identity.OMIT: source/hardware/time provenance and arm annotation.'),
        execution=dict(partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', clone='~/acs2-tu',
            target='x86_64-unknown-linux-musl', tooling='experiments/grid/', scope_roots=scope_roots(evidence.get('previous_pilot_root'), evidence.get('failed_pilot_root')),
            dispatch='One-shot dispatcher, one registered 20--40-job array per invocation; at most two active streams and forty running jobs. No dependent queue of future arrays.',
            admission='Re-derive each manifest from exact approved bytes and recomputed saved selection before every sbatch; grant and sbatch --test-only; pilot + used + reserved + new hard limits within approved cap.',
            accounting='One scope over every root; zero pilot offset; all allocations and failed attempts included exactly once; hard reservations include whole array maxima.',
            guards='Pinned clean source, static binary and immutable operations; node guards; shared STOP on real failure; diagnosed recovery and bitwise partial-prefix equality before retry acceptance.',
            archive='~/tu_runs/replay-schedule-and-order/ and ~/Desktop/tu-runs/replay-schedule-and-order/; independently verify both copies.'),
        authorization='Stage1 only: the original sixty and corrected twenty-four seed42 pilot allocations share twelve CPU-hours including reservations. Screening and test require explicit approval of this plan and a cumulative CPU-hour limit; stop after registration.')
