import hashlib
import json
import math
from pathlib import Path

from thresholds import BASELINE_CROSSINGS, THRESHOLDS

AGENTS = ['er_episodic', 'her_full', 'her_values', 'her_values_shortened', 'acs2er_u1']
VALUES = ['her_values', 'her_values_shortened']
MATCHED_CONTROL = 'her_values_p2of3'
PROPORTIONS = [1 / 3, 2 / 3, 0.9]
PAIR_SHA256 = 'eb1d2c01669b4471f68cbc1f671b81fea77e71cfc8e72b75e4f17cd5dcf6b189'
BASELINE_SHA256 = '913ce814484b742265ba88db09d72d4feaf8f981b200811454f9fce11b6b5a78'
ROOT = Path(__file__).resolve().parents[2]


def configurations():
    path = ROOT / 'experiments/maze-baselines/pair-pilot.json'
    registration = ROOT / 'experiments/maze-baselines/pair-protocol.json'
    if hashlib.sha256(registration.read_bytes()).hexdigest() != PAIR_SHA256:
        raise ValueError('registered pair protocol changed')
    result = json.loads(path.read_text())['configurations']
    for config in result:
        config.pop('limits_minutes')
    return result


def pilot():
    configs = configurations()
    for config in configs:
        config['limits_minutes'] = {agent: 15 for agent in AGENTS}
    return dict(stage='authorized_cost_pilot', experiment='her-variants', seeds=[42], agents=AGENTS,
                partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', memory_mib=2048,
                concurrency=30, phase_cpu_hour_limit=60, pilot_allocation_cpu_seconds=0,
                accounting_since='2026-10-12', registration_sha256=PAIR_SHA256,
                configurations=configs, cost_pilot_proportion=2 / 3)


def select_proportion(scores, candidates=PROPORTIONS):
    if set(scores) != set(candidates) or any(not math.isfinite(score) for score in scores.values()):
        raise ValueError('incomplete or nonfinite validation scores')
    maximum = max(scores.values())
    return min(proportion for proportion, score in scores.items() if maximum - score <= 1e-12)


def forecast(cells, spent_seconds, proportions=PROPORTIONS):
    if set(cells) != {(config['id'], agent) for config in configurations() for agent in AGENTS}:
        raise ValueError('cost pilot must cover every agent and maze')
    grid_hard = tuning_hard = grid_expected = tuning_expected = 0.0
    limits, memory = {}, 512
    control_costs = {}
    for key, cell in cells.items():
        body = cell['body_wall_seconds']
        allocated = cell['allocation_cpu_seconds']
        if body <= 0 or allocated < 0 or cell['peak_rss_kib'] <= 0:
            raise ValueError('invalid pilot cost or memory')
        limit = max(2, math.ceil((1.5 * body + 30) / 60))
        limits[key] = limit
        memory = max(memory, math.ceil(3 * cell['peak_rss_kib'] / 1024 / 128) * 128)
        estimate = body + max(0, allocated - cell['process_cpu_seconds'])
        grid_expected += 20 * estimate
        grid_hard += 20 * limit * 60
        if key[1] in VALUES:
            tuning_expected += 5 * len(proportions) * estimate
        if key[1] == 'her_values':
            control_costs[key[0]] = dict(expected_cpu_hours=20 * estimate / 3600,
                                        reservation_cpu_seconds=20 * limit * 60)
    for config in configurations():
        tuning_hard += 2 * 5 * len(proportions) * max(limits[config['id'], agent] for agent in VALUES) * 60
    control_expected = sum(cell['expected_cpu_hours'] for cell in control_costs.values())
    control_hard = sum(cell['reservation_cpu_seconds'] for cell in control_costs.values())
    return dict(grid_expected_cpu_hours=grid_expected / 3600, tuning_expected_cpu_hours=tuning_expected / 3600,
                doubled_forecast_with_pilots_cpu_hours=(spent_seconds + 2 * (grid_expected + tuning_expected)) / 3600,
                hard_bound_with_pilots_cpu_hours=(spent_seconds + grid_hard + tuning_hard) / 3600,
                pilot_allocation_cpu_seconds=spent_seconds, grid_reservation_cpu_seconds=grid_hard,
                tuning_reservation_cpu_seconds=tuning_hard, memory_mib=memory,
                additional_control_maximum_runs=120, additional_control_costs=control_costs,
                additional_control_expected_cpu_hours=control_expected,
                additional_control_reservation_cpu_seconds=control_hard,
                expected_with_all_additional_controls_cpu_hours=(spent_seconds + grid_expected + tuning_expected) / 3600 + control_expected,
                doubled_forecast_with_all_additional_controls_cpu_hours=(spent_seconds + 2 * (grid_expected + tuning_expected)) / 3600 + 2 * control_expected,
                hard_bound_with_all_additional_controls_cpu_hours=(spent_seconds + grid_hard + tuning_hard + control_hard) / 3600,
                limits_minutes={config['id']: {agent: limits[config['id'], agent] for agent in AGENTS} for config in configurations()})


def fit_forecast(cells, spent_seconds):
    for candidates in [PROPORTIONS, PROPORTIONS[:2]]:
        prediction = forecast(cells, spent_seconds, candidates)
        if max(prediction['hard_bound_with_pilots_cpu_hours'], prediction['doubled_forecast_with_pilots_cpu_hours']) <= 60:
            return candidates, prediction
    raise ValueError('the fixed final grid plus two tuning proportions cannot fit 60 CPU-hours')


def study(thresholds, cells, spent_seconds):
    candidates, prediction = fit_forecast(cells, spent_seconds)
    configs = configurations()
    if thresholds != THRESHOLDS:
        raise ValueError('both baseline thresholds must be the registered exact fractions')
    for config in configs:
        config['limits_minutes'] = prediction['limits_minutes'][config['id']]
    return dict(stage='grid_pending_approval', experiment='her-variants', phase_cpu_hour_limit=60,
                pair_protocol_sha256=PAIR_SHA256, baseline_plan_sha256=BASELINE_SHA256,
                agents=AGENTS, configurations=configs, seeds=list(range(42, 62)),
                tuning=dict(agents=VALUES, seeds=list(range(42, 47)), proportions=candidates,
                            evaluation_blocks=['validation'], selection='For each maze and values variant separately, maximize equally weighted checkpoint-mean validation success, averaged equally over five seeds. Scores within 1e-12 tie; choose the smallest requested relabel proportion.',
                            tie_tolerance=1e-12, tie='smaller_proportion', budgets='same registered 2M/4M budgets and points as final grid',
                            her_full='fixed 2/3; no tuning', test_access='none'),
                final_grid=dict(runs=600, seeds=list(range(42, 62)), agents=AGENTS,
                                evaluation_blocks=['training', 'validation', 'test'], tuned_parameters='apply the registered per-maze per-variant selection mechanically'),
                matched_proportion_control=dict(arm=MATCHED_CONTROL, measurement_agent='her_values',
                    relabeled_proportion=2 / 3, relabeled_proportion_exact='2/3', seeds=list(range(42, 62)),
                    condition='For each maze whose selected her_values proportion differs from 2/3, run twenty additional seeds at 2/3. Otherwise reuse the tuned her_values rows as this arm.',
                    protocol='same registered pairs, cap, budget and nominal points as the main grid on that maze',
                    tuning='none; this arm enters only the matched update-rule comparison', maximum_additional_runs=120,
                    row_identity=dict(agent='her_values', arm=MATCHED_CONTROL),
                    row_storage='Keep unchanged binary-rows.jsonl and an annotated rows.jsonl with arm=her_values_p2of3; checksum both. The measurement binary and its agent identifiers stay pinned.',
                    admission='Before every submission, pilots plus sacct spent plus all outstanding hard reservations plus every new hard limit must be at most 60 CPU-hours; include tuning, main grid, controls and every retry across batch roots.',
                    timing='Submit each required maze arm as soon as admission passes after selection; recheck when accounting changes and at main-grid completion, when spent replaces reservations. If required controls never fit, STOP and report the missing arms without shrinking any other scope.'),
                analysis=dict(primary='test success averaged equally over points within each seed, then mean and SE over twenty seeds',
                              paired_interval='marginal Student t(19) 95%; critical value 2.093024054408263; no multiplicity correction',
                              comparisons=[['her_full','er_episodic'], ['her_values','er_episodic'], ['her_values_shortened','er_episodic'],
                                           ['her_values','her_full'], ['her_values_shortened','her_values'], ['er_episodic','acs2er'],
                                           ['er_episodic','acs2'], ['acs2er_u1','acs2er'], ['acs2er_u1','acs2'],
                                           [MATCHED_CONTROL,'her_full']],
                              comparison_interpretation={'her_values_p2of3_minus_her_full': 'update-rule ablation at the same requested proportion 2/3 on every maze',
                                                         'her_values_minus_her_full': 'separately tuned values control against the fixed predecessor proportion; both proportion and copy update may differ'},
                              threshold='For each of the two exact fractions, take the first registered nominal point at which the exact mean test success over seeds 42--61 reaches it; no interpolation; null if never reached. Report both for every measured arm and both archived baselines.',
                              threshold_arithmetic='Recover each integer success count from exhaustive success times registered test pairs, requiring distance from the nearest integer at most 1e-8. All ratios, seed means and >= comparisons then use fractions.Fraction; never compare floating means with thresholds.',
                              thresholds=thresholds, baseline_threshold_crossing_checks=BASELINE_CROSSINGS,
                              baseline_reuse='archived ACS2 and ACS2ER pair grid only; seed-42 replay is a regression check, not a new baseline grid',
                              values_update_volume='Report observed full steps / actual environment steps and knowledge curves beside every comparison involving values copies. Every-transition copies have expected full volume 3*(1-p); shortening fallbacks increase this volume, so use realized route counts.',
                              secondary=['knowledge','realized_relabeled_share','composition_by_route','objective_cost_by_purpose',
                                         'population','memory','updates_by_kind','full_steps_per_environment_step','knowledge_curves','process_time','allocation_cpu_hours']),
                identity_checks=dict(comparison='IEEE 754 binary64 bits for floating fields; exact equality otherwise. Exclude only source/hardware identity, three wall times and the separately validated control arm label. Compare every learning, evaluation and knowledge field on shared points.',
                    pilot='Seed 42 for er_episodic, her_full and acs2er_u1 must reproduce the verified pilot at all registered points, using the observed U1 pilot.',
                    tuning='For each maze and values variant at its selected proportion, seeds 42--46 must reproduce validation-only tuning rows at every shared point; include validation evaluation, learning and knowledge fields.',
                    selected_two_thirds='Where a selected values proportion is 2/3, seed 42 also reproduces all three evaluation blocks and learning/knowledge fields of the pilot.',
                    matched_control='her_values_p2of3 seeds 42--46 reproduce her_values tuning rows at 2/3 at shared points; seed 42 also reproduces the pilot. Reused tuned rows satisfy the same checks.',
                    failure='STOP on any mismatch before analysis; retain the differing artifacts and account for the attempt'),
                execution=dict(partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', concurrency=20, batch_size=20,
                               tuning_batches='one group per maze, both values variants and all proportions; 20 or 30 jobs, each reserves the larger values-agent limit',
                               limit_rule='max(2, ceil((1.5 * seed42 body wall at registered budget + 30) / 60)) minutes; node process watchdog retains 30 seconds; this is a sizing margin, not a guarantee about new-seed growth',
                               guards='shared experiments/grid tools; clean pinned musl source and binary, node-side guards, STOP on failure; no lease, pinning or laptop dependency',
                               accounting='allocation CPUTimeRAW once per attempt, including failed attempts and regression pilots; spent plus reserved plus new limits at most 60 CPU-hours',
                               additional_control_budget='The extra control is conditional on tuning. Maximum simultaneous hard reservations can exceed sixty; use measured spent cost as main-grid reservations are released. The 1.5 sizing rule and doubled forecasts are diagnostics, not permission to exceed the admission bound.',
                               copies='cluster and local raw artifacts independently checksum-verified'),
                forecast=prediction, tuning_set_shrunk=candidates != PROPORTIONS)
