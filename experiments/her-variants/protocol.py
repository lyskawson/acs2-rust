import hashlib
import json
import math
from pathlib import Path

AGENTS = ['er_episodic', 'her_full', 'her_values', 'her_values_shortened', 'acs2er_u1']
VALUES = ['her_values', 'her_values_shortened']
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
    for key, cell in cells.items():
        body = cell['body_wall_seconds']
        allocated = cell['allocation_cpu_seconds']
        if body <= 0 or allocated < 0 or cell['peak_rss_kib'] <= 0:
            raise ValueError('invalid pilot cost or memory')
        limit = max(2, math.ceil((2.5 * body + 60) / 60))
        limits[key] = limit
        memory = max(memory, math.ceil(3 * cell['peak_rss_kib'] / 1024 / 128) * 128)
        estimate = body + max(0, allocated - cell['process_cpu_seconds'])
        grid_expected += 20 * estimate
        grid_hard += 20 * limit * 60
        if key[1] in VALUES:
            tuning_expected += 5 * len(proportions) * estimate
            tuning_hard += 5 * len(proportions) * limit * 60
    return dict(grid_expected_cpu_hours=grid_expected / 3600, tuning_expected_cpu_hours=tuning_expected / 3600,
                doubled_forecast_with_pilots_cpu_hours=(spent_seconds + 2 * (grid_expected + tuning_expected)) / 3600,
                hard_bound_with_pilots_cpu_hours=(spent_seconds + grid_hard + tuning_hard) / 3600,
                pilot_allocation_cpu_seconds=spent_seconds, grid_reservation_cpu_seconds=grid_hard,
                tuning_reservation_cpu_seconds=tuning_hard, memory_mib=memory,
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
    if set(thresholds) != {config['id'] for config in configs}:
        raise ValueError('missing baseline thresholds')
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
                analysis=dict(primary='test success averaged equally over points within each seed, then mean and SE over twenty seeds',
                              paired_interval='marginal Student t(19) 95%; critical value 2.093024054408263; no multiplicity correction',
                              comparisons=[['her_full','er_episodic'], ['her_values','er_episodic'], ['her_values_shortened','er_episodic'],
                                           ['her_values','her_full'], ['her_values_shortened','her_values'], ['er_episodic','acs2er'],
                                           ['er_episodic','acs2'], ['acs2er_u1','acs2er'], ['acs2er_u1','acs2']],
                              threshold='first nominal point at which across-seed mean test success reaches archived phase-4b ACS2 mean final test success; no interpolation; null if never reached',
                              thresholds=thresholds, baseline_reuse='archived phase-4b grid only; seed-42 replay is a regression check, not a new baseline grid',
                              secondary=['knowledge','realized_relabeled_share','composition_by_route','objective_cost_by_purpose',
                                         'population','memory','updates_by_kind','process_time','allocation_cpu_hours']),
                execution=dict(partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor', concurrency=20, batch_size=20,
                               guards='shared experiments/grid tools; clean pinned musl source and binary, node-side guards, STOP on failure; no lease, pinning or laptop dependency',
                               accounting='allocation CPUTimeRAW once per attempt, including failed attempts and regression pilots; spent plus reserved plus new limits at most 60 CPU-hours',
                               copies='cluster and local raw artifacts independently checksum-verified'),
                forecast=prediction, tuning_set_shrunk=candidates != PROPORTIONS)
