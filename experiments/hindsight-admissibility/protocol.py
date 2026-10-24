import copy
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'experiments/grid'))
from grid_protocol import load_plan

AGENTS = ['acs2her_nontrivial', 'acs2her_shortened']
PAIR_SHA256 = 'eb1d2c01669b4471f68cbc1f671b81fea77e71cfc8e72b75e4f17cd5dcf6b189'
HER_SHA256 = 'e18aca0f513ac3f6b6c0b9a52063acc78d9dc1becf369bcd5e755f5050ac6444'
BASELINE_SHA256 = '913ce814484b742265ba88db09d72d4feaf8f981b200811454f9fce11b6b5a78'


def configurations():
    registration = load_plan(ROOT / 'experiments/maze-baselines/pair-protocol.json', PAIR_SHA256)
    source = json.loads((ROOT / 'experiments/maze-baselines/pair-pilot.json').read_text())
    configs = copy.deepcopy(source['configurations'])
    for config in configs:
        config.pop('limits_minutes')
        if config['registration'] not in registration['configurations']:
            raise ValueError('pilot configuration differs from registered pairs')
    return configs


def pilot():
    configs = configurations()
    for config in configs:
        config['limits_minutes'] = {agent: 10 for agent in AGENTS}
    return dict(stage='authorized_cost_pilot', experiment='hindsight-admissibility',
                seeds=[42], agents=AGENTS, configurations=configs,
                partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor',
                memory_mib=2048, concurrency=20, phase_cpu_hour_limit=2,
                pilot_allocation_cpu_seconds=0, accounting_since='2026-10-24',
                registration_sha256=PAIR_SHA256)


def forecast(cells, spent_seconds):
    expected = {(config['id'], agent) for config in configurations() for agent in AGENTS}
    if set(cells) != expected or not 0 < spent_seconds <= 7200:
        raise ValueError('forecast requires all twelve pilot cells and their allocation cost')
    estimates, limits, memory = [], {}, 512
    for config in configurations():
        limits[config['id']] = {}
        for agent in AGENTS:
            cost = cells[config['id'], agent]
            body, process, allocation, rss = [cost[key] for key in
                ('body_wall_seconds', 'process_cpu_seconds', 'allocation_cpu_seconds', 'peak_rss_kib')]
            if not all(math.isfinite(value) and value > 0 for value in (body, process, allocation, rss)):
                raise ValueError('invalid measured pilot cost')
            limit = max(3, math.ceil((2 * body + 60) / 60))
            estimate = body + max(0, allocation - process)
            limits[config['id']][agent] = limit
            estimates.append(dict(configuration_id=config['id'], agent=agent,
                                  expected_grid_cpu_seconds=20 * estimate,
                                  reservation_cpu_seconds=20 * limit * 60))
            memory = max(memory, math.ceil(3 * rss / 1024 / 128) * 128)
    expected_seconds = sum(cell['expected_grid_cpu_seconds'] for cell in estimates)
    reservations = sum(cell['reservation_cpu_seconds'] for cell in estimates)
    doubled = spent_seconds + 2 * expected_seconds
    hard = spent_seconds + reservations
    return dict(cells=estimates, pilot_allocation_cpu_seconds=spent_seconds,
                grid_expected_cpu_hours=expected_seconds / 3600,
                doubled_forecast_with_pilot_cpu_hours=doubled / 3600,
                hard_bound_with_pilot_cpu_hours=hard / 3600,
                grid_reservation_cpu_seconds=reservations,
                requested_cpu_hour_limit=5 * math.ceil(max(doubled, hard) / 3600 / 5),
                memory_mib=memory, limits_minutes=limits,
                limit_rule='max(3, ceil((2 * seed42 body wall seconds + 60) / 60)) minutes; shared node watchdog retains 30 seconds',
                limitation='One seed estimates cost, not future seed growth; all allocations and retries remain charged, with hard admission and STOP.')


def study(cells, spent_seconds, identity, pilot_evidence):
    prediction = forecast(cells, spent_seconds)
    configs = configurations()
    for config in configs:
        config['limits_minutes'] = prediction['limits_minutes'][config['id']]
    her = load_plan(ROOT / 'experiments/her-variants/plan.json', HER_SHA256)
    comparisons = [[agent, reference] for agent in AGENTS for reference in ('acs2her', 'acs2er_episodic')]
    comparisons.append(['acs2her_shortened', 'acs2her_nontrivial'])
    return dict(stage='grid_pending_approval', experiment='hindsight-admissibility', **identity,
        agents=AGENTS, seeds=list(range(42, 62)), configurations=configs,
        pair_protocol_sha256=PAIR_SHA256, tuning='none; full updates, future, p=2/3 fixed for both rules',
        runs=240, evaluation_blocks=['training', 'validation', 'test'], cost_pilot=pilot_evidence,
        references=dict(agents=['acs2her', 'acs2er_episodic', 'acs2', 'acs2er'],
            her_plan_sha256=HER_SHA256, baseline_plan_sha256=BASELINE_SHA256,
            names='Reuse archived her_full and er_episodic through experiments/grid/agent_names.json; never rerun reference grids.'),
        analysis=dict(views=['checkpoint_mean', 'final_point'], block='test',
            primary='Equally weighted mean over registered points within a seed, then mean and SE over seeds 42--61.',
            final='Last registered nominal point within each seed, then mean and SE over seeds 42--61.',
            comparisons=comparisons,
            paired_interval='Paired seed differences; marginal 95% Student t(19), critical value 2.093024054408263; no multiplicity correction.',
            thresholds=her['analysis']['thresholds'], baseline_threshold_crossing_checks=her['analysis']['baseline_threshold_crossing_checks'],
            threshold_arithmetic='Recover integer successes over registered exhaustive test pairs (residual <= 1e-8), then use fractions.Fraction for seed means and >= both archived final ACS2/ACS2ER means. First registered point; no interpolation; null if unreached.',
            diagnostics=['trivial copies / relabeled draws', 'after-counterfactual-end copies / relabeled draws',
                'fallbacks / all draws', 'terminal relabeled copies / relabeled draws',
                'unchanged-state relabeled draws / relabeled draws', 'terminal unchanged-state draws / unchanged-state relabeled draws',
                'first-action value minus actual discounted return on test: all, successful and failed pairs',
                'knowledge', 'population', 'memory', 'process and allocation time', 'objective calls by scoring, selection and provenance, including admissibility'],
            aggregation='Each cumulative route share divides summed counters within one row, then summarize equally across seeds in both views. Knowledge/population/memory/value diagnostics follow both views; time and cumulative cost also report final totals. Empty conditional groups remain null with contributing seed counts.',
            proposals='HER has no rejection stage: proposals equal accepted draws. No separate proposal distribution exists.',
            post_hoc='Anything unregistered is labelled post hoc. acs2her_shortened versus acs2her_values_shortened is post hoc only: p differs.',
            interpretation='Removing inadmissible goals also increases original-goal fallbacks; comparisons estimate the whole registered rule, not a causal decomposition of overestimation.'),
        identity_checks=dict(seed42='All pilot-overlapping learning, evaluation, knowledge, composition and wall fields agree in binary64 bits; source/hardware and wall times excluded.',
            references='Verify frozen registrations, complete selected attempts, raw hashes, exact pair assignments and all twenty seeds before analysis.', failure='STOP on mismatch; retain artifacts and charge every attempt.'),
        execution=dict(partition='lem-cpu-normal', cpu_model='AMD EPYC 9554 64-Core Processor',
            concurrency=20, batch_size=20, batches=12, tooling='experiments/grid; one maze and arm per twenty-seed batch',
            admission='Before every sbatch derive manifest from approved registration; verify scope membership before accounting, grant and sbatch --test-only; pilot + allocated CPUTimeRAW for every attempt + remaining hard reservations + new limits <= approved cumulative CPU-hour limit.',
            guards='Clean pinned musl clone ~/acs2-tu; immutable operations snapshots, source/binary/preset/hardware/rows/pilot guards, shared STOP on real failure. No lease, host pinning or laptop dependency.',
            artifacts='~/tu_runs/5b/ and ~/Desktop/tu-runs/5b/; pull and independently checksum-verify both copies.',
            references='Historical reference allocations are reused and reported separately; no new charge or rerun.'),
        forecast=prediction, requested_cpu_hour_limit=prediction['requested_cpu_hour_limit'],
        authorization='Stage 1 only is authorized. The 240-run grid requires user approval of these plan bytes and a cumulative CPU-hour limit.')
