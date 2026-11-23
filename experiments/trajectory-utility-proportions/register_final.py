import argparse
import csv
import json
from pathlib import Path

from proportions import HERE, SCOUTS, checksum, pilot_cells, read


def final_plan(costs, identity, pilot_root):
    forecast = read(costs / 'forecast.json')
    evidence = read(identity)
    _, _, _, runs = pilot_cells()
    pilot = read(pilot_root / 'pilot-summary.json')
    pilot_digest = checksum(pilot_root / 'files.sha256.json')
    if (forecast['state'] != 'complete' or forecast['pilot_cells'] != len(runs)
            or forecast['forecast_cells'] != 3360 or forecast['total_pilot_cpu_hours'] > 40
            or forecast['inputs']['pilot_manifest'] != pilot_digest
            or forecast['measurement_commit'] != pilot['measurement_commit']
            or evidence['archive_rows'] != 84 or evidence['registered_rows_with_and_without_cap'] != 63):
        raise ValueError('final registration requires the completed pilot and all identity checks')
    with (costs / 'time-limits.csv').open() as stream:
        limits = list(csv.DictReader(stream))
    if (len(limits) != len(runs)
            or {(row['task'], row['arm']) for row in limits} != {(run['task'], run['label']) for run in runs}):
        raise ValueError('time limits differ from the registered arms')
    timing = []
    for row in limits:
        scout = row['task'] in SCOUTS
        full = row['full_pilot_reused'] == 'True'
        limit = {key: int(row[key]) if row[key] else None for key in
                 ('ordinary_limit_seconds', 'scout_limit_seconds', 'after_scout_initial_limit_seconds')}
        if (scout and (full or limit['ordinary_limit_seconds'] is not None
                       or not limit['scout_limit_seconds'] or not limit['after_scout_initial_limit_seconds'])
                or not scout and (not limit['ordinary_limit_seconds'] or limit['scout_limit_seconds']
                                  or limit['after_scout_initial_limit_seconds'])):
            raise ValueError('scout or ordinary time limit is missing')
        timing.append(dict(task=row['task'], arm=row['arm'], reuse_full_seed_42=full,
                           seed_spread=float(row['archive_seed_spread']), **limit))
    return dict(
        study=HERE.name, stage='grid_pending_approval',
        draft_sha256=checksum(HERE / 'plan.json'), pilot_cells_sha256=checksum(HERE / 'pilot-cells.json'),
        pilot_manifest_sha256=pilot_digest, identity_sha256=checksum(identity),
        measurement_commit=pilot['measurement_commit'], binary_sha256=pilot['binary_sha256'],
        definitions='The pinned draft defines the arms, protocol, observations, analysis and carry rule.',
        analysis_details={
            'floor': 'No compared arm has a lower 95% final-mean bound above the exact random-policy test floor at the same cap.',
            'too_easy': 'Use the original ablation ACS2 labels for both caps.',
            'ceiling': 'Each of the two arms in the contrast has a final mean of at least 0.99 at the cap under analysis.',
            'descriptive_steps': 'Weight by the registered pair distribution. Infer failed steps at the registered cap from original success and conditional solved steps. Divide the resulting weighted mean by the weighted mean shortest path of the same pairs.',
            'cpu_ratios': 'Divide per-seed process CPU and per-seed utility CPU by the paired S+1 process CPU before taking medians per maze. Report peak memory from the measured process.',
        },
        pilot=dict(seed=42, cells=168, cpu_hours=forecast['total_pilot_cpu_hours'],
                   full_budget_cells=forecast['full_pilot_cells'], wave_count=forecast['pilot_waves']),
        costs=dict(
            forecast_sha256=checksum(costs / 'forecast.json'),
            time_limits_sha256=checksum(costs / 'time-limits.csv'),
            archive_sha256=forecast['inputs']['archive'], new_cells=forecast['forecast_cells'],
            expected_total_grid_cpu_hours=forecast['forecast_cpu_hours'],
            expected_remaining_cpu_hours=forecast['forecast_remaining_cpu_hours'],
            sum_of_initial_allocations_cpu_hours=forecast['initial_reserved_grid_cpu_hours'],
            checkpoint_scenario_cpu_hours=forecast['forecast_cpu_hours_range'],
            model=forecast['cost_model'], limits=forecast['limit_model'],
        ),
        scout_gate=dict(tasks=list(SCOUTS), seed=42, concurrency=40, limits=timing,
                        dependency='Submit each dependent array behind the full-budget seed-42 scout gate. '
                        'Validate accounting and reproduce the pilot prefix before updating the dependent time limits. '
                        'Stop if any resulting limit exceeds the initial envelope or a scout fails.'),
        authorization='No grid submission until explicit approval states a CPU-hour limit covering the approved envelope and existing spend.'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--costs', required=True, type=Path)
    parser.add_argument('--identity', required=True, type=Path)
    parser.add_argument('--pilot', required=True, type=Path)
    args = parser.parse_args()
    path = HERE / 'final-plan.json'
    with path.open('x') as stream:
        stream.write(json.dumps(final_plan(args.costs, args.identity, args.pilot), indent=2) + '\n')
