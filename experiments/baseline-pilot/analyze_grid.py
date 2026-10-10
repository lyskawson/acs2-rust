import argparse
import hashlib
import json
from pathlib import Path
import statistics

from collect import complete_rows, outside_repository, validate_row
from grid_attempts import attempts, select_attempts
from grid_protocol import PLAN_SHA256, atomic_json, compare_pilot, load_plan, summarize, verdict

REFERENCE_FIELDS = ('preset', 'agent_parameters', 'random_floor', 'reachable_within_cap',
                    'reachable_after_cap', 'unreachable_or_ambiguous', 'reference_distribution',
                    'evaluation_distribution', 'evaluation_pairs', 'evaluation_sample_seed',
                    'evaluation_sample_stream', 'classifier_size_bytes', 'replay_sample_size_bytes')
COST_FIELDS = ('actual_steps', 'episodes', 'online_updates', 'replay_updates', 'match_formations_train',
               'classifier_perception_tests_train', 'population_classifiers', 'population_numerosity',
               'population_logical_bytes', 'population_known_bytes_lower_bound', 'population_mark_entries',
               'replay_samples', 'replay_logical_bytes', 'trajectory_logical_bytes')


def analyze_values(series, config, seeds, floor, ceiling):
    expected = {(agent, seed) for agent in ('acs2', 'acs2er') for seed in seeds}
    if set(series) != expected or len(seeds) != 20:
        raise ValueError('missing or unexpected agent/seed series')
    if any(len(values) != len(config['targets']) for values in series.values()):
        raise ValueError('missing evaluation points')
    points = [{'agent': agent, 'nominal_step': step, 'random_floor': floor, 'ceiling': ceiling,
               **summarize([series[agent, seed][index] for seed in seeds])}
              for agent in ('acs2', 'acs2er') for index, step in enumerate(config['targets'])]
    scores = {key: statistics.mean(values) for key, values in series.items()}
    seed_scores = [{'seed': seed, 'acs2': scores['acs2', seed], 'acs2er': scores['acs2er', seed],
                    'paired_difference': scores['acs2er', seed] - scores['acs2', seed]} for seed in seeds]
    score_summary = {agent: summarize([scores[agent, seed] for seed in seeds]) for agent in ('acs2', 'acs2er')}
    paired = summarize([row['paired_difference'] for row in seed_scores])
    count = len(config['targets'])
    decision = verdict(points[count - 1], points[-1], points[0], floor, ceiling)
    return dict(points=points, seed_scores=seed_scores, score_summary=score_summary,
                paired_score_difference=paired, verdict=decision, floor=floor, ceiling=ceiling)


def verify_series(rows, config, agent, seed, plan, pilot_rows, host):
    if len(rows) != len(config['targets']):
        raise ValueError('unexpected row count')
    for row, point, pilot in zip(rows, config['targets'], pilot_rows):
        validate_row(row, config, agent, seed, plan['measurement_commit'], point)
        if row['cpu_model'] != plan['cpu_model'] or (host and row['host'].split('.')[0] != host):
            raise ValueError('hardware differs from assignment')
        for field in REFERENCE_FIELDS:
            if row.get(field) != pilot.get(field):
                raise ValueError(f'reference field differs from pilot: {field}')
    if seed == 42:
        return compare_pilot(rows, pilot_rows)
    return 0


def analyze(root, pilot, config_id):
    plan = load_plan(root / 'approved-plan.json')
    config = next(item for item in plan['configurations'] if item['id'] == config_id)
    entries = [item for item in json.loads((root / 'ledger.json').read_text()) if item['configuration_id'] == config_id]
    known_attempts = attempts(root, entries, config_id)
    registered = {item['path'] for item in known_attempts}
    discovered = set((root / 'batches').glob(f'*-{config_id}/runs/*/rows.jsonl'))
    discovered.update((root / 'attempts').glob(f'*/measurement/{config_id}_*/rows.jsonl'))
    if any(path.parent not in registered for path in discovered):
        raise ValueError('unregistered measurement directory')
    assignments, inventory = select_attempts(root, entries, config)
    expected = {(agent, seed) for agent in plan['agents'] for seed in plan['seeds']}
    if set(assignments) != expected:
        destination = outside_repository(root / 'analysis' / config_id)
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(destination / 'attempts.json', inventory)
        raise ValueError('not all approved 40 runs have a complete attempt; see attempts.json')
    complete_ids = {item['attempt_id'] for item in inventory if item['status'] == 'complete'}
    complete = [item for item in known_attempts if item['attempt_id'] in complete_ids]
    paths = [item['path'] for item in complete]
    by_path = {item['path']: item for item in complete}
    series, metrics, checksums, issues = {}, [], {}, []
    compared, matched = 0, 0
    floor = ceiling = None
    for path in sorted(paths):
        try:
            launch = json.loads((path / 'launch.json').read_text())
            manifest = launch['manifest']
            agent, seed = manifest['agent'], manifest['seed']
            assignment = by_path[path]
            if (agent, seed) != (assignment['agent'], assignment['seed']):
                raise ValueError('manifest differs from attempt')
            if path.name != f'{config_id}_{agent}_s{seed}' or manifest['configuration'] != config:
                raise ValueError('run identity differs from directory or plan')
            if launch['binary_sha256'] != plan['binary_sha256'] or launch['partition'] != plan['partition'] or manifest['plan_sha256'] != PLAN_SHA256:
                raise ValueError('binary, partition or plan identity differs')
            if str(launch['array_job_id']) != str(assignment['array_id']) or int(launch['array_task_id']) != assignment['task_id']:
                raise ValueError('allocation assignment differs')
            result = json.loads((path / 'completion.json').read_text())
            rows, incomplete = complete_rows(path / 'rows.jsonl')
            resources, resource_tail = complete_rows(path / 'resources.jsonl')
            if incomplete or resource_tail:
                raise ValueError('incomplete final line preserved; excluded from analysis')
            if result['exit_code'] != 0 or result['failure'] or result['incomplete_final_line'] or result['rows_seen'] != len(config['targets']):
                raise ValueError('run did not finish successfully')
            if len(resources) != len(rows) or [item['nominal_step'] for item in resources] != config['targets']:
                raise ValueError('resource records do not match evaluation points')
            if not 0 <= result['first_row_seconds'] <= plan['first_row_timeout_seconds']:
                raise ValueError('first row missed deadline')
            pilot_rows, pilot_tail = complete_rows(pilot / 'runs' / f'{config_id}_{agent}_s42' / 'rows.jsonl')
            if pilot_tail:
                raise ValueError('pilot reference has an incomplete tail')
            count = verify_series(rows, config, agent, seed, plan, pilot_rows, assignment.get('host'))
            if assignment['attempt_id'] != assignments[agent, seed]['attempt_id']:
                continue
            compared += count
            if seed == 42:
                matched += 1
            floor, ceiling = rows[0]['random_floor'], rows[0]['reachable_within_cap']
            series[agent, seed] = [row['success'] for row in rows]
            training = sum(row['train_cpu_seconds_estimate'] for row in resources)
            evaluation = sum(row['eval_cpu_seconds_estimate'] for row in resources)
            if training <= 0:
                raise ValueError('nonpositive estimated training CPU')
            metrics.append(dict(attempt_id=assignment['attempt_id'], agent=agent, seed=seed, host=rows[-1]['host'], cpu_model=rows[-1]['cpu_model'],
                                training_cpu_seconds_estimate=training, evaluation_cpu_seconds_estimate=evaluation,
                                steps_per_training_cpu_second_estimate=rows[-1]['actual_steps'] / training,
                                process_cpu_seconds=result['user_cpu_seconds'] + result['system_cpu_seconds'],
                                peak_rss_kib=result['peak_rss_kib'], first_row_seconds=result['first_row_seconds'],
                                bytes_per_row=(path / 'rows.jsonl').stat().st_size / len(rows),
                                final_counters={field: rows[-1][field] for field in COST_FIELDS},
                                points=[dict(nominal_step=row['nominal_step'], counters={field: row[field] for field in COST_FIELDS}, resources=resource)
                                        for row, resource in zip(rows, resources)]))
            for source in path.iterdir():
                if source.is_file():
                    checksums[str(source.relative_to(root))] = hashlib.sha256(source.read_bytes()).hexdigest()
        except (ValueError, KeyError, OSError, TypeError) as error:
            issues.append({'path': str(path.relative_to(root)), 'error': str(error)})
    destination = outside_repository(root / 'analysis' / config_id)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / 'verification.json', {'issues': issues, 'verified_runs': len(series), 'raw_files_preserved': True})
    if issues:
        raise ValueError(f'{len(issues)} invalid runs; see {destination / "verification.json"}')
    result = analyze_values(series, config, plan['seeds'], floor, ceiling)
    result.update(configuration=config, plan_sha256=PLAN_SHA256, runs=40,
                  pilot_identity={'matched_runs': matched, 'compared_fields': compared,
                                  'excluded_fields': ['host', 'cpu_model', 'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total']},
                  metrics=metrics, attempts=inventory, checksums=checksums,
                  uncertainty='Training/evaluation CPU is estimated from 20 ms schedstat sampling aligned to runner wall intervals. Process CPU and peak RSS are wait4 measurements. Between-seed t(19) intervals do not include sampling/alignment uncertainty or the BitFlipping evaluation sample uncertainty.')
    atomic_json(destination / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--pilot', required=True, type=Path)
    parser.add_argument('--configuration', required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.pilot, args.configuration)
    print(json.dumps({key: result[key] for key in ('configuration', 'runs', 'verdict', 'pilot_identity', 'paired_score_difference')}))


if __name__ == '__main__':
    main()
