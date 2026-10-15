import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from analysis import exact_success, head_spread, paired, seed_scores, summarize, threshold_crossing
from execution import APPROVED_SHA256, approved
from references import reused
from verify import verify_study

VIEWS = ['checkpoint_mean_within_seed', 'final_point']
SEEDS = list(range(42, 62))
REFERENCES = ['acs2her', 'acs2er_episodic', 'acs2', 'acs2er']


def numeric_fields(value, prefix=''):
    result = {}
    if value is None:
        return result
    for name, item in value.items():
        key = prefix + name
        if type(item) in (int, float):
            if not math.isfinite(item):
                raise ValueError('nonfinite diagnostic: ' + key)
            result[key] = item
        elif isinstance(item, dict):
            result.update(numeric_fields(item, key + '.'))
    return result


def metrics(row):
    names = ['actual_steps', 'episodes', 'online_updates', 'replay_updates', 'knowledge',
             'match_formations_train', 'classifier_perception_tests_train', 'population_classifiers',
             'population_numerosity', 'population_logical_bytes', 'population_known_bytes_lower_bound',
             'population_mark_entries', 'replay_logical_bytes', 'trajectory_logical_bytes',
             'wall_seconds_train', 'wall_seconds_eval', 'wall_seconds_total']
    result = {name: row[name] for name in names if name in row}
    for name in ['learning_update_counts', 'replay_diagnostics', 'ensemble']:
        result.update(numeric_fields(row.get(name, {}), name + '.'))
    ensemble = row.get('ensemble') or {}
    for head in ensemble.get('heads', []):
        result.update(numeric_fields(head, f"head.{head['head']}."))
    heads = ensemble.get('heads')
    if heads:
        for name in ['knowledge', 'population_classifiers', 'population_numerosity']:
            result['heads.' + name + '.population_standard_deviation'] = statistics.pstdev(head[name] for head in heads)
    for name, block in row['evaluation_blocks'].items():
        prefix = 'evaluation.' + name + '.'
        result[prefix + 'success'] = float(exact_success(block))
        if 'heads' in block:
            result[prefix + 'head_success_population_standard_deviation'] = head_spread(block)
            for index, head in enumerate(block['heads']):
                result[prefix + f'head.{index}.success'] = head['success']
        result.update(numeric_fields(block.get('head_disagreement', {}), prefix + 'head_disagreement.'))
    return result


def metric_summaries(rows, targets):
    seed_scores(rows, targets)
    by_seed = {seed: {} for seed in SEEDS}
    for row in rows:
        by_seed[row['seed']][row['nominal_step']] = metrics(row)
    expected = set(next(iter(by_seed[42].values())))
    if any(set(value) != expected for points in by_seed.values() for value in points.values()):
        raise ValueError('diagnostic fields differ within an arm; missing values cannot be averaged away')
    result = {view: {} for view in VIEWS}
    for name in sorted(expected):
        result[VIEWS[0]][name] = summarize([statistics.mean(by_seed[seed][point][name] for point in targets) for seed in SEEDS])
        result[VIEWS[1]][name] = summarize([by_seed[seed][targets[-1]][name] for seed in SEEDS])
    return result


def distribution_summaries(row):
    priority = (row.get('replay_diagnostics') or {}).get('priority') or {}
    values = {'training.' + name: priority.get(name) for name in ['proposal_variance', 'accepted_variance']}
    values.update({'evaluation.' + name: block.get('head_disagreement') for name, block in row['evaluation_blocks'].items()})
    return {name: value for name, value in values.items() if value is not None}


def pooled_bins(rows):
    result = {}
    for row in rows:
        for name, value in distribution_summaries(row).items():
            logarithmic = value['logarithmic']
            bounds = logarithmic['positive_bin_upper_bounds']
            specifications = [('logarithmic', 'zero', 0, 0, logarithmic['exact_zero_count'])]
            specifications.extend(('logarithmic', str(index), 0 if index == 0 else bounds[index-1], upper, count)
                                  for index, (upper, count) in enumerate(zip(bounds, logarithmic['positive_bin_counts'])))
            specifications.extend(('equal_width', str(index), 250000 * index / 16, 250000 * (index+1) / 16, count)
                                  for index, count in enumerate(value['histogram']))
            for kind, index, lower, upper, count in specifications:
                key = row['nominal_step'], name, kind, index, lower, upper
                cell = result.setdefault(key, dict(count=0, total=0))
                cell['count'] += count
                cell['total'] += value['count']
    return [dict(nominal_step=key[0], scope=key[1], bins=key[2], bin=key[3], lower_bound=key[4], upper_bound=key[5],
                 count=value['count'], total=value['total'], share=value['count']/value['total'])
            for key, value in result.items()]


def analyze_maze(datasets, config, plan):
    targets, identifier = config['targets'], config['id']
    expected = set(plan['agents'] + REFERENCES)
    if set(datasets) != expected:
        raise ValueError('analysis arm scope differs from the approved registration')
    scores = {name: seed_scores(rows, targets) for name, rows in datasets.items()}
    arms = {}
    for name, values in scores.items():
        arms[name] = dict(views={view: summarize([values[seed][view] for seed in SEEDS]) for view in VIEWS},
                          thresholds={key: threshold_crossing(datasets[name], targets, exact)
                                      for key, exact in plan['analysis']['thresholds'][identifier].items()})
    for threshold, checks in plan['analysis']['baseline_threshold_crossing_checks'][identifier].items():
        for name, crossing in checks.items():
            if arms[name]['thresholds'][threshold] != crossing:
                raise ValueError('archived baseline threshold crossing differs from the registration')
    comparisons = [dict(left=left, right=right, views=paired(scores[left], scores[right]))
                   for left, right in plan['analysis']['comparisons']]
    return dict(configuration=identifier, arms=arms, comparisons=comparisons), scores


def run_cost(entry, rows):
    path = entry['path']
    completion = json.loads((path / 'completion.json').read_text())
    resources = [json.loads(line) for line in (path / 'resources.jsonl').read_text().splitlines()]
    train_cpu = sum(value['train_cpu_seconds_estimate'] for value in resources)
    return dict(allocation_cpu_seconds=entry['allocation_cpu_seconds'],
                process_cpu_seconds=completion['user_cpu_seconds'] + completion['system_cpu_seconds'],
                process_wall_seconds=completion['elapsed_seconds'], peak_rss_kib=completion['peak_rss_kib'],
                training_cpu_seconds_estimate=train_cpu,
                training_steps_per_cpu_second_estimate=rows[-1]['actual_steps']/train_cpu,
                wall_seconds_train=rows[-1]['wall_seconds_train'], wall_seconds_eval=rows[-1]['wall_seconds_eval'],
                wall_seconds_total=rows[-1]['wall_seconds_total'])


def write_csv(path, values):
    if not values:
        return
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def analyze(study, her, baseline, destination):
    study, destination = Path(study), Path(destination)
    entries, verification = verify_study(study)
    if verification['pilot_identity_rows'] != 180:
        raise ValueError('all 180 pilot-overlapping rows must pass before analysis')
    plan = approved(study / 'approved-plan.json')
    entries.update(reused(plan, her, baseline))
    destination.mkdir(parents=True, exist_ok=False)
    tables = {name: [] for name in ['primary', 'comparisons', 'seed-scores', 'thresholds', 'learning-curves',
                                   'diagnostics', 'distributions', 'run-costs', 'point-diagnostics']}
    results = []
    for config in plan['configurations']:
        identifier, datasets = config['id'], {}
        for name in plan['agents'] + REFERENCES:
            datasets[name] = []
            for seed in SEEDS:
                entry = entries[identifier, name, seed]
                rows = [json.loads(line) for line in (entry['path'] / 'rows.jsonl').read_text().splitlines()]
                datasets[name].extend(rows)
                tables['run-costs'].append(dict(configuration=identifier, arm=name, seed=seed,
                    allocation_scope='new_selected_run' if name in plan['agents'] else 'reused_reference',
                    source_directory=str(entry['path']), **run_cost(entry, rows)))
        result, scores = analyze_maze(datasets, config, plan)
        results.append(result)
        for name, arm in result['arms'].items():
            for view, summary in arm['views'].items():
                tables['primary'].append(dict(configuration=identifier, arm=name, view=view, **summary))
            for threshold, crossing in arm['thresholds'].items():
                tables['thresholds'].append(dict(configuration=identifier, arm=name, threshold=threshold,
                    exact_threshold=plan['analysis']['thresholds'][identifier][threshold], first_registered_nominal_step=crossing))
            for seed, views in scores[name].items():
                for view, exact in views.items():
                    tables['seed-scores'].append(dict(configuration=identifier, arm=name, seed=seed, view=view,
                                                      exact_score=str(exact), score=float(exact)))
            summaries = metric_summaries(datasets[name], config['targets'])
            for view, values in summaries.items():
                for metric, summary in values.items():
                    tables['diagnostics'].append(dict(configuration=identifier, arm=name, view=view, metric=metric, **summary))
            for point in config['targets']:
                point_rows = [row for row in datasets[name] if row['nominal_step'] == point]
                for block in ['training', 'validation', 'test']:
                    tables['learning-curves'].append(dict(configuration=identifier, arm=name, nominal_step=point,
                        block=block, **summarize([exact_success(row['evaluation_blocks'][block]) for row in point_rows])))
                values = [metrics(row) for row in point_rows]
                for metric in sorted(values[0]):
                    tables['point-diagnostics'].append(dict(configuration=identifier, arm=name, nominal_step=point,
                        metric=metric, **summarize([value[metric] for value in values])))
            for value in pooled_bins(datasets[name]):
                tables['distributions'].append(dict(configuration=identifier, arm=name, **value))
        for comparison in result['comparisons']:
            for view, summary in comparison['views'].items():
                tables['comparisons'].append(dict(configuration=identifier, left=comparison['left'], right=comparison['right'],
                                                   view=view, **summary))
        del datasets
    for name, values in tables.items():
        write_csv(destination / (name + '.csv'), values)
    output = dict(approved_plan_sha256=APPROVED_SHA256, verification=verification, configurations=results,
                  statistical_unit='twenty paired seeds; heads averaged inside each seed',
                  primary='test success, equal checkpoint weights within seed',
                  intervals='marginal t(19) 95%; no multiplicity correction',
                  thresholds='exact integer per-head counts and rational arithmetic; first registered point, no interpolation',
                  diagnostics='mean and SE of per-seed diagnostics; quantiles are per-seed reference-set quantiles, then averaged',
                  histogram_weighting='pooled raw counts at each point; proposals weight training bins, fixed transitions weight evaluation bins',
                  time='process wait4 CPU, allocation CPUTimeRAW and wall time reported separately; sampled training CPU is an estimate',
                  allocation_scope='run-costs separates selected new runs from historical reused references; verification charges all new attempts and the pilot',
                  memory='logical payload and known inline headers are lower bounds; process peak RSS reported separately',
                  horizon='registered 2M/4M interactions; no convergence claim',
                  source_archives=dict(study=str(study), her=str(Path(her)), baseline=str(Path(baseline))))
    (destination / 'results.json').write_text(json.dumps(output, indent=2) + '\n')
    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(destination.iterdir())}
    (destination / 'files.sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--her', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = analyze(args.study, args.her, args.baseline, args.out)
    print(json.dumps(dict(configurations=len(value['configurations']), verification=value['verification'])))
