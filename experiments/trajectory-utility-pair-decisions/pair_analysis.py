import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
from statistics import mean, median
import subprocess

from pair_measures import ANALYSIS, BINS, MEASURES, VERDICTS, attach, distance_bin, distance_sums, judge, measures, \
    paired, pooled, summarize
from pair_register import COUNTERS, families
from pair_sources import (ARMS, E1, GROUPS, HERE, LAMBDAS, MEAN, OUTPUT, PAIR_SEEDS, RANDOM, REPO, RUNS, S1, SEEDS,
                          check_identity, json_lines, pair_universe, registered_configs, sha256, truth_model, verified)

PLAN = HERE / 'plan.json'
PLAN_PATH = 'experiments/trajectory-utility-pair-decisions/plan.json'
TRUTH = {}


def git(*args):
    return subprocess.check_output(['/usr/bin/git', '-C', str(REPO), *args], text=True).strip()


def registration_commit():
    commits = git('log', '--format=%H', '--', PLAN_PATH).split()
    if len(commits) != 1 or git('status', '--porcelain', '--', PLAN_PATH):
        raise ValueError('the registered plan must be committed once and unchanged')
    return commits[0]


def arm_label(arm):
    names = {E1: 'E+1', S1: 'S+1'}
    for share, suffix in LAMBDAS:
        names[f'{MEAN}_{suffix}'] = f'M {share}'
        names[f'{RANDOM}_{suffix}'] = f'R {share}'
    return names[arm]


def lookup(row, dotted):
    value = row
    for part in dotted.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_truth(path, digest):
    if path not in TRUTH:
        TRUTH[path] = ANALYSIS.truth_index(json.loads(verified(Path(path), digest).read_bytes()))
    return TRUTH[path]


def summarize_run(item):
    path, digest, task, arm, seed, config, truth_path, truth_digest = item
    rows = json_lines(verified(Path(path), digest))
    check_identity(rows, task, arm, seed, config)
    counters = {row['nominal_step']: {name: lookup(row, name) for name in COUNTERS} for row in rows}
    pairs = None
    if seed in PAIR_SEEDS:
        truth = load_truth(truth_path, truth_digest)
        pairs = {}
        for row in rows:
            test = row['evaluation_blocks']['test']
            pair_universe(test, config)
            pairs[row['nominal_step']] = distance_sums(attach(test['starts'], truth))
    return task, arm, seed, pairs, counters


def read_runs(plan, configs, truth_dir, workers):
    if plan['inputs_fields'] != ['task', 'arm', 'seed', 'archive', 'path', 'sha256', 'kind']:
        raise ValueError('registered input fields differ')
    items = []
    for task, arm, seed, archive, relative, digest, _ in plan['inputs']:
        truth = plan['truth_models'][task]
        items.append((str(RUNS / archive / relative), digest, task, arm, seed, configs[task],
                      str(truth_dir / f'{task}.json'), truth['sha256']))
    items.sort(key=lambda item: (item[2], item[3], item[4]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(summarize_run, items, chunksize=4))
    pairs, counters = {}, {}
    for task, arm, seed, run_pairs, run_counters in results:
        counters[task, arm, seed] = run_counters
        if run_pairs is not None:
            pairs[task, arm, seed] = run_pairs
    if len(pairs) != len(configs) * len(ARMS) * len(PAIR_SEEDS) or len(counters) != len(plan['inputs']):
        raise ValueError('STOP: a registered run lacks per-pair records')
    return pairs, counters


def views(configs):
    points = sorted({point for config in configs.values() for point in config['targets']})
    for group in GROUPS:
        for point in points:
            tasks = [task for task, config in configs.items() if config['group'] == group and point in config['targets']]
            if tasks:
                yield group, point, tasks


def arm_cells(pairs, configs, distances_of):
    seeds = {}
    for group, point, tasks in views(configs):
        present = sorted({distance for task in tasks for distance in pairs[task, S1, 42][point]})
        for key, distances in distances_of(present):
            for arm in ARMS:
                values = {}
                for seed in PAIR_SEEDS:
                    sums = pooled({task: pairs[task, arm, seed][point] for task in tasks}, tasks, distances)
                    if sums is None:
                        break
                    values[seed] = (sums, measures(sums))
                if values:
                    mazes = [task for task in tasks if any(d in pairs[task, arm, 42][point] for d in distances)]
                    seeds[arm, group, key, point] = dict(mazes=len(mazes), pairs=values[42][0]['pairs'], seeds=values)
    return seeds


def by_bins(present):
    for name, low, high in BINS:
        distances = [d for d in present if distance_bin(d) == name]
        if distances:
            yield name, distances
    yield 'all', present


def by_distance(present):
    for distance in present:
        yield distance, [distance]


def arm_table(cells, kind):
    rows = []
    for (arm, group, key, point), cell in sorted(cells.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]), item[0][3])):
        row = dict(arm=arm, label=arm_label(arm), group=group, **{kind: key}, checkpoint=point, mazes=cell['mazes'],
                   pairs=cell['pairs'])
        for measure in MEASURES:
            values = [cell['seeds'][seed][1][measure] for seed in PAIR_SEEDS]
            stats = summarize(values)
            row[measure] = stats['mean']
            row[f'{measure}_n'] = stats['n']
            row[f'{measure}_ci_low'] = stats['ci_low']
            row[f'{measure}_ci_high'] = stats['ci_high']
            row[f'{measure}_seeds'] = values
        rows.append(row)
    return rows


def comparison_table(cells, kind):
    rows = []
    for contrast in families():
        for (arm, group, key, point), cell in cells.items():
            if arm != contrast['left']:
                continue
            other = cells[contrast['right'], group, key, point]
            if other['pairs'] != cell['pairs']:
                raise ValueError('compared arms cover different pairs')
            for measure in MEASURES:
                left = {seed: cell['seeds'][seed][1][measure] for seed in PAIR_SEEDS}
                right = {seed: other['seeds'][seed][1][measure] for seed in PAIR_SEEDS}
                result = paired(left, right)
                rows.append(dict(family=contrast['family'], contrast=contrast['tag'], share=contrast.get('share'),
                                 left=contrast['left'], right=contrast['right'], group=group, **{kind: key},
                                 checkpoint=point, measure=measure, mazes=cell['mazes'], pairs=cell['pairs'],
                                 n=result['n'], mean=result['mean'], se=result['se'], ci_low=result['ci_low'],
                                 ci_high=result['ci_high'], seed_differences=result['seed_differences']))
    rows.sort(key=lambda row: (row['family'], row['contrast'], row['group'], str(row[kind]), row['checkpoint'], row['measure']))
    return rows


def maze_table(pairs, configs):
    rows = []
    for task, config in configs.items():
        for point in config['targets']:
            present = sorted(pairs[task, S1, 42][point])
            for key, distances in by_bins(present):
                for arm in ARMS:
                    row = dict(task=task, group=config['group'], arm=arm, label=arm_label(arm), bin=key, checkpoint=point)
                    for seed in PAIR_SEEDS:
                        sums = pooled({task: pairs[task, arm, seed][point]}, [task], distances)
                        row['pairs'] = sums['pairs']
                        for measure, value in measures(sums).items():
                            row.setdefault(f'{measure}_seeds', []).append(value)
                    for measure in MEASURES:
                        row[measure] = summarize(row[f'{measure}_seeds'])['mean']
                    rows.append(row)
    return rows


def counter_tables(counters, configs):
    availability, per_maze = [], []
    for arm in ARMS:
        for name in COUNTERS:
            values = [run[point][name] for (task, other, seed), run in counters.items() if other == arm for point in run]
            recorded = sum(value is not None for value in values)
            status = 'recorded' if recorded == len(values) else 'not recorded' if recorded == 0 else 'partly recorded'
            availability.append(dict(arm=arm, label=arm_label(arm), counter=name, rows=len(values), recorded=recorded,
                                     status=status))
    status = {(row['arm'], row['counter']): row['status'] for row in availability}
    for task, config in configs.items():
        for arm in ARMS:
            for point in config['targets']:
                for name in COUNTERS:
                    if status[arm, name] == 'not recorded':
                        continue
                    values = [counters[task, arm, seed][point][name] for seed in SEEDS]
                    present = [value for value in values if value is not None]
                    per_maze.append(dict(task=task, group=config['group'], arm=arm, label=arm_label(arm), checkpoint=point,
                                         counter=name, seeds=len(present), mean=mean(present) if present else None,
                                         median=median(present) if present else None,
                                         minimum=min(present) if present else None, maximum=max(present) if present else None))
    grouped = defaultdict(list)
    for row in per_maze:
        if row['mean'] is not None:
            grouped[row['arm'], row['group'], row['checkpoint'], row['counter']].append(row['mean'])
    groups = [dict(arm=arm, label=arm_label(arm), group=group, checkpoint=point, counter=name, mazes=len(values),
                   median_of_maze_means=median(values), mean_of_maze_means=mean(values), minimum=min(values),
                   maximum=max(values))
              for (arm, group, point, name), values in sorted(grouped.items())]
    return availability, per_maze, groups


def verdicts(plan, comparisons):
    index = {(row['group'], row['contrast'], row['measure'], row['bin'], row['checkpoint']): row for row in comparisons}
    cells, summary = [], {}
    for prediction in plan['predictions']:
        counts = dict.fromkeys(VERDICTS, 0)
        for cell in prediction['cells']:
            row = index[cell['group'], cell['contrast'], cell['measure'], cell['bin'], cell['checkpoint']]
            verdict = judge(row, cell['predicted_sign'])
            counts[verdict] += 1
            cells.append(dict(prediction=prediction['id'], **cell, n=row['n'], mean=row['mean'], ci_low=row['ci_low'],
                              ci_high=row['ci_high'], verdict=verdict))
        summary[prediction['id']] = dict(cells=len(prediction['cells']), **counts)
    return cells, summary


def write_table(out, name, rows):
    (out / f'{name}.json').write_text(json.dumps(rows, indent=1) + '\n')
    if not rows:
        raise ValueError(f'empty table {name}')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (out / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def main(args):
    commit = registration_commit()
    plan = json.loads(PLAN.read_bytes())
    configs = registered_configs()
    if set(plan['mazes']) != set(configs) or any(plan['mazes'][task]['cap'] != configs[task]['cap'] for task in configs):
        raise ValueError('registered mazes differ')
    args.out.mkdir(parents=True, exist_ok=False)
    truth_dir = args.out / 'truth'
    truth_dir.mkdir()
    for task, config in configs.items():
        content, digest = truth_model(task, config['cap'])
        if digest != plan['truth_models'][task]['sha256']:
            raise ValueError(f'ground truth differs from the registration: {task}')
        (truth_dir / f'{task}.json').write_bytes(content)
    pairs, counters = read_runs(plan, configs, truth_dir, args.workers)
    bins = arm_cells(pairs, configs, by_bins)
    distances = arm_cells(pairs, configs, by_distance)
    comparisons = comparison_table(bins, 'bin')
    cells, summary = verdicts(plan, comparisons)
    availability, counter_mazes, counter_groups = counter_tables(counters, configs)
    tables = {'arm-measures-bins': arm_table(bins, 'bin'), 'arm-measures-distances': arm_table(distances, 'distance'),
              'comparisons-bins': comparisons, 'comparisons-distances': comparison_table(distances, 'distance'),
              'maze-measures': maze_table(pairs, configs), 'verdicts': cells, 'counter-availability': availability,
              'copy-counters-mazes': counter_mazes, 'copy-counters-groups': counter_groups}
    for name, rows in tables.items():
        write_table(args.out, name, rows)
    result = dict(post_hoc=True, label='Post hoc associations in stored rows; no causal claim.',
                  plan_sha256=sha256(PLAN), registration_commit=commit, analysis_commit=git('rev-parse', 'HEAD'),
                  analysis_tree_dirty=bool(git('status', '--porcelain')), runs=len(counters), pair_runs=len(pairs),
                  verdicts=summary, tables=sorted(tables))
    (args.out / 'summary.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUTPUT / 'analysis')
    parser.add_argument('--workers', type=int, default=4)
    main(parser.parse_args())
