import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from protocol import HERE, ARMS, SEEDS, CONTRASTS
from grid_execution import APPROVED_SHA256, registered_plan
from agent_names import canonical_run_name
from analyze import calculate, interval, difference, check_reference_intervals


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read_source(entry):
    path = Path(entry['source'])
    if checksum(path) != entry['sha256']:
        raise ValueError('analysis source checksum differs')
    expected = entry['task'], entry['agent'], entry['seed']
    rows, secondary = [], []
    with path.open() as stream:
        for line in stream:
            raw = json.loads(line)
            task, agent, seed = raw['task'], canonical_run_name(raw['agent']), raw['seed']
            if (task, agent, seed) != expected:
                raise ValueError('analysis source identity differs')
            step = raw['nominal_step']
            success = raw['evaluation_blocks']['test']['success']
            rows.append(dict(task=task, agent=agent, seed=seed, nominal_step=step,
                             evaluation_blocks=dict(test=dict(success=success))))
            rewards = (raw.get('replay_diagnostics') or {}).get('per_goal_rewards') or {}
            secondary.append(dict(
                task=task, arm=agent, seed=seed, nominal_step=step,
                knowledge=raw.get('knowledge'), population_classifiers=raw.get('population_classifiers'),
                population_numerosity=raw.get('population_numerosity'),
                population_logical_bytes=raw.get('population_logical_bytes'),
                per_goal_existing_pairs=rewards.get('existing_pairs', 0),
                per_goal_updated_entries=rewards.get('updated_entries', 0),
                wall_seconds_train=raw.get('wall_seconds_train'), wall_seconds_eval=raw.get('wall_seconds_eval'),
                host=raw.get('host'), cpu_model=raw.get('cpu_model'),
            ))
    return rows, secondary


def load_sources(entries, configs):
    expected = {(config['task'], arm, seed) for config in configs for arm in ARMS for seed in SEEDS}
    inputs, secondary = {}, []
    for entry in entries:
        key = entry['task'], entry['agent'], entry['seed']
        if key not in expected or key in inputs:
            raise ValueError('unexpected or duplicate analysis run')
        inputs[key], observations = read_source(entry)
        secondary.extend(observations)
    if set(inputs) != expected:
        raise ValueError('registered analysis requires every full run')
    return inputs, secondary


def tables(configs, inputs, names):
    result = calculate(configs, inputs, names)
    seeds, curves, arms, paired = [], [], [], []
    for config in configs:
        task = config['task']
        common = dict(task=task, maze=names[task], group=config['group'])
        scores = {}
        for arm in ARMS:
            series = {}
            for seed in SEEDS:
                values = [row['evaluation_blocks']['test']['success'] for row in inputs[task, arm, seed]]
                series[seed] = values
                score = dict(primary=statistics.mean(values), final=values[-1])
                scores[arm, seed] = score
                seeds.append(dict(**common, arm=arm, seed=seed, **score))
            for measure in ['primary','final']:
                values = [scores[arm, seed][measure] for seed in SEEDS]
                summary = interval(values) if measure == 'final' else dict(n=20, mean=statistics.mean(values))
                arms.append(dict(**common, arm=arm, measure=measure, **summary))
            for index, target in enumerate(config['targets']):
                mean = statistics.mean(series[seed][index] for seed in SEEDS)
                curves.append(dict(**common, arm=arm, nominal_step=target, n=20, mean=mean))
        for contrast in CONTRASTS:
            for measure in ['primary', 'final']:
                for seed in SEEDS:
                    values = {arm: scores[arm, seed][measure] for arm in ARMS}
                    paired.append(dict(**common, contrast=contrast, measure=measure, seed=seed,
                                       difference=difference(contrast, values)))
    return dict(result, seed_scores=seeds, arm_success=arms, curves=curves, paired_differences=paired)


def write_csv(rows, path):
    if not rows:
        raise ValueError('empty analysis table')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(index, earlier, out, configs=None, names=None):
    configs = configs or registered_plan()['configurations']
    names = names or json.loads((HERE / 'maze-names.json').read_text())
    entries = json.loads(Path(index).read_text())
    inputs, secondary = load_sources(entries, configs)
    result = tables(configs, inputs, names)
    previous = json.loads(Path(earlier).read_text())
    if isinstance(previous, dict):
        previous = previous['rows']
    checked = check_reference_intervals(result['comparisons'], previous)
    groups = {config['task']: config['group'] for config in configs}
    for row in secondary:
        row.update(maze=names[row['task']], group=groups[row['task']])
    output = Path(out)
    output.mkdir(parents=True, exist_ok=False)
    for key in ['comparisons', 'labels', 'grouped', 'seed_scores', 'arm_success', 'curves', 'paired_differences']:
        write_csv(result[key], output / (key.replace('_', '-') + '.csv'))
    write_csv(secondary, output / 'secondary-checkpoints.csv')
    sustained = [{key: value for key, value in row.items() if key not in ['targets', 'means']}
                 for row in result['sustained']]
    write_csv(sustained, output / 'sustained-095.csv')
    (output / 'registered-results.json').write_text(json.dumps(result, indent=2) + '\n')
    evidence = dict(registered=True, full_runs=len(entries), earlier_intervals_bit_identical=checked,
        plan_sha256=APPROVED_SHA256, source_index_sha256=checksum(index), earlier_comparisons_sha256=checksum(earlier),
        note='The reader normalizes archived agent names and projects each checkpoint to scalar values.')
    (output / 'analysis-verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', required=True)
    parser.add_argument('--earlier', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.index, args.earlier, args.out)))
