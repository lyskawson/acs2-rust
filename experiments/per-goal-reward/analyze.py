import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from protocol import ARMS, NEW_ARMS, REFERENCES, E, S, BLOCKS, configurations
from analysis import attach, pool_blocks, truth_index
from decision_metrics import decision_metrics
from results import contrasts, decisions
from grid_protocol import atomic_json
from agent_names import canonical_row
from collect import outside_repository
from export_metadata import annotate_outputs


def read_rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield canonical_row(json.loads(line))


def calculate(configs, inputs, truth):
    success, mechanisms, curves = {}, {}, []
    for config in configs:
        for arm in ARMS:
            for seed in range(42, 62):
                key = config['id'], arm, seed
                path = inputs['new' if arm in NEW_ARMS else 'success'][key]
                points, scores = [], []
                for row in read_rows(path):
                    if row['agent'] != arm or row['seed'] != seed or row['task'] != config['task']:
                        raise ValueError('analysis row identity differs from its indexed arm, seed or maze')
                    points.append(row['nominal_step'])
                    scores.append(row['evaluation_blocks']['test']['success'])
                if points != config['targets']:
                    raise ValueError('success analysis requires every registered checkpoint in order')
                success[config['task'], arm, seed] = dict(primary=statistics.mean(scores), final=scores[-1])
                if seed >= 47 or arm not in (*NEW_ARMS, E, S):
                    continue
                pair_path = path if arm in NEW_ARMS else inputs['pairs'][key]
                pair_points = []
                for row in read_rows(pair_path):
                    if row['agent'] != arm or row['seed'] != seed or row['task'] != config['task']:
                        raise ValueError('pair analysis row identity differs')
                    point = row['nominal_step']
                    pair_points.append(point)
                    blocks = {name: attach(row['evaluation_blocks'][name]['starts'], truth[config['task']]) for name in BLOCKS}
                    measured = decision_metrics(pool_blocks(blocks))
                    curves.append(dict(maze=config['task'], arm=arm, seed=seed, nominal_step=point,
                        view='pooled', F=measured['F'], F_mod=measured['F_mod'], post_hoc=True))
                    if point == config['targets'][-1]:
                        mechanisms[config['task'], arm, seed] = measured
                if pair_points != config['targets']:
                    raise ValueError('mechanism analysis requires all registered checkpoints in order')
    registered = contrasts(success, mechanisms, configs)
    return dict(registered=registered, decisions=decisions(registered), curves=curves,
        success=[dict(maze=maze, arm=arm, seed=seed, **values) for (maze, arm, seed), values in success.items()],
        mechanism=[dict(maze=maze, arm=arm, seed=seed, **values) for (maze, arm, seed), values in mechanisms.items()])


def checked_path(root, entry):
    path = Path(root) / entry['file']
    if not path.resolve().is_relative_to(Path(root).resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        raise ValueError('analysis input escaped its root or changed checksum')
    return path


def analyze(root, verification, output):
    root, output = Path(root), outside_repository(output)
    plan_path = Path(__file__).with_name('plan.json')
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    plan = json.loads(plan_path.read_text())
    for name in ('reference-index', 'model-index'):
        if hashlib.sha256((root / (name + '.json')).read_bytes()).hexdigest() != plan['cost_pilot'][name + '_sha256']:
            raise ValueError('registered reference or model index changed')
    approval = json.loads((root / 'approval.json').read_text())
    verified = json.loads(Path(verification).read_text())
    if ((root / 'approved-plan.json').read_bytes() != plan_path.read_bytes()
            or approval.get('approved') is not True or approval.get('plan_sha256') != digest
            or type(approval.get('phase_cpu_hour_limit')) not in (int, float) or approval['phase_cpu_hour_limit'] <= 0
            or verified.get('verified') is not True or verified.get('plan_sha256') != digest):
        raise ValueError('analysis requires the approved registration, CPU-hour limit and complete verified grid')
    inputs = dict(new={}, success={}, pairs={})
    for group, entries, base in [('new', verified['runs_index'], root),
            ('references', json.loads((root / 'reference-index.json').read_text()), root / 'references')]:
        for entry in entries:
            kind = 'new' if group == 'new' else entry['kind']
            key = entry['configuration_id'], entry['agent'], entry['seed']
            if kind not in inputs or key in inputs[kind]:
                raise ValueError('duplicate or unknown analysis input')
            inputs[kind][key] = checked_path(base, entry)
    configs = configurations()
    for kind, arms, seeds in [('new', NEW_ARMS, range(42, 62)), ('success', REFERENCES, range(42, 62)),
                             ('pairs', REFERENCES, range(42, 47))]:
        required = {(config['id'], arm, seed) for config in configs for arm in arms for seed in seeds}
        if set(inputs[kind]) != required:
            raise ValueError('missing or extra registered analysis input')
    truth = {entry['task']: truth_index(json.loads(checked_path(root, entry).read_text()))
             for entry in json.loads((root / 'model-index.json').read_text())}
    if set(truth) != {config['task'] for config in configs}:
        raise ValueError('analysis requires all six environment models')
    result = calculate(configs, inputs, truth)
    output.mkdir(parents=True, exist_ok=False)
    for name, rows in [('registered-comparisons', result['registered']), ('post-hoc-curves', result['curves']),
                       ('success-seed-metrics', result['success']), ('mechanism-seed-metrics', result['mechanism'])]:
        atomic_json(output / (name + '.json'), dict(plan_sha256=digest, rows=rows))
        with (output / (name + '.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(output / 'registered-decisions.json', dict(plan_sha256=digest, **result['decisions']))
    annotate_outputs(output)
    return dict(registered_comparisons=len(result['registered']), **result['decisions'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--verification', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.verification, args.out)))
