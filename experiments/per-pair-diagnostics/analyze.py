import argparse
import csv
import hashlib
import json
from pathlib import Path

from analysis import attach, consistency, pool_blocks, statistics_for, weighted_mean
from archive import models
from comparisons import registered_results
from post_hoc import curves, final_contrasts, modal_statistics, uniform_baselines, uniform_pairs, winner_attributes
from protocol import ARMS, configurations
from prepare_execution import APPROVED_SHA256, approved
from collect import outside_repository
from grid_protocol import atomic_json

CURVE_METRICS = ['F', 'C', 'random_reference', 'M', 'D', 'D_both_optimal', 'estimate_minus_q', 'winner_goal_mass',
    'F_mod', 'F_minus_F_mod', 'nonmodal_mass', 'nonmodal_share_eligible', 'nonmodal_optimal', 'nonmodal_modal_optimal',
    'winner_goal_attributes_0', 'winner_goal_attributes_1', 'winner_goal_attributes_2'] + [
    group + '_' + metric for group in ('solved', 'failed') for metric in
    ('estimate_minus_q', 'estimate_minus_v', 'winner_goal_mass', 'winner_goal_attributes_0', 'winner_goal_attributes_1', 'winner_goal_attributes_2')]


def write_csv(path, rows, columns=None):
    rows = list(rows)
    columns = columns or list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def flatten(measured, modal, attributes):
    result = {key: measured[key] for key in ('F', 'C', 'random_reference', 'M', 'D', 'D_both_optimal',
        'pair_count', 'eligible_count', 'eligible_weight', 'random_fallback_share', 'C_eligible', 'D_weight')}
    for group in ('all', 'solved', 'failed'):
        values = measured['values'][group]
        for key in ('estimate_minus_q', 'estimate_minus_v', 'winner_goal_mass', 'winner_goal_share', 'weight', 'winner_weight'):
            result[key if group == 'all' else group + '_' + key] = values[key]
        for count in range(3):
            prefix = '' if group == 'all' else group + '_'
            result[prefix + 'winner_goal_attributes_' + str(count)] = attributes[group]['mass'][count]
    result.update({key: value for key, value in modal.items() if key != 'post_hoc'})
    return result


def process_view(records):
    measured = statistics_for(records)
    modal = modal_statistics(records)
    attributes = winner_attributes(records)
    return measured, dict(post_hoc=True, modal=modal, winner_attributes=attributes), flatten(measured, modal, attributes)


def distance_values(records):
    result = []
    for distance in sorted({record['shortest_distance'] for record in records}):
        selected = [record for record in records if record['shortest_distance'] == distance]
        for name, group in [('all', selected), ('solved', [record for record in selected if record['success']]),
                            ('failed', [record for record in selected if not record['success']])]:
            result.append(dict(post_hoc=True, shortest_distance=distance, group=name, pair_count=len(group),
                weight=sum(record['weight'] for record in group),
                estimate_minus_q=weighted_mean(group, lambda record: record['estimated_first_action_value'] - record['q_star']),
                estimate_minus_v=weighted_mean(group, lambda record: record['estimated_first_action_value'] - record['v_star'])))
    return result


def process_row(row, truth):
    views = {name: attach(block['starts'], truth) for name, block in row['evaluation_blocks'].items()}
    views['pooled'] = pool_blocks(views)
    return views


def read_verified(root, verification):
    root = Path(root)
    checked = json.loads(Path(verification).read_text())
    if checked.get('verified') is not True or checked.get('plan_sha256') != APPROVED_SHA256:
        raise ValueError('analysis requires independent verified raw artifacts')
    expected = {(config['id'], arm, seed) for config in configurations() for arm in ARMS for seed in range(42, 47)}
    entries = {}
    for entry in checked['runs_index']:
        key = entry['configuration_id'], entry['arm'], entry['seed']
        path = root / entry['file']
        if key in entries or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError('duplicate or escaped verified input')
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('verified raw row file changed')
        entries[key] = dict(entry, path=path)
    if set(entries) != expected:
        raise ValueError('analysis requires all 150 five-seed runs')
    return entries


def analyze(root, output, verification):
    root, output = Path(root), outside_repository(output)
    plan, approval = approved()
    if (root / 'approved-plan.json').read_bytes() != Path(__file__).with_name('plan.json').read_bytes() or json.loads((root / 'approval.json').read_text()) != approval:
        raise ValueError('analysis needs the exact approved registration')
    entries = read_verified(root, verification)
    truth = models(root)
    output.mkdir(parents=True, exist_ok=False)
    registered_metrics, flat_metrics, final_records = {}, {}, {}
    seed_rows, distances = [], []
    with (output / 'post-hoc-diagnostics.jsonl').open('w') as diagnostics:
        for config in configurations():
            for arm in ARMS:
                for seed in range(42, 47):
                    entry = entries[config['id'], arm, seed]
                    with entry['path'].open() as stream:
                        for line in stream:
                            row = json.loads(line)
                            point = row['nominal_step']
                            views = process_row(row, truth[config['task']])
                            final = point == config['targets'][-1]
                            if final:
                                views['uniform_pairs'] = uniform_pairs(views['pooled'])
                            for view, records in views.items():
                                key = config['task'], arm, seed, point, view
                                measured, extra, flat = process_view(records)
                                metadata = dict(maze=config['task'], arm=arm, seed=seed, nominal_step=point, view=view, post_hoc=True)
                                if view == 'pooled':
                                    registered_metrics[key] = measured
                                flat_metrics[key] = flat
                                seed_rows.append(dict(metadata, **flat))
                                diagnostics.write(json.dumps(dict(metadata, metrics=measured, **{key: value for key, value in extra.items() if key != 'post_hoc'})) + '\n')
                                distances.extend(dict(metadata, **{key: value for key, value in item.items() if key != 'post_hoc'}) for item in distance_values(records))
                            if final:
                                final_records.setdefault((config['task'], arm), {})[seed] = [
                                    {key: record[key] for key in ('start', 'goal', 'weight', 'success')} for record in views['pooled']]
    results = registered_results(registered_metrics, configurations())
    atomic_json(output / 'registered-comparisons.json', dict(plan_sha256=APPROVED_SHA256, comparisons=results))
    write_csv(output / 'registered-comparisons.csv', results,
        ['maze', 'comparison', 'metric', 'nominal_step', 'view', 'registered', 'n', 'mean', 'se', 'ci_low', 'ci_high'])
    write_csv(output / 'post-hoc-seed-metrics.csv', seed_rows)
    write_csv(output / 'post-hoc-distance-values.csv', distances)
    curve_rows = curves(flat_metrics, CURVE_METRICS)
    write_csv(output / 'post-hoc-curves.csv', curve_rows)
    uniform = final_contrasts(flat_metrics, configurations(), 'uniform_pairs',
        ['F', 'C', 'M', 'D', 'D_both_optimal', 'estimate_minus_q', 'winner_goal_mass',
         'winner_goal_attributes_0', 'winner_goal_attributes_1', 'winner_goal_attributes_2'])
    uniform.extend(uniform_baselines(flat_metrics, configurations()))
    modal = final_contrasts(flat_metrics, configurations(), 'pooled',
        ['F_mod', 'F_minus_F_mod', 'nonmodal_mass', 'nonmodal_share_eligible', 'nonmodal_optimal', 'nonmodal_modal_optimal'])
    write_csv(output / 'post-hoc-uniform-comparisons.csv', uniform)
    write_csv(output / 'post-hoc-modal-comparisons.csv', modal)
    final_points = {config['task']: config['targets'][-1] for config in configurations()}
    write_csv(output / 'post-hoc-final-profile.csv', [item for item in curve_rows if item['nominal_step'] == final_points[item['maze']]
        and ('winner_goal' in item['metric'] or item['metric'] in ('F_mod', 'F_minus_F_mod', 'nonmodal_mass', 'nonmodal_optimal', 'nonmodal_modal_optimal'))])
    atomic_json(output / 'post-hoc-consistency.json', [dict(maze=maze, arm=arm, **consistency(records))
                for (maze, arm), records in sorted(final_records.items())])
    index = {str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}
    summary = dict(plan_sha256=APPROVED_SHA256, registered_comparisons=len(results), seed_cells=len(seed_rows),
        verification_sha256=hashlib.sha256(Path(verification).read_bytes()).hexdigest(),
        uniform_comparisons=len(uniform), modal_comparisons=len(modal), files=index)
    atomic_json(output / 'analysis-summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--verification', required=True)
    args = parser.parse_args()
    result = analyze(args.root, args.out, args.verification)
    print(json.dumps({key: value for key, value in result.items() if key != 'files'}))
