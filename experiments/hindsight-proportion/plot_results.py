import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from protocol import ROOT
from collect import outside_repository
from grid_protocol import load_plan

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#555555']


def save(figure, output, name):
    figure.savefig(output / (name + '.png'), dpi=180)
    figure.savefig(output / (name + '.pdf'))
    plt.close(figure)


def plot(analysis, output):
    analysis, output = Path(analysis), outside_repository(output)
    result = json.loads((analysis / 'results.json').read_text())
    load_plan(ROOT / 'experiments/hindsight-proportion/plan.json', result['plan_sha256'])
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)
    for axis, maze in zip(axes.flat, result['mazes']):
        rows = [row for row in result['tuning_table'] if row['configuration_id'] == maze['configuration_id']]
        rows.sort(key=lambda row: row['proportion'])
        positions = list(range(len(rows)))
        axis.errorbar(positions, [row['mean'] for row in rows], yerr=[row['se'] for row in rows],
                      color=COLORS[0], marker='o', capsize=3)
        chosen = next(index for index, row in enumerate(rows) if row['selected'])
        axis.scatter([chosen], [rows[chosen]['mean']], color=COLORS[1], s=75, zorder=3)
        axis.set_xticks(positions, [str(Fraction(row['proportion']).limit_denominator()) for row in rows])
        axis.set_title(maze['task'])
        axis.set_xlabel('Requested copy proportion p')
        axis.set_ylabel('Validation success')
        axis.grid(axis='y', alpha=0.2)
    figure.suptitle('Validation selection: checkpoint mean within seed, mean ± SE over seeds 42–46')
    save(figure, output, 'tuning')
    figure, axes = plt.subplots(2, 3, figsize=(12, 7.2), constrained_layout=True)
    for axis, maze in zip(axes.flat, result['mazes']):
        shown = set()
        for color, alias in zip(COLORS, ('S@selected', 'S@2/3', 'H@1/2', 'H@2/3', 'E')):
            name = maze['aliases'][alias]
            if name in shown:
                continue
            shown.add(name)
            points = maze['arms'][name]['points']
            x = [point['nominal_step'] / 1e6 for point in points]
            mean = [point['blocks']['test']['mean'] for point in points]
            error = [point['blocks']['test']['se'] for point in points]
            axis.plot(x, mean, color=color, label=name)
            axis.fill_between(x, [value - se for value, se in zip(mean, error)],
                              [value + se for value, se in zip(mean, error)], color=color, alpha=0.13)
        axis.set_title(maze['task'])
        axis.set_xlabel('Interactions (millions)')
        axis.set_ylabel('Test success')
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=6.5, loc='upper right')
    figure.suptitle('Registered held-pair trajectories: mean ± SE over seeds 42–61')
    save(figure, output, 'test-trajectories')
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for axis, maze in zip(axes.flat, result['mazes']):
        pairs = list(dict.fromkeys((item['left_alias'], item['right_alias']) for item in maze['comparisons']))
        for shift, marker, view, color in [(-0.13, 'o', 'checkpoint_mean', COLORS[0]), (0.13, 's', 'final_point', COLORS[1])]:
            records = [next(item for item in maze['comparisons'] if item['view'] == view and
                            (item['left_alias'], item['right_alias']) == pair)['difference'] for pair in pairs]
            axis.errorbar([row['mean'] for row in records], [index + shift for index in range(len(pairs))],
                          xerr=[[row['mean'] - row['ci_low'] for row in records],
                                [row['ci_high'] - row['mean'] for row in records]],
                          fmt=marker, color=color, capsize=3, label=view.replace('_', ' '))
        axis.set_yticks(range(len(pairs)), [left.replace('selected', 'p*') + ' − ' + right for left, right in pairs])
        axis.invert_yaxis()
        axis.axvline(0, color='#777777', linewidth=0.8)
        axis.set_title(maze['task'])
        axis.set_xlabel('Paired test-success difference')
        axis.xaxis.set_major_locator(MaxNLocator(4))
        axis.grid(axis='x', alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncol=2)
    figure.suptitle('Registered contrasts: marginal 95% t(19) intervals, seeds 42–61, no multiplicity correction')
    save(figure, output, 'paired-comparisons')
    return dict(output=str(output), figures=3, formats=['png', 'pdf'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(args.analysis, args.out)))
