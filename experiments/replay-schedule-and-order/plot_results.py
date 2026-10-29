import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from approved_execution import APPROVED_SHA256
from protocol import ROOT
from collect import outside_repository
from grid_protocol import load_plan

COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']


def save(figure, output, name):
    figure.savefig(output / (name + '.png'), dpi=180)
    figure.savefig(output / (name + '.pdf'))
    plt.close(figure)


def plot(analysis, output):
    result = json.loads((Path(analysis) / 'results.json').read_text())
    plan = load_plan(ROOT / 'experiments/replay-schedule-and-order/plan.json', APPROVED_SHA256)
    if result['plan_sha256'] != APPROVED_SHA256:
        raise ValueError('plot input differs from the approved registration')
    output = outside_repository(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    modes = ['base', '3to1at30000', '3to1at100000', '3to1at300000', 'backward']
    for axis, maze in zip(axes.flat, result['mazes']):
        rows = {row['setting']: row for row in result['screening_table'] if row['configuration_id'] == maze['configuration_id']}
        axis.errorbar(range(5), [rows[mode]['mean'] for mode in modes], yerr=[rows[mode]['se'] for mode in modes],
                      color=COLORS[0], marker='o', capsize=3, linestyle='none')
        for family, marker, color in [('B', 's', COLORS[1]), ('C', 'x', '#222222')]:
            position = modes.index(maze['selections'][family]['setting'])
            options = dict(facecolors='none', edgecolors=color) if marker == 's' else dict(color=color)
            axis.scatter([position], [rows[modes[position]]['mean']], marker=marker, s=90,
                         label='Selected ' + family, zorder=4, **options)
        axis.set_xticks(range(5), ['Base', '30k', '100k', '300k', 'Backward'])
        axis.set_title(maze['task'] + ', p=' + str(Fraction(plan['base_proportions'][maze['task']]).limit_denominator()))
        axis.set_ylabel('Validation success')
        axis.set_xlabel('Replay setting; schedule switch in interactions')
        axis.grid(axis='y', alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='outside lower center', ncols=2)
    figure.suptitle('Validation screening: checkpoint mean ± SE, five seeds')
    save(figure, output, 'validation-screening')
    for family in ('B', 'C'):
        mazes = [maze for maze in result['mazes'] if family in maze['aliases']]
        if not mazes:
            continue
        figure, axes = plt.subplots(1, len(mazes), figsize=(4.5 * len(mazes), 4), squeeze=False, constrained_layout=True)
        for axis, maze in zip(axes.flat, mazes):
            aliases = maze['aliases'][family]
            for alias, color, style in [('S', COLORS[0], '--'), ('S_X', COLORS[0], '-'), ('E', COLORS[1], '--'), ('E_X', COLORS[1], '-')]:
                points = maze['arms'][aliases[alias]]['points']
                x = [point['nominal_step'] for point in points]
                y = [point['blocks']['test']['mean'] for point in points]
                error = [point['blocks']['test']['se'] for point in points]
                axis.plot(x, y, color=color, linestyle=style, label=alias)
                axis.fill_between(x, [mean - se for mean, se in zip(y, error)], [mean + se for mean, se in zip(y, error)], color=color, alpha=0.12)
            axis.set_xscale('log')
            axis.set_title(maze['task'] + ', ' + maze['selections'][family]['setting'])
            axis.set_xlabel('Environment interactions')
            axis.set_ylabel('Test success')
            axis.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='outside lower center', ncols=4)
        figure.suptitle(f'Replay axis {family}: mean ± SE, twenty seeds; S=shortened HER, E=episodic ER')
        save(figure, output, 'test-curves-' + family)
        names = plan['analysis']['comparisons'] + ([plan['analysis']['C_additional']] if family == 'C' else [])
        figure, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6), squeeze=False, constrained_layout=True)
        for view_index, view in enumerate(('checkpoint_mean', 'final_point')):
            for column, name in enumerate(names):
                axis = axes[view_index, column]
                rows = [next(value['difference'] for value in maze['comparisons'] if
                    value['axis'] == family and value['view'] == view and value['comparison'] == name) for maze in mazes]
                means = [row['mean'] for row in rows]
                errors = [[row['mean'] - row['ci_low'] for row in rows], [row['ci_high'] - row['mean'] for row in rows]]
                axis.errorbar(means, range(len(mazes)), xerr=errors, fmt='o', color=COLORS[view_index], capsize=3)
                axis.axvline(0, color='#555555', linestyle=':', linewidth=1)
                axis.set_yticks(range(len(mazes)), [maze['task'] for maze in mazes])
                axis.invert_yaxis()
                axis.set_title(name)
                axis.set_xlabel('Success difference')
                axis.grid(axis='x', alpha=0.2)
                if column == 0:
                    axis.set_ylabel('Checkpoint mean' if view_index == 0 else 'Final checkpoint')
        figure.suptitle(f'Axis {family}: paired contrasts, marginal 95% t(19) intervals; no multiplicity correction')
        save(figure, output, 'paired-contrasts-' + family)
    return dict(output=str(output), figures=len(list(output.glob('*.pdf'))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.analysis, args.out)))
