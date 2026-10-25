import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'acs2her_nontrivial': 'Nontrivial', 'acs2her_shortened': 'Shortened', 'acs2her': 'HER',
         'acs2er_episodic': 'Episodic replay', 'acs2': 'ACS2', 'acs2er': 'ACS2ER'}
COLORS = {'acs2her_nontrivial': '#246ab0', 'acs2her_shortened': '#c74734', 'acs2her': '#7b4c9d',
          'acs2er_episodic': '#26866c', 'acs2': '#5c626b', 'acs2er': '#b39d29'}
VIEWS = [('checkpoint_mean', 'Mean across registered points'), ('final_point', 'Final registered point')]


def save(figure, directory, name):
    for suffix in ['png', 'svg']:
        figure.savefig(directory / (name + '.' + suffix), dpi=180, bbox_inches='tight')
    plt.close(figure)


def plot(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    mazes = results['mazes']
    contrasts = results['analysis_rules']['comparisons']
    figure, axes = plt.subplots(6, 2, figsize=(13, 15), sharex=True)
    for index, maze in enumerate(mazes):
        for column, (view, title) in enumerate(VIEWS):
            ax = axes[index, column]
            ax.axvline(0, color='#777777', linewidth=0.8)
            for position, (left, right) in enumerate(contrasts):
                result = next(item['difference'] for item in maze['comparisons'] if (item['view'], item['left'], item['right']) == (view, left, right))
                ax.errorbar(result['mean'], position, xerr=[[result['mean'] - result['ci_low']], [result['ci_high'] - result['mean']]],
                            fmt='o', color=COLORS[left], capsize=3, markersize=4)
            ax.set_yticks(range(5), [NAMES[left] + ' − ' + NAMES[right] for left, right in contrasts] if column == 0 else [''] * 5)
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.2)
            ax.set_title(maze['task'].upper() + (' · ' + title if index == 0 else ''), loc='left')
    for ax in axes[-1]:
        ax.set_xlabel('Test-success difference; paired marginal 95% t(19) interval')
    figure.suptitle('Full hindsight replay under goal admissibility · 20 paired seeds (42–61)', fontsize=13, y=1.005)
    figure.tight_layout(h_pad=1.3, w_pad=1.1)
    save(figure, output, 'paired-success-differences')
    figure, axes = plt.subplots(3, 2, figsize=(12, 10))
    for ax, maze in zip(axes.flat, mazes):
        for name in NAMES:
            points = maze['arms'][name]['points']
            x = [point['nominal_step'] for point in points]
            means = [point['blocks']['test']['mean'] for point in points]
            errors = [point['blocks']['test']['se'] for point in points]
            ax.errorbar(x, means, yerr=errors, label=NAMES[name], color=COLORS[name],
                        linestyle='--' if name in ('acs2', 'acs2er') else '-', marker='o', markersize=3, linewidth=1)
        ax.set_xscale('log')
        ax.set_title(maze['task'].upper(), loc='left')
        ax.set_xlabel('Environment interactions (registered nominal points)')
        ax.set_ylabel('Held-pair test success, mean ± SE')
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.025))
    figure.tight_layout(h_pad=1.8)
    save(figure, output, 'test-success-curves')
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    names = ['acs2her_nontrivial', 'acs2her_shortened', 'acs2her', 'acs2er_episodic']
    for column, name in enumerate(names):
        positions = [index + (column - 1.5) * 0.19 for index in range(6)]
        for ax, metric, label in zip(axes, ['test_value_gap_all', 'population_classifiers'],
                                     ['First-action value minus discounted return', 'Population classifiers']):
            measured = [maze['arms'][name]['diagnostics']['final_point'][metric] for maze in mazes]
            ax.bar(positions, [cell['mean'] for cell in measured], width=0.18, yerr=[cell['se'] for cell in measured],
                   label=NAMES[name], color=COLORS[name], capsize=2)
            ax.set_ylabel(label + '\nFinal point, mean ± SE over 20 seeds')
            ax.grid(axis='y', alpha=0.2)
    axes[-1].set_xticks(range(6), [maze['task'].upper() for maze in mazes])
    axes[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.2))
    figure.tight_layout(h_pad=1.8)
    save(figure, output, 'value-gap-and-population')
    return dict(figures=3, formats=['png', 'svg'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(plot(json.loads(args.results.read_text()), args.out)))
