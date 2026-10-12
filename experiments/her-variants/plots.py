import argparse
import json
from pathlib import Path

LABELS = {'acs2': 'ACS2', 'acs2er': 'ACS2ER U=3', 'acs2er_u1': 'ACS2ER U=1',
          'er_episodic': 'Episodic ER', 'her_full': 'HER full, p=2/3',
          'her_values': 'HER values, tuned p', 'her_values_shortened': 'HER values shortened, tuned p',
          'her_values_p2of3': 'HER values, p=2/3'}
COLORS = {'acs2': '#222222', 'acs2er': '#777777', 'acs2er_u1': '#a6761d',
          'er_episodic': '#1f77b4', 'her_full': '#d62728', 'her_values': '#2ca02c',
          'her_values_shortened': '#9467bd', 'her_values_p2of3': '#17becf'}


def curve(axis, arm, metric, name):
    points = arm['points']
    x = [point['nominal_step'] for point in points]
    y = [point['metrics'][metric]['mean'] for point in points]
    se = [point['metrics'][metric]['se'] for point in points]
    axis.plot(x, y, marker='o', markersize=3, linewidth=1.5, label=LABELS[name], color=COLORS[name])
    axis.fill_between(x, [mean-error for mean,error in zip(y,se)], [mean+error for mean,error in zip(y,se)],
                      color=COLORS[name], alpha=0.12)
    axis.set_xscale('log')
    axis.grid(alpha=0.25)
    axis.set_xlabel('Nominal environment interactions')


def save(figure, destination, name):
    figure.savefig(destination / (name + '.svg'), bbox_inches='tight')
    figure.savefig(destination / (name + '.png'), dpi=180, bbox_inches='tight')


def plot(results, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    figure, axes = plt.subplots(2, 3, figsize=(18, 11))
    for axis, maze in zip(axes.flat, results['configurations']):
        pairs = list(reversed(maze['comparisons']))
        for index, pair in enumerate(pairs):
            value = pair['primary_difference']
            axis.errorbar(value['mean'], index, xerr=[[value['mean']-value['ci_low']], [value['ci_high']-value['mean']]],
                          fmt='o', color='#222222', capsize=3)
        axis.axvline(0, color='#888888', linewidth=1)
        axis.set_yticks(range(len(pairs)), [LABELS[pair['left']] + '\n− ' + LABELS[pair['right']] for pair in pairs], fontsize=7)
        axis.set_title(maze['task'])
        axis.set_xlabel('Paired primary difference, marginal 95% t(19) interval')
        axis.grid(axis='x', alpha=0.25)
    figure.suptitle('Held-pair test success: mean over registered points within each of 20 seeds', fontsize=13)
    figure.tight_layout()
    save(figure, destination, 'paired-primary-comparisons')
    plt.close(figure)
    for maze in results['configurations']:
        pairs = [pair for pair in maze['comparisons'] if 'model_updates_and_knowledge' in pair]
        figure, axes = plt.subplots(len(pairs), 3, figsize=(16, 3.0*len(pairs)), squeeze=False)
        for row, pair in zip(axes, pairs):
            left, right = pair['left'], pair['right']
            value = pair['primary_difference']
            row[0].errorbar(value['mean'], 0, xerr=[[value['mean']-value['ci_low']], [value['ci_high']-value['mean']]],
                            fmt='o', color='#222222', capsize=4)
            row[0].axvline(0, color='#888888', linewidth=1)
            row[0].set_yticks([])
            row[0].set_title(LABELS[left] + '\n− ' + LABELS[right])
            row[0].set_xlabel('Primary difference, marginal 95% interval')
            row[0].grid(axis='x', alpha=0.25)
            for name in [left, right]:
                curve(row[1], maze['arms'][name], 'knowledge', name)
                curve(row[2], maze['arms'][name], 'full_steps_per_environment_step', name)
            row[1].set_ylabel('Knowledge, mean ± SE')
            row[1].set_ylim(0, 1.03)
            row[2].set_ylabel('Full learning steps / actual environment step, mean ± SE')
            row[2].set_ylim(0, 3.15)
            row[1].legend(fontsize=7)
        figure.suptitle(maze['task'] + ': values comparisons with observed model update volume and knowledge', fontsize=13)
        figure.tight_layout()
        save(figure, destination, maze['configuration'] + '-values-comparisons')
        plt.close(figure)
    for metric, ylabel in [('success','Test success'), ('knowledge','Knowledge'), ('population','Population classifiers')]:
        figure, axes = plt.subplots(2, 3, figsize=(16, 10))
        for axis, maze in zip(axes.flat, results['configurations']):
            for name, arm in maze['arms'].items():
                curve(axis, arm, metric, name)
            axis.set_title(maze['task'])
            axis.set_ylabel(ylabel + ', mean ± SE')
            if metric == 'knowledge': axis.set_ylim(0, 1.03)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=4, fontsize=8)
        figure.tight_layout(rect=(0,0.08,1,1))
        save(figure, destination, metric + '-curves')
        plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    plot(json.loads(args.results.read_text()), args.destination)
