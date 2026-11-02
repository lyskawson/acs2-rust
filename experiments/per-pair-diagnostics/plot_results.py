import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from protocol import ARMS, configurations

LABELS = dict(zip(ARMS, ['ACS2', 'Episodic ER', 'HER', 'Nontrivial HER', 'Shortened HER']))
COLORS = dict(zip(ARMS, ['#252525', '#4477aa', '#ee7733', '#cc3311', '#009988']))
FIGURES = [
    ('F', 'First-action optimality', 'first_action_optimality'),
    ('M', 'Goal independence', 'goal_independence'),
    ('D', 'Different first actions for disjoint optimal sets', 'discrimination'),
    ('winner_goal_mass', 'Winner specifying at least one goal attribute', 'goal_specific_winner'),
    ('F_mod', 'Optimality of the agent modal action', 'modal_optimality'),
    ('F_minus_F_mod', 'Gain over the agent modal action', 'modal_gain'),
    ('nonmodal_mass', 'Nonmodal decisions among all pairs', 'nonmodal_mass'),
    ('D_both_optimal', 'Both first actions optimal for disjoint optimal sets', 'both_actions_optimal'),
]


def number(row, key):
    return float(row[key]) if row[key] else None


def plot(root):
    root = Path(root)
    rows = list(csv.DictReader((root / 'post-hoc-curves.csv').open()))
    output = root / 'figures'
    output.mkdir(exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    for metric, title, filename in FIGURES:
        figure, axes = plt.subplots(2, 3, figsize=(13.2, 7), sharex=True)
        for axis, config in zip(axes.flat, configurations()):
            for arm in ARMS:
                selected = sorted([row for row in rows if row['view'] == 'pooled' and row['maze'] == config['task'] and row['arm'] == arm and row['metric'] == metric], key=lambda row: int(row['nominal_step']))
                valid = [row for row in selected if number(row, 'mean') is not None]
                x = [int(row['nominal_step']) for row in valid]
                y = [number(row, 'mean') for row in valid]
                axis.plot(x, y, color=COLORS[arm], label=LABELS[arm], linewidth=1.6)
                if all(number(row, 'ci_low') is not None for row in valid):
                    axis.fill_between(x, [number(row, 'ci_low') for row in valid], [number(row, 'ci_high') for row in valid], color=COLORS[arm], alpha=.10, linewidth=0)
            if metric == 'F':
                for reference, style, label in [('C', '--', 'Goal-blind ceiling'), ('random_reference', ':', 'Random reference')]:
                    selected = [row for row in rows if row['view'] == 'pooled' and row['maze'] == config['task'] and row['arm'] == 'acs2' and row['metric'] == reference]
                    if selected:
                        axis.axhline(number(selected[0], 'mean'), color='#777777', linestyle=style, linewidth=1, label=label)
            axis.set_xscale('log')
            axis.set_title(config['task'])
            axis.grid(alpha=.18)
            if metric == 'F_minus_F_mod':
                axis.axhline(0, color='#777777', linewidth=.7)
            else:
                axis.set_ylim(0, 1)
            axis.set_xlabel('Environment interactions')
            axis.set_ylabel('Fraction' if metric != 'F_minus_F_mod' else 'F - F_mod')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=len(labels), frameon=False, fontsize=8)
        figure.suptitle('Post hoc: ' + title + '\nRegistered block weights; five-seed means and marginal 95% t(4) intervals', fontsize=12)
        figure.tight_layout(rect=(0, .065, 1, .91))
        for extension in ('png', 'pdf'):
            figure.savefig(output / ('post_hoc_' + filename + '.' + extension), dpi=180, bbox_inches='tight')
        plt.close(figure)
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.8), sharey=True)
    for axis, config in zip(axes.flat, configurations()):
        bottom = [0.0] * len(ARMS)
        for count, color in [(0, '#9aa4ae'), (1, '#4477aa'), (2, '#009988')]:
            heights = []
            for arm in ARMS:
                selected = [row for row in rows if row['view'] == 'pooled' and row['maze'] == config['task'] and row['arm'] == arm and int(row['nominal_step']) == config['targets'][-1] and row['metric'] == 'winner_goal_attributes_' + str(count)]
                heights.append(number(selected[0], 'mean'))
            axis.bar(range(len(ARMS)), heights, bottom=bottom, color=color, label=str(count) + ' specified goal attributes')
            bottom = [left + right for left, right in zip(bottom, heights)]
        axis.set_xticks(range(len(ARMS)), ['ACS2', 'E', 'H', 'N', 'S'])
        axis.set_title(config['task'])
        axis.set_ylim(0, 1)
        axis.set_ylabel('Fraction of all first decisions')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    figure.suptitle('Post hoc: specified goal attributes of tied winners at the final point\nEqual fractional tie credit; fallbacks have no winner', fontsize=12)
    figure.tight_layout(rect=(0, .065, 1, .91))
    for extension in ('png', 'pdf'):
        figure.savefig(output / ('post_hoc_winner_goal_attributes.' + extension), dpi=180, bbox_inches='tight')
    plt.close(figure)
    index = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in output.iterdir() if path.is_file()}
    (output / 'checksums.json').write_text(json.dumps(dict(post_hoc=True, files=index), indent=2) + '\n')
    return dict(figures=len(index), post_hoc=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    print(json.dumps(plot(parser.parse_args().root)))
