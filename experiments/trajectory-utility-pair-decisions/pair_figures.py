import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pair_sources import E1, GROUPS, LAMBDAS, MEAN, OUTPUT, RANDOM, S1

BINS = ('1-2', '3-4', '5-6', '7+')
RAMP = ('#6baed6', '#4292c6', '#2171b5', '#08306b')
MARKERS = ('o', 's', '^', 'D')
TITLES = dict(no_aliasing='Without aliasing', aliasing='With aliasing')
LABELS = dict(F='F (optimal first action, non-fallback)', value_error='Estimate - Q* of the chosen action')
NAMES = {MEAN: 'Mean value', RANDOM: 'Random goal'}


def series(rows, arm, group, bin_name, measure):
    points = sorted((row['checkpoint'], row[measure]) for row in rows
                    if row['arm'] == arm and row['group'] == group and row['bin'] == bin_name and row[measure] is not None)
    return [point for point, _ in points], [value for _, value in points]


def figure(rows, agent, measure, out):
    fig, axes = plt.subplots(len(BINS), len(GROUPS), figsize=(11, 13), sharex=True, squeeze=False)
    styles = [(E1, 'E+1', '#969696', 'v', '--'), (S1, 'S+1', '#000000', 'x', '-')]
    styles += [(f'{agent}_{suffix}', f'{agent == MEAN and "M" or "R"} {share}', color, marker, '-')
               for (share, suffix), color, marker in zip(LAMBDAS, RAMP, MARKERS)]
    for row_index, bin_name in enumerate(BINS):
        for column, group in enumerate(GROUPS):
            axis = axes[row_index][column]
            for arm, label, color, marker, line in styles:
                x, y = series(rows, arm, group, bin_name, measure)
                axis.plot(x, y, line, color=color, marker=marker, markersize=5, linewidth=1.6, label=label)
            if measure == 'value_error':
                axis.axhline(0, color='#bdbdbd', linewidth=0.8)
            axis.set_xscale('log')
            axis.grid(color='#eeeeee', linewidth=0.6)
            axis.set_title(f'{TITLES[group]}, d {bin_name}', fontsize=10)
            if column == 0:
                axis.set_ylabel(LABELS[measure], fontsize=8)
            if row_index == len(BINS) - 1:
                axis.set_xlabel('Environment steps (checkpoint)')
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(f'Post hoc: {LABELS[measure]}, {NAMES[agent]}; pooled test pairs, mean of seeds 42-46; '
                 f'the 4M point is Woods14 alone', y=0.995, fontsize=11)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.975), ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    stem = out / f'{measure.replace("_", "-")}-{NAMES[agent].lower().replace(" ", "-")}'
    fig.savefig(stem.with_suffix('.pdf'))
    fig.savefig(stem.with_suffix('.png'), dpi=110)
    plt.close(fig)
    return stem.name


def main(args):
    rows = json.loads((args.analysis / 'arm-measures-bins.json').read_bytes())
    args.out.mkdir(parents=True, exist_ok=False)
    made = [figure(rows, agent, measure, args.out) for measure in ('F', 'value_error') for agent in (RANDOM, MEAN)]
    (args.out / 'figures.json').write_text(json.dumps(dict(post_hoc=True, figures=made), indent=1) + '\n')
    print(json.dumps(made))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, default=OUTPUT / 'analysis')
    parser.add_argument('--out', type=Path, default=OUTPUT / 'figures')
    main(parser.parse_args())
