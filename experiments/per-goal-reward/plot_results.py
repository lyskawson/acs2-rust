import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from protocol import E, E1, S, S1, T4, configurations

ARMS = (E, E1, S, S1)
LABELS = {E: 'E: episodic ER', E1: 'E+1: per-goal values', S: 'S: shortened HER', S1: 'S+1: per-goal values'}
COLORS = {E: '#3268a8', E1: '#3268a8', S: '#c06025', S1: '#c06025'}


def summarize(rows):
    groups = defaultdict(dict)
    for row in rows:
        if row['arm'] not in ARMS or row.get('post_hoc') is not True or row['view'] != 'pooled':
            raise ValueError('curves require the four registered arms and explicit pooled post-hoc labels')
        for metric in ('F', 'F_mod'):
            key = row['maze'], row['arm'], row['nominal_step'], metric
            if row['seed'] in groups[key]:
                raise ValueError('duplicate curve seed')
            groups[key][row['seed']] = row[metric]
    expected = {(config['task'], arm, point, metric) for config in configurations() for arm in ARMS
                for point in config['targets'] for metric in ('F', 'F_mod')}
    if set(groups) != expected:
        raise ValueError('curves require every registered maze, arm and checkpoint')
    result = []
    for (maze, arm, point, metric), seeds in sorted(groups.items()):
        if set(seeds) != set(range(42, 47)):
            raise ValueError('curves require exactly seeds42--46')
        values = [seeds[seed] for seed in range(42, 47)]
        if any(value is None for value in values):
            mean = se = low = high = None
        else:
            mean = statistics.mean(values)
            se = statistics.stdev(values) / 5 ** .5
            low, high = mean - T4 * se, mean + T4 * se
        result.append(dict(post_hoc=True, view='pooled', maze=maze, arm=arm, nominal_step=point, metric=metric,
            n=sum(value is not None for value in values), mean=mean, se=se, ci_low=low, ci_high=high))
    return result


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory = Path(directory)
    data = json.loads((directory / 'post-hoc-curves.json').read_text())
    rows = summarize(data['rows'])
    figures = directory / 'figures'
    figures.mkdir(exist_ok=False)
    with (directory / 'post-hoc-curve-summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    def draw(axes, maze):
        for axis, metric, title in zip(axes, ('F', 'F_mod'), ('Chosen first action optimal (F)', 'Modal action optimal (F_mod)')):
            for arm in ARMS:
                values = sorted((row for row in rows if row['maze'] == maze and row['arm'] == arm and row['metric'] == metric),
                    key=lambda row: row['nominal_step'])
                points = [row['nominal_step'] for row in values]
                get = lambda key: [float('nan') if row[key] is None else row[key] for row in values]
                axis.plot(points, get('mean'), label=LABELS[arm], color=COLORS[arm], linewidth=2,
                    linestyle='--' if arm in (E, S) else '-')
                axis.fill_between(points, get('ci_low'), get('ci_high'), color=COLORS[arm], alpha=.10, linewidth=0)
            axis.set_xscale('log')
            axis.set_ylim(-.04, 1.04)
            axis.set_title(title)
            axis.set_xlabel('Nominal environment interactions')
            axis.set_ylabel('Weighted fraction')
            axis.grid(alpha=.18)
    configs = configurations()
    for config in configs:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
        draw(axes, config['task'])
        axes[0].legend(loc='lower right', frameon=False)
        fig.suptitle(config['task'].upper() + ' | post hoc, pooled blocks, seeds42--46, marginal 95% t(4) intervals')
        for suffix in ('png', 'svg'):
            fig.savefig(figures / (config['task'] + '-first-actions.' + suffix), dpi=220)
        plt.close(fig)
    fig, axes = plt.subplots(6, 2, figsize=(12, 20), constrained_layout=True)
    for index, config in enumerate(configs):
        draw(axes[index], config['task'])
        axes[index, 0].set_ylabel(config['task'].upper() + '\nWeighted fraction')
    axes[0, 0].legend(loc='lower right', frameon=False)
    fig.suptitle('First-action optimality | post hoc, pooled blocks, seeds42--46\nMean and marginal 95% t(4) intervals; random fallbacks excluded')
    for suffix in ('png', 'svg'):
        fig.savefig(figures / ('first-actions-over-training.' + suffix), dpi=180)
    plt.close(fig)
    return dict(post_hoc=True, curve_cells=len(rows), figure_files=14, plan_sha256=data['plan_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(plot(parser.parse_args().directory)))
