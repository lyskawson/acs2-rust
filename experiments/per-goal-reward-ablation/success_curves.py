import argparse
import csv
import json
import math
from pathlib import Path

from protocol import HERE,ARMS
from grid_execution import registered_plan


LABELS = dict(zip(ARMS, ['ACS2', 'ACS2 r(g)', 'E', 'S', 'E+1', 'S+1']))
COLORS = dict(zip(ARMS, ['#555555', '#111111', '#875bb6', '#d68b32', '#2289b5', '#cf493e']))
MARKERS = dict(zip(ARMS, ['o', 's', '^', 'D', 'v', 'P']))


def read_curves(path, configs, names):
    with Path(path).open() as stream:
        rows = list(csv.DictReader(stream))
    expected = {(config['task'], arm, step) for config in configs for arm in ARMS for step in config['targets']}
    lookup = {config['task']: config for config in configs}
    seen, result = set(), []
    for raw in rows:
        task, arm, step = raw['task'], raw['arm'], int(raw['nominal_step'])
        key = task, arm, step
        if key not in expected or key in seen:
            raise ValueError('curve checkpoint, arm or maze differs')
        seen.add(key)
        mean = float(raw['mean'])
        if (int(raw['n']) != 20 or not math.isfinite(mean) or not 0 <= mean <= 1 + 1e-12
                or raw['group'] != lookup[task]['group'] or raw['maze'] != names[task]):
            raise ValueError('curve seed count, group, name or mean differs')
        result.append(dict(raw, nominal_step=step, mean=mean, n=20))
    if seen != expected:
        raise ValueError('all registered curve points are required')
    return result


def plot(data, output, configs=None, names=None):
    configs = configs or registered_plan()['configurations']
    names = names or json.loads((HERE / 'maze-names.json').read_text())
    rows = read_curves(data, configs, names)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.ticker import FuncFormatter
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)

    def draw(axis, config):
        task = config['task']
        targets = config['targets']
        for arm in ARMS:
            values = sorted((row for row in rows if row['task'] == task and row['arm'] == arm),
                            key=lambda row: row['nominal_step'])
            axis.plot([row['nominal_step'] for row in values], [row['mean'] for row in values],
                      label=LABELS[arm], color=COLORS[arm], marker=MARKERS[arm], markersize=3,
                      linewidth=1.6, linestyle='--' if arm in (ARMS[0], ARMS[2], ARMS[3]) else '-')
        axis.axhline(.95, color='#aaaaaa', linewidth=.7, linestyle=':')
        axis.set_xscale('log')
        axis.set_xlim(targets[0] * .95, targets[-1] * 1.05)
        axis.set_ylim(-.02, 1.02)
        axis.set_yticks([0, .25, .5, .75, 1])
        axis.set_xticks([step for step in [1000, 10000, 100000, 1000000] if step <= targets[-1]])
        axis.xaxis.set_major_formatter(FuncFormatter(
            lambda value, _: f'{value / 1000000:g}M' if value >= 1000000 else f'{value / 1000:g}k'))
        axis.set_xlabel('Environment interactions')
        axis.set_ylabel('Test success')
        axis.grid(alpha=.18)
        axis.set_title(f"{names[task]} | {100 * config.get('aliased_share', 0):.1f}% aliased cells", fontsize=10)

    caption = 'Points show registered checkpoint means. The dotted line marks 0.95.'
    pdf_metadata = {'CreationDate': None, 'ModDate': None}
    for config in configs:
        folder = output / config['group']
        folder.mkdir(exist_ok=True)
        fig, axis = plt.subplots(figsize=(6.6, 4.5))
        draw(axis, config)
        handles, labels = axis.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, frameon=False)
        fig.suptitle('Test block | mean over seeds 42–61', fontsize=11)
        fig.text(.5, .005, caption, ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .16, 1, .95))
        fig.savefig(folder / (config['task'] + '.png'), dpi=200)
        fig.savefig(folder / (config['task'] + '.pdf'), metadata=pdf_metadata)
        plt.close(fig)
    groups = {}
    for group, title in [('no_aliasing', 'No observation aliasing'), ('aliasing', 'Observation aliasing')]:
        selected = [config for config in configs if config['group'] == group]
        if not selected:
            continue
        groups[group] = [config['task'] for config in selected]
        with PdfPages(output / (group + '-test-success.pdf'), metadata=pdf_metadata) as pdf:
            for start in range(0, len(selected), 9):
                page = selected[start:start + 9]
                page_number = start // 9 + 1
                columns = min(3, len(page))
                height = math.ceil(len(page) / columns)
                fig, axes = plt.subplots(height, columns, figsize=(max(6.6, 4 * columns), 3.1 * height + 1), squeeze=False)
                flat = list(axes.flat)
                for axis, config in zip(flat, page):
                    draw(axis, config)
                    axis.tick_params(labelsize=8)
                for axis in flat[len(page):]:
                    axis.set_visible(False)
                handles, labels = flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .025),
                           ncol=3 if columns == 1 else 6, frameon=False)
                fig.suptitle(f'{title} | test block, means over 20 seeds | page {page_number}', fontsize=13)
                fig.text(.5, .005, caption, ha='center', fontsize=8)
                fig.tight_layout(rect=(0, .17 if columns == 1 else .07, 1, .95))
                pdf.savefig(fig)
                fig.savefig(output / f'{group}-page-{page_number}.png', dpi=160)
                plt.close(fig)
    metadata = dict(mazes=len(configs), arms=list(ARMS), groups=groups, individual_figures=len(configs),
        group_pages=sum(math.ceil(len(tasks) / 9) for tasks in groups.values()),
        block='test', seeds=list(range(42, 62)), curves='Means at every registered checkpoint.',
        uncertainty='The analysis tables contain the registered paired primary and final contrast intervals.')
    (output / 'figures.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.data, args.out)))
