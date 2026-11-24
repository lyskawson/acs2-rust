import argparse
from pathlib import Path
from statistics import mean

from grid_results import GROUPS, NEW_ARMS, REFERENCE_ARMS, csv_rows
from proportions import ABLATION, MEAN, RANDOM, S1, SHARES, label, read, registered


def figures(analysis, output):
    import success_curves
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _, _, configs, _ = registered()
    names = read(ABLATION / 'maze-names.json')
    shares = (0.25, 0.5, 0.75, 1.0)
    colors = {MEAN: '#2289b5', RANDOM: '#d68b32'}
    symbols = dict(zip(shares, ['o', '^', 's', 'D']))
    success_curves.ARMS = REFERENCE_ARMS + NEW_ARMS
    success_curves.COLORS = {S1: '#cf493e'}
    success_curves.MARKERS = {S1: 'P'}
    success_curves.LABELS = {S1: 'S+1'}
    for agent, name in ((MEAN, 'Mean value'), (RANDOM, 'Random goal')):
        for share in shares:
            arm = label(agent, share)
            success_curves.COLORS[arm] = colors[agent]
            success_curves.MARKERS[arm] = symbols[share]
            success_curves.LABELS[arm] = f'{name} λ = {share:g}'
    report = success_curves.plot(analysis/'curves-registered.csv', output, list(configs.values()), names)

    all_shares = (0.0, 0.25, 0.5, *SHARES[1:])
    scores = {(row['task'], row['arm']): float(row['mean'])
              for row in csv_rows(analysis/'arm-success.csv')
              if row['test_cap']=='registered' and row['measure']=='primary'}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    for group, axis, title in zip(GROUPS, axes, ('No observation aliasing', 'Observation aliasing')):
        tasks = [task for task, cfg in configs.items() if cfg['group']==group]
        for agent, name, marker in ((MEAN, 'Mean value', 'o'), (RANDOM, 'Random goal', 's')):
            group_means = [mean(scores[task, S1 if share==0 else label(agent, share)]
                                for task in tasks) for share in all_shares]
            axis.plot(all_shares, group_means, label=name, color=colors[agent], marker=marker)
        axis.set_title(title)
        axis.set_xlabel('TU share among requested copies (λ)')
        axis.set_xticks(all_shares, ['0', '¼', '½', '¾', '1'])
        axis.set_ylim(0, 1)
        axis.grid(alpha=.15)
    axes[0].set_ylabel('Primary test success')
    axes[1].legend(frameon=False, loc='lower right')
    fig.text(.5, .02, 'Mean over mazes and 20 seeds, registered caps. λ = 0 is S+1.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.06,1,1))
    for extension in ('png','pdf'):
        target = output/f'primary-success-by-tu-share.{extension}'
        if extension=='pdf':
            fig.savefig(target, metadata={'CreationDate': None, 'ModDate': None})
        else:
            fig.savefig(target, dpi=200)
    plt.close(fig)
    return report


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(figures(args.analysis, args.out)))
