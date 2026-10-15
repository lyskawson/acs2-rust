import argparse
import csv
import hashlib
import json
from pathlib import Path

LABELS = dict(acs2vcp_k2='VCP K=2, T=24', acs2vcp_k4='VCP K=4, T=24',
              acs2vcp_k2_uniform='Uniform K=2', acs2vcp_k4_uniform='Uniform K=4',
              acs2her='HER', acs2er_episodic='Episodic ER', acs2='ACS2', acs2er='ACS2ER')
COLORS = dict(acs2vcp_k2='#0072b2', acs2vcp_k4='#cc79a7', acs2vcp_k2_uniform='#0072b2',
              acs2vcp_k4_uniform='#cc79a7', acs2her='#009e73', acs2er_episodic='#d55e00',
              acs2='#222222', acs2er='#777777')
TASKS = dict(maze4='Maze4', maze5='Maze5', maze6='Maze6', maze7='Maze7', woods14='Woods14', mazema='MazeMA')


def checked_tables(source):
    source = Path(source)
    hashes = json.loads((source / 'files.sha256.json').read_text())
    for name, digest in hashes.items():
        path = source / name
        if not path.resolve().is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('analysis artifact checksum differs')
    results = json.loads((source / 'results.json').read_text())
    if (results['approved_plan_sha256'] != '9428773fa7861f6102fdccb165ae711b597d1fd61a9e644f8becb17399ebf4eb'
            or results['verification']['runs'] != 480 or results['verification']['pilot_identity_rows'] != 180):
        raise ValueError('plots require the fully verified approved analysis')
    tables = {}
    for name in ['learning-curves', 'point-diagnostics', 'comparisons']:
        with (source / (name + '.csv')).open() as stream:
            tables[name] = list(csv.DictReader(stream))
    return results, tables


def curve(axis, rows, name):
    rows = sorted(rows, key=lambda row: int(row['nominal_step']))
    x = [int(row['nominal_step']) for row in rows]
    y = [float(row['mean']) for row in rows]
    errors = [float(row['se']) for row in rows]
    line = '--' if name.endswith('_uniform') else ':' if name in ['acs2', 'acs2er'] else '-'
    axis.plot(x, y, color=COLORS[name], linestyle=line, marker='o', markersize=3,
              linewidth=1.5, label=LABELS[name])
    axis.fill_between(x, [value-error for value, error in zip(y, errors)],
                      [value+error for value, error in zip(y, errors)], color=COLORS[name], alpha=.08)


def save(figure, output, name):
    for suffix in ['svg', 'png']:
        figure.savefig(output / (name + '.' + suffix), dpi=180, bbox_inches='tight')


def plot(source, output):
    results, tables = checked_tables(source)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    specifications = [('test-success', 'Test success, mean ± SE', None),
                      ('head-disagreement', 'Head-value population variance, mean ± SE', 'evaluation.test.head_disagreement.mean'),
                      ('head-disagreement-exact-zeros', 'Exact-zero disagreement share, mean ± SE', 'evaluation.test.head_disagreement.exact_zero_share')]
    for filename, ylabel, metric in specifications:
        figure, axes = plt.subplots(2, 3, figsize=(15, 9))
        for axis, maze in zip(axes.flat, results['configurations']):
            identifier = maze['configuration']
            rows = [row for row in tables['point-diagnostics' if metric else 'learning-curves']
                    if row['configuration'] == identifier and (row.get('metric') == metric if metric else row['block'] == 'test')]
            for name in LABELS:
                group = [row for row in rows if row['arm'] == name]
                if group:
                    curve(axis, group, name)
            axis.set_title(TASKS[identifier.split('_')[0]])
            axis.set_xscale('log')
            axis.set_xlabel('Nominal environment interactions')
            axis.set_ylabel(ylabel)
            axis.grid(alpha=.2)
            if filename == 'head-disagreement':
                axis.set_yscale('symlog', linthresh=1)
                axis.set_ylim(bottom=0)
            else:
                axis.set_ylim(0, 1.02)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc='lower center', ncol=4)
        detail = 'heads averaged within each seed' if metric is None else 'fixed non-identity block transitions within each seed'
        figure.suptitle('Held-pair test block; 20 seeds, ' + detail)
        figure.tight_layout(rect=(0, .08, 1, .96))
        save(figure, output, filename)
        plt.close(figure)
    for view, filename, title in [('checkpoint_mean_within_seed', 'paired-primary', 'Mean across registered checkpoints within seed'),
                                  ('final_point', 'paired-final', 'Final registered point')]:
        figure, axes = plt.subplots(2, 3, figsize=(20, 11))
        for axis, maze in zip(axes.flat, results['configurations']):
            identifier = maze['configuration']
            rows = [row for row in tables['comparisons'] if row['configuration'] == identifier and row['view'] == view]
            for index, row in enumerate(rows):
                mean, lower, upper = map(float, [row['mean'], row['ci_low'], row['ci_high']])
                axis.errorbar(mean, index, xerr=[[mean-lower], [upper-mean]], fmt='o', capsize=3, color=COLORS[row['left']])
            axis.axvline(0, color='#888888', linewidth=1)
            axis.set_yticks(range(len(rows)), [LABELS[row['left']] + '\n− ' + LABELS[row['right']] for row in rows], fontsize=7)
            axis.invert_yaxis()
            axis.set_title(TASKS[identifier.split('_')[0]])
            axis.set_xlabel('Paired success difference; marginal 95% t(19) interval')
            axis.grid(axis='x', alpha=.2)
        figure.suptitle(title + '; test block, 20 paired seeds, no multiplicity correction')
        figure.tight_layout(rect=(0, 0, 1, .96))
        save(figure, output, filename)
        plt.close(figure)
    (output / 'software.json').write_text(json.dumps(dict(matplotlib=matplotlib.__version__), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    plot(args.analysis, args.output)
