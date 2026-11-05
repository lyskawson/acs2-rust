import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from protocol import ARM, G, S, T4

ARMS=(ARM,G,S)
LABELS={ARM:'R: random reward skips',G:'G: goal-rule copies',S:'S: shortened HER'}
COLORS={ARM:'#3366aa',G:'#087e8b',S:'#d9822b'}
MAZES=('maze4','maze5','maze6','maze7','woods14','mazema')


def summarize_curves(rows):
    groups=defaultdict(dict)
    for row in rows:
        if row['arm'] not in ARMS:
            continue
        if row.get('post_hoc') is not True or row['view']!='pooled':
            raise ValueError('curves must carry the registered pooled-view post-hoc label')
        for metric in ('F','F_mod'):
            key=row['maze'],row['arm'],row['nominal_step'],metric
            if row['seed'] in groups[key]:
                raise ValueError('duplicate curve seed')
            groups[key][row['seed']]=row[metric]
    result=[]
    for (maze,arm,point,metric),seeds in sorted(groups.items()):
        if set(seeds)!=set(range(42,47)):
            raise ValueError('curves require exactly seeds42--46')
        values=[seeds[seed] for seed in range(42,47)]
        if any(value is None for value in values):
            mean=se=low=high=None
        else:
            mean=statistics.mean(values)
            se=statistics.stdev(values)/5**.5
            low,high=mean-T4*se,mean+T4*se
        result.append(dict(post_hoc=True,view='pooled',maze=maze,arm=arm,nominal_step=point,metric=metric,
            n=sum(value is not None for value in values),mean=mean,se=se,ci_low=low,ci_high=high))
    return result


def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=Path(directory)
    content=json.loads((directory/'post-hoc-curves.json').read_text())
    rows=summarize_curves(content['rows'])
    figures=directory/'figures'
    figures.mkdir(exist_ok=False)
    with (directory/'post-hoc-curve-summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def draw(axes,maze):
        for axis,metric,title in zip(axes,('F','F_mod'),('Chosen first action optimal (F)','Modal action optimal (F_mod)')):
            for arm in ARMS:
                values=[row for row in rows if row['maze']==maze and row['arm']==arm and row['metric']==metric]
                values.sort(key=lambda row:row['nominal_step'])
                points=[row['nominal_step'] for row in values]
                mean=[float('nan') if row['mean'] is None else row['mean'] for row in values]
                low=[float('nan') if row['ci_low'] is None else row['ci_low'] for row in values]
                high=[float('nan') if row['ci_high'] is None else row['ci_high'] for row in values]
                axis.plot(points,mean,label=LABELS[arm],color=COLORS[arm],linewidth=2)
                axis.fill_between(points,low,high,color=COLORS[arm],alpha=.13,linewidth=0)
            axis.set_xscale('log')
            axis.set_ylim(-.03,1.03)
            axis.set_title(title)
            axis.set_xlabel('Nominal environment interactions')
            axis.set_ylabel('Weighted fraction')
            axis.grid(alpha=.18)
    for maze in MAZES:
        fig,axes=plt.subplots(1,2,figsize=(11,4.7),constrained_layout=True)
        draw(axes,maze)
        axes[0].legend(loc='lower right',frameon=False)
        fig.suptitle(maze.upper()+' | post hoc, seeds42--46, mean and marginal 95% t(4) intervals')
        for suffix in ('png','svg'):
            fig.savefig(figures/(maze+'-first-actions.'+suffix),dpi=220)
        plt.close(fig)
    fig,axes=plt.subplots(6,2,figsize=(12,20),constrained_layout=True)
    for index,maze in enumerate(MAZES):
        draw(axes[index],maze)
        axes[index,0].set_ylabel(maze.upper()+'\nWeighted fraction')
    axes[0,0].legend(loc='lower right',frameon=False)
    fig.suptitle('First-action optimality during training | post hoc, pooled blocks, seeds42--46\nMean and marginal 95% t(4) intervals; F and F_mod exclude random fallbacks')
    for suffix in ('png','svg'):
        fig.savefig(figures/('first-actions-over-training.'+suffix),dpi=180)
    plt.close(fig)
    return dict(post_hoc=True,curve_cells=len(rows),figure_files=14,plan_sha256=content['plan_sha256'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    print(json.dumps(plot(parser.parse_args().directory)))
