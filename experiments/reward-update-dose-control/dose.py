import statistics

from protocol import ARM, G
from random_rule_schema import probability


def shares(row):
    routes = row['replay_diagnostics']['reward_updates']
    skipped = lambda route: route.get('skipped_with_goal', 0) + route['skipped_without_goal']
    relabeled = routes['relabeled']
    total = sum(route['action_set_rules'] for route in routes.values())
    if not total or not relabeled['action_set_rules']:
        raise ValueError('dose requires positive relabeled and total opportunities')
    return dict(relabeled=skipped(relabeled) / relabeled['action_set_rules'],
                all=sum(skipped(route) for route in routes.values()) / total)


def check(finals, configs, seeds=range(42, 62)):
    seeds = list(seeds)
    rows = []
    for config in configs:
        maze = config['task']
        pairs = [(shares(finals[maze, ARM, seed]), shares(finals[maze, G, seed])) for seed in seeds]
        relabeled = statistics.mean(left['relabeled'] for left, _ in pairs)
        difference = statistics.mean(left['all'] - right['all'] for left, right in pairs)
        pi = probability(maze)
        rows.append(dict(maze=maze, n=len(seeds), registered=True, pi=pi,
            relabeled_skip_share=relabeled, relabeled_difference=relabeled-pi,
            all_r_skip_share=statistics.mean(left['all'] for left, _ in pairs),
            all_g_skip_share=statistics.mean(right['all'] for _, right in pairs),
            all_difference=difference,
            limited=abs(relabeled-pi) > 0.01 or abs(difference) > 0.02))
    return rows
