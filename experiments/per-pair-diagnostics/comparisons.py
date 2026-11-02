from analysis import paired, summarize
from protocol import ARMS, analysis_registration


def registered_results(metrics, configurations):
    registration = analysis_registration()
    result = []
    for config in configurations:
        maze = config['task']
        final = config['targets'][-1]
        def values(arm, point, metric):
            return {seed: metric(metrics[maze, arm, seed, point, 'pooled']) for seed in range(42, 47)}
        for arm in ARMS:
            for point in (30000, 100000, final):
                left = values(arm, point, lambda item: item['F'])
                right = values(arm, point, lambda item: item['C'])
                result.append(dict(maze=maze, comparison=arm + ': F - C', metric='F-C', nominal_step=point,
                                   view='pooled', registered=True, **paired(left, right)))
        for left, right in registration['final_policy_contrasts']:
            for metric in registration['final_policy_metrics']:
                result.append(dict(maze=maze, comparison=left + ' - ' + right, metric=metric,
                    nominal_step=final, view='pooled', registered=True,
                    **paired(values(left, final, lambda item: item[metric]), values(right, final, lambda item: item[metric]))))
        for left, right in registration['final_value_contrasts']:
            result.append(dict(maze=maze, comparison=left + ' - ' + right, metric='estimate_minus_Q*',
                nominal_step=final, view='pooled', registered=True,
                **paired(values(left, final, lambda item: item['values']['all']['estimate_minus_q']),
                         values(right, final, lambda item: item['values']['all']['estimate_minus_q']))))
        for arm in ARMS:
            observed = values(arm, final, lambda item: item['values']['all']['winner_goal_share'])
            result.append(dict(maze=maze, comparison=arm, metric='winner_goal_share', nominal_step=final,
                               view='pooled', registered=True, seed_values=list(observed.values()),
                               **summarize(list(observed.values()))))
    return result
