ARM = 'acs2her_shortened_goal_rules'
FIELDS = {'draws', 'with_goal', 'without_goal', 'skipped_without_goal', 'draws_with_goal', 'action_set_rules'}


def validate_reward_updates(row, agent):
    counts = row['replay_diagnostics'].get('reward_updates')
    if agent != ARM:
        if counts is not None:
            raise ValueError('reward filtering counters require the exact goal-rule arm')
        return
    if type(counts) is not dict or set(counts) != {'original', 'relabeled'}:
        raise ValueError('missing reward update routes')
    diagnostics = row['replay_diagnostics']
    for origin, route in counts.items():
        if (type(route) is not dict or set(route) != FIELDS
                or any(type(value) is not int or value < 0 for value in route.values())):
            raise ValueError('invalid reward update counters')
        if (route['draws'] != diagnostics[origin]
                or route['action_set_rules'] != route['with_goal'] + route['without_goal'] + route['skipped_without_goal']
                or not route['draws_with_goal'] <= route['draws'] <= route['action_set_rules']
                or route['with_goal'] < route['draws_with_goal']
                or bool(route['with_goal']) != bool(route['draws_with_goal'])
                or (not route['draws'] and any(route.values()))):
            raise ValueError('reward counts do not reconcile with draws and post-ALP action sets')
        if ((origin == 'original' and route['skipped_without_goal'])
                or (origin == 'relabeled' and route['without_goal'])):
            raise ValueError('reward filter does not follow actual origin')
