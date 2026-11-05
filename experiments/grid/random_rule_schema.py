import json
from pathlib import Path

ARM = 'acs2her_shortened_random_rules'
SKIP_STREAM = 0x726577617264736b
FIELDS = {'draws', 'with_goal', 'without_goal', 'skipped_with_goal', 'skipped_without_goal', 'draws_with_goal', 'action_set_rules'}


def probability(maze):
    source = Path(__file__).resolve().parents[1] / 'reward-update-dose-control/probabilities.json'
    counts = json.loads(source.read_text())['mazes'][maze]
    value = counts['skipped_without_goal'] / (counts['with_goal'] + counts['skipped_without_goal'])
    if value.hex() != counts['binary64_hex'] or value != counts['skip_probability']:
        raise ValueError('registered binary64 dose disagrees with integer sums')
    return value


def validate_random_rewards(row):
    counts = row['replay_diagnostics'].get('reward_updates')
    if type(counts) is not dict or set(counts) != {'original', 'relabeled'}:
        raise ValueError('missing random reward-update routes')
    for origin, route in counts.items():
        if (type(route) is not dict or set(route) != FIELDS
                or any(type(value) is not int or value < 0 for value in route.values())):
            raise ValueError('invalid random reward-update counters')
        with_goal = route['with_goal'] + route['skipped_with_goal']
        if (route['draws'] != row['replay_diagnostics'][origin]
                or route['action_set_rules'] != with_goal + route['without_goal'] + route['skipped_without_goal']
                or not route['draws_with_goal'] <= route['draws'] <= route['action_set_rules']
                or with_goal < route['draws_with_goal']
                or bool(with_goal) != bool(route['draws_with_goal'])
                or (not route['draws'] and any(route.values()))
                or (origin == 'original' and (route['skipped_with_goal'] or route['skipped_without_goal']))):
            raise ValueError('random reward counts disagree with actual routes or opportunities')
