import copy
import math

from registration import ARMS, RANDOM, S1, TRIAL
from schema import validate_row as validate_ablation


ROUTES = ('original', 'hindsight', 'trajectory_utility')
COST_COUNTS = (
    'evaluations', 'insertion_evaluations', 'refresh_evaluations', 'candidates_sampled',
    'candidates_scored', 'candidate_slots_at_four', 'insertion_candidates',
    'insertion_candidate_slots_at_four', 'match_sets', 'observation_match_sets', 'value_reads',
    'trial_value_updates', 'trial_reward_entries', 'buffer_entries', 'buffer_logical_bytes',
)
CPU_FIELDS = ('cpu_seconds', 'scoring_cpu_seconds', 'insertion_cpu_seconds', 'insertion_scoring_cpu_seconds')


def nat(value):
    return type(value) is int and value >= 0


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def validate_row(row, config, arm, seed, commit, target):
    if arm not in (*ARMS, S1) or row.get('agent') != arm:
        raise ValueError('study accepts exactly the three TU arms and S+1')
    if arm == S1:
        if 'trajectory_utility' in row or 'trajectory_utility' in row['agent_parameters']:
            raise ValueError('the S+1 reference must have TU disabled')
        validate_ablation(row, config, arm, seed, commit, target)
        return
    utility = row['trajectory_utility']
    cost, diagnostics = utility['cost'], utility['diagnostics']
    declared = config['trajectory_utility']
    share = {route: 1 / 3 for route in ROUTES}
    expected = dict(measure=arm, candidates=declared['candidates'], refresh_steps=declared['refresh_steps'],
                    route_stream=10, candidate_stream=11, route_probabilities=share)
    if (expected['candidates'] not in (4, 8) or expected['refresh_steps'] not in (None, 2000, 5000)
            or row['agent_parameters']['trajectory_utility'] != expected):
        raise ValueError('unregistered utility parameters')
    replay = row['replay_diagnostics']
    draws = diagnostics['draw_requests']
    writes = diagnostics['reward_updates']
    hindsight_fallbacks, tu_fallbacks = diagnostics['hindsight_fallbacks'], diagnostics['tu_fallbacks']
    if (set(draws) != set(ROUTES) or set(writes) != set(ROUTES)
            or not all(map(nat, (*draws.values(), *writes.values(), hindsight_fallbacks, tu_fallbacks)))
            or not 0 <= hindsight_fallbacks <= draws['hindsight']
            or not 0 <= tu_fallbacks <= draws['trajectory_utility']):
        raise ValueError('invalid route diagnostics')
    total = sum(draws.values())
    hindsight = draws['hindsight'] - hindsight_fallbacks
    tu = draws['trajectory_utility'] - tu_fallbacks
    if (total != row['replay_updates'] or replay['fallbacks'] != hindsight_fallbacks + tu_fallbacks
            or replay['relabeled'] != hindsight + tu or replay['strategies']['future'] != hindsight
            or replay['strategies']['trajectory_utility'] != tu
            or diagnostics['realized_copy_shares'] != {
                'total': ratio(hindsight + tu, total), 'hindsight': ratio(hindsight, total),
                'trajectory_utility': ratio(tu, total)}
            or replay['routes']['relabeled']['outside_candidates'] > hindsight):
        raise ValueError('requested and realized replay routes disagree')
    base_writes = replay['per_goal_rewards']['reward_updates']
    if (base_writes != {'original': writes['original'], 'relabeled': writes['hindsight'] + writes['trajectory_utility']}
            or (not tu and writes['trajectory_utility']) or (not hindsight and writes['hindsight'])):
        raise ValueError('utility learning origin counters disagree')
    if (set(cost) != {*CPU_FIELDS, *COST_COUNTS} or not all(nat(cost[key]) for key in COST_COUNTS)
            or any(type(cost[key]) not in (float, int) or not math.isfinite(cost[key]) for key in CPU_FIELDS)):
        raise ValueError('invalid utility counters')
    if (not 0 < cost['cpu_seconds'] <= row['wall_seconds_train']
            or not 0 <= cost['insertion_scoring_cpu_seconds'] <= min(cost['insertion_cpu_seconds'], cost['scoring_cpu_seconds'])
            or not 0 <= cost['scoring_cpu_seconds'] <= cost['cpu_seconds']
            or not 0 < cost['insertion_cpu_seconds'] <= cost['cpu_seconds']
            or cost['scoring_cpu_seconds'] - cost['insertion_scoring_cpu_seconds']
            > cost['cpu_seconds'] - cost['insertion_cpu_seconds'] + 1e-9
            or cost['insertion_evaluations'] != row['episodes']
            or cost['evaluations'] != cost['insertion_evaluations'] + cost['refresh_evaluations']
            or cost['refresh_evaluations'] > draws['trajectory_utility']
            or expected['refresh_steps'] is None and cost['refresh_evaluations']
            or cost['buffer_entries'] != replay['stored_episodes'] or cost['buffer_logical_bytes'] <= 0
            or row['match_formations_train'] != row['actual_steps'] + 3 * total
            or not 0 <= cost['insertion_candidates'] <= cost['candidates_sampled'] <= expected['candidates'] * cost['evaluations']
            or not 0 <= cost['insertion_candidate_slots_at_four'] <= cost['candidate_slots_at_four']
            <= min(cost['candidates_sampled'], 4 * cost['evaluations'])
            or cost['insertion_candidate_slots_at_four'] > cost['insertion_candidates']):
        raise ValueError('utility work, retention or measurement separation differs')
    if arm == RANDOM:
        if any(cost[key] for key in ('candidates_scored', 'match_sets', 'observation_match_sets',
                                     'value_reads', 'scoring_cpu_seconds')):
            raise ValueError('random control evaluated values')
    elif (cost['candidates_scored'] != cost['candidates_sampled']
          or cost['match_sets'] < cost['candidates_scored'] or cost['value_reads'] < cost['candidates_scored']
          or cost['observation_match_sets'] > cost['match_sets']):
        raise ValueError('value work does not reconcile with scored candidates')
    level = diagnostics['mean_post_update_level']
    if arm == TRIAL and cost['candidates_scored']:
        if (cost['trial_value_updates'] < cost['candidates_scored'] or type(level) not in (int, float)
                or not math.isfinite(level) or level < 0):
            raise ValueError('trial updates or their level are missing')
    elif cost['trial_value_updates'] or cost['trial_reward_entries'] or level is not None:
        raise ValueError('unexpected trial updates')
    assignments = diagnostics['assignments']
    reached = diagnostics['assignments_reached_in_trajectory']
    distinct = diagnostics['distinct_assigned_goals']
    spread = diagnostics['updates_per_goal']
    if (not all(map(nat, (assignments, reached, distinct, spread['minimum'], spread['maximum'])))
            or not 0 <= reached <= assignments <= min(cost['evaluations'], cost['candidates_sampled'])
            or not int(assignments > 0) <= distinct <= min(assignments, replay['per_goal_rewards']['goals'])
            or diagnostics['assigned_goal_reached_share'] != ratio(reached, assignments)
            or not spread['minimum'] <= spread['median'] <= spread['maximum'] <= sum(writes.values())):
        raise ValueError('invalid descriptive goal statistics')
    adapted = copy.deepcopy(row)
    adapted['agent'] = S1
    del adapted['agent_parameters']['trajectory_utility']
    del adapted['trajectory_utility']
    adapted_strategies = adapted['replay_diagnostics']['strategies']
    adapted_strategies['future'] += adapted_strategies.pop('trajectory_utility')
    validate_ablation(adapted, config, S1, seed, commit, target)
