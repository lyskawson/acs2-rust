import math
import json

from agent_names import agent_settings, canonical_run_name
from vcp_schema import validate_vcp
from replay_schema import AXES, parameters as replay_parameters, registered_axes, validate_passes

EVALUATION_FIELDS = ('success', 'random_floor', 'reachable_within_cap', 'reachable_after_cap',
                     'unreachable_or_ambiguous', 'evaluation_pairs', 'evaluation_distribution',
                     'evaluation_standard_error', 'evaluation_sample_seed', 'evaluation_sample_stream',
                     'mean_success_steps', 'mean_success_steps_over_shortest', 'value_diagnostics',
                     'starts', 'reference_distribution')


def evaluation_blocks(row, required_blocks=None):
    if row.get('schema') not in (2, 3, 4, 5):
        raise ValueError('unsupported measurement row schema')
    blocks = row.get('evaluation_blocks')
    if blocks is None:
        return {'full_pool': {key: row[key] for key in EVALUATION_FIELDS}}
    names = set(required_blocks or ('training', 'validation', 'test'))
    if row['schema'] not in (4, 5) or set(blocks) != names:
        raise ValueError('expected three separate evaluation blocks')
    if any(key in row for key in EVALUATION_FIELDS):
        raise ValueError('split row contains ambiguous top-level evaluation fields')
    ensemble = canonical_run_name(row.get('agent', '')).startswith('acs2vcp')
    expected_fields = set(EVALUATION_FIELDS) | ({'heads'} if ensemble else set())
    if ensemble and 'training_variance_measured' in row.get('replay_diagnostics', {}).get('priority', {}):
        expected_fields.add('head_disagreement')
    if any(set(block) != expected_fields for block in blocks.values()):
        raise ValueError('incomplete evaluation block')
    if ensemble and any(set(head) != set(EVALUATION_FIELDS) for block in blocks.values() for head in block['heads']):
        raise ValueError('incomplete per-head evaluation block')
    return blocks


def validate_blocks(row, config):
    blocks = evaluation_blocks(row, config.get('evaluation_blocks'))
    if row['schema'] == 5 and config.get('protocol') == 'maze-pairs-v1':
        knowledge = row.get('knowledge')
        if not isinstance(knowledge, (int, float)) or not math.isfinite(knowledge) or not 0 <= knowledge <= 1:
            raise ValueError('pair protocol requires finite knowledge in every row')
        if row.get('knowledge_transition_goal_pairs') != config['knowledge_transition_goal_pairs']:
            raise ValueError('knowledge uses a different transition-goal universe')
    if 'protocol' not in config:
        if set(blocks) != {'full_pool'}:
            raise ValueError('legacy configuration requires a full-pool row')
        return
    expected = {'id': config['protocol'], 'configuration': config['registration']}
    if row.get('goal_protocol') != expected:
        raise ValueError('goal protocol registration differs')
    for name, block in blocks.items():
        reference = config['registration']['blocks'][name]
        if block['evaluation_pairs'] != reference['evaluation_pairs']:
            raise ValueError('evaluation pair count differs from the registered block')
        if block['evaluation_distribution'] != 'exhaustive_exact_weights' or block['evaluation_standard_error'] != 0:
            raise ValueError('maze evaluation must use all pairs with exact weights')
        if block['starts'] is not None or block['evaluation_sample_seed'] is not None or block['evaluation_sample_stream'] is not None:
            raise ValueError('grid rows must be compact and exhaustive')
        for key in ('random_floor', 'reachable_within_cap'):
            if not math.isclose(block[key], reference[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f'{name}: {key} differs from its own reference')
        if not math.isclose(block['reachable_after_cap'], 1 - reference['reachable_within_cap'], abs_tol=1e-12) or block['unreachable_or_ambiguous'] != 0:
            raise ValueError('incorrect reachability decomposition')
        if not math.isfinite(block['success']) or not 0 <= block['success'] <= block['reachable_within_cap'] + 1e-12:
            raise ValueError('success is outside its block ceiling')


def validate_replay(row, config, agent):
    agent = canonical_run_name(agent)
    if agent.startswith('acs2vcp'):
        validate_vcp(row, config, agent)
        return
    restricted_full = agent in ('acs2her_nontrivial', 'acs2her_shortened')
    if agent not in ('acs2er_episodic', 'acs2her', 'acs2her_values', 'acs2her_values_shortened', 'acs2er_one_update') and not restricted_full:
        return
    updates = row['replay_updates']
    kinds = row.get('learning_update_counts', {})
    if set(kinds) != {'full', 'values_only', 'skipped_warmup_steps'} or any(type(value) is not int or value < 0 for value in kinds.values()):
        raise ValueError('invalid update kinds')
    if kinds['full'] + kinds['values_only'] != updates or kinds['skipped_warmup_steps'] != 0:
        raise ValueError('update kinds disagree with thesis warmup or replay count')
    parameters = row['agent_parameters']
    expected = dict(replay=True, replay_capacity=10000, replay_warmup=1,
                    replay_updates_per_step=1 if agent == 'acs2er_one_update' else 3)
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError('agent replay parameters differ from the registered preset')
    if agent == 'acs2er_one_update':
        if kinds['values_only'] or row['trajectory_logical_bytes'] != 0:
            raise ValueError('diagnostic arm must use unchanged core ACS2ER')
        diagnostics = row['replay_diagnostics']
        if diagnostics is None:
            if config.get('u1_draw_observer'):
                raise ValueError('missing required U1 draw observation')
            return
        if (diagnostics['draws'] != updates or diagnostics['original'] != updates or diagnostics['relabeled'] != 0
                or diagnostics['realized_relabeled_share'] != 0 or diagnostics['objective_evaluations'] != 0
                or diagnostics['routes']['original']['done'] > updates
                or diagnostics['routes']['original']['reward_sum'] != 1000 * diagnostics['routes']['original']['done']
                or diagnostics['draw_observer'] != 'core_u1_outcomes_no_extra_random_or_objective_calls'):
            raise ValueError('invalid observed U1 replay composition')
        if diagnostics['failed_draws'] or diagnostics['fallbacks'] or diagnostics['strategies'] != dict(original=updates, final=0, future=0, episode=0, uniform_real=0):
            raise ValueError('invalid observed U1 replay routes')
        for key in ('already_reached', 'after_counterfactual_end', 'outside_candidates'):
            if diagnostics[key] or any(route[key] for route in diagnostics['routes'].values()):
                raise ValueError('original U1 samples must come from active real-goal episodes')
        original = diagnostics['routes']['original']
        relabeled = diagnostics['routes']['relabeled']
        if original['draws'] != updates or diagnostics['done'] != original['done'] or not 0 <= original['done'] <= updates or original['mean_reward'] != original['reward_sum'] / updates:
            raise ValueError('invalid U1 original-route composition')
        if relabeled['draws'] or relabeled['done'] or relabeled['reward_sum'] or relabeled['mean_reward'] is not None:
            raise ValueError('U1 cannot have relabeled route observations')
        if diagnostics['reward_evaluations'] or diagnostics['reach_evaluations'] or any(any(value.values()) for value in diagnostics['objective_evaluations_by_purpose'].values()):
            raise ValueError('U1 stored replay performs no objective evaluations')
        return
    expected.update(schedule='after_complete_episode', strategy='future', candidate_filter=False,
                    draws='uniform_transitions_with_replacement', truncation='Bootstrap', sampler_truncation='Bootstrap',
                    sampler_stream=7, learning_clock='episode_end_environment_steps',
                    warmup_rule='retained_steps_after_episode_insertion_at_least_warmup_no_backpay',
                    admissibility='CounterfactualEpisode' if agent == 'acs2her_values_shortened' else 'EveryTransition',
                    copy_update='Full' if agent in ('acs2er_episodic', 'acs2her') else 'ValuesOnly',
                    relabeled_proportion=0.0 if agent == 'acs2er_episodic' else agent_settings(config, agent).get('relabeled_proportion', 2 / 3))
    settings = agent_settings(config, agent)
    axes = registered_axes(config, agent)
    expected.update(replay_parameters(config, agent))
    if agent in ('acs2her', 'acs2her_shortened') and settings:
        proportion = settings.get('relabeled_proportion')
        if (set(settings) != ({'relabeled_proportion'} | (AXES if axes else set())) or type(proportion) is not float
                or not math.isfinite(proportion) or not 0 <= proportion <= 1):
            raise ValueError('invalid full hindsight proportion settings')
    if restricted_full:
        expected.update(copy_update='Full', relabeled_proportion=settings.get('relabeled_proportion', 2 / 3),
                        admissibility='FromNonGoalState' if agent == 'acs2her_nontrivial' else 'CounterfactualEpisode')
        if (set(parameters) != set(expected) or (agent == 'acs2her_nontrivial' and set(settings) - AXES)
                or any(type(parameters[key]) is not type(value) for key, value in expected.items())):
            raise ValueError('restricted full hindsight presets have fixed parameters')
    if agent in ('acs2her', 'acs2her_shortened') and type(parameters.get('relabeled_proportion')) is not float:
        raise ValueError('full hindsight row proportion must be a binary64 number')
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError('episodic agent rules differ from the registered settings')
    if axes and set(parameters) != set(expected):
        raise ValueError('new replay settings have unregistered row parameters')
    if axes and json.dumps(parameters, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError('new replay parameter types or values differ from registration')
    validate_passes(row, config, agent)
    diagnostics = row['replay_diagnostics']
    if diagnostics['draws'] != updates or diagnostics['failed_draws'] != 0:
        raise ValueError('sampler draw count differs from learning')
    original, relabeled = diagnostics['original'], diagnostics['relabeled']
    if original + relabeled != updates or not 0 <= diagnostics['fallbacks'] <= original:
        raise ValueError('replay routes do not partition draws')
    if diagnostics['realized_relabeled_share'] != relabeled / updates:
        raise ValueError('realized relabeled share differs from route counts')
    if agent in ('acs2her_values', 'acs2her_values_shortened'):
        if kinds['values_only'] != relabeled or kinds['full'] != original:
            raise ValueError('values-only learning does not follow replay routes')
    elif kinds['values_only'] or (agent == 'acs2er_episodic' and relabeled):
        raise ValueError('full/control arm has invalid copy updates or relabeling')
    if not 0 < diagnostics['stored_steps'] <= 10000 or row['trajectory_logical_bytes'] <= 0 or row['replay_samples'] or row['replay_logical_bytes']:
        raise ValueError('episodic replay must retain raw trajectories only')
    routes = diagnostics['routes']
    for route, count in (('original', original), ('relabeled', relabeled)):
        if routes[route]['draws'] != count:
            raise ValueError('route draw count differs')
        for key in ('already_reached', 'after_counterfactual_end', 'done', 'outside_candidates'):
            if not 0 <= routes[route][key] <= count:
                raise ValueError('invalid route composition count')
        expected_mean = routes[route]['reward_sum'] / count if count else None
        if routes[route]['mean_reward'] != expected_mean:
            raise ValueError('invalid route reward mean')
    for key in ('already_reached', 'after_counterfactual_end', 'done', 'outside_candidates'):
        if diagnostics[key] != sum(route[key] for route in routes.values()):
            raise ValueError('pooled composition differs from routes')
    purposes = diagnostics['objective_evaluations_by_purpose']
    for key in ('reward_evaluations', 'reach_evaluations', 'objective_evaluations'):
        if diagnostics[key] != sum(value[key] for value in purposes.values()):
            raise ValueError('objective purposes do not sum to the total')
    if any(value['objective_evaluations'] != value['reward_evaluations'] + value['reach_evaluations'] for value in purposes.values()):
        raise ValueError('objective cost decomposition differs')
    if restricted_full:
        copies = routes['relabeled']
        if copies['already_reached'] or (agent == 'acs2her_shortened' and copies['after_counterfactual_end']):
            raise ValueError('restricted hindsight retained an inadmissible copy')
        walls = diagnostics.get('wall_bumps', {})
        if set(walls) != {'relabeled_draws', 'unchanged_state', 'unchanged_state_terminal', 'proposals_equal_draws'}:
            raise ValueError('missing restricted hindsight wall diagnostic')
        if any(type(walls[key]) is not int or walls[key] < 0 for key in ('relabeled_draws', 'unchanged_state', 'unchanged_state_terminal')):
            raise ValueError('invalid wall counters')
        if (walls['relabeled_draws'] != relabeled or walls['proposals_equal_draws'] is not True
                or not 0 <= walls['unchanged_state_terminal'] <= walls['unchanged_state'] <= relabeled
                or walls['unchanged_state_terminal'] != 0):
            raise ValueError('wall diagnostic differs from admissible relabeled draws')
