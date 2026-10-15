import math

AGENTS = ('acs2vcp_k2', 'acs2vcp_k4', 'acs2vcp_k2_uniform', 'acs2vcp_k4_uniform')
POSITIVE_UPPER_BOUNDS = [1e-12, 1e-9, 1e-6, 1e-3, .01, .1, 1., 2., 4., 8., 16., 32.,
                        64., 128., 256., 512., 1024., 2048., 4096., 8192., 16384., 32768.,
                        65536., 131072., 250000.]


def validate_logarithmic(summary, count):
    bins = summary['positive_bin_counts']
    zeros = summary['exact_zero_count']
    if (summary['count'] != count or summary['positive_bin_upper_bounds'] != POSITIVE_UPPER_BOUNDS
            or len(bins) != len(POSITIVE_UPPER_BOUNDS)
            or any(type(value) is not int or value < 0 for value in [zeros, *bins])
            or zeros + sum(bins) != count
            or summary['exact_zero_share'] != (zeros / count if count else None)
            or summary['bin_rule'] != 'exact_zero_separate_positive_lower_exclusive_upper_inclusive'):
        raise ValueError('VCP logarithmic variance summary differs')


def validate_variance(summary, count):
    if summary['count'] != count or sum(summary['histogram']) != count:
        raise ValueError('VCP variance histogram/count differs')
    if not 0 <= summary['minimum'] <= summary['mean'] <= summary['maximum'] <= 250000:
        raise ValueError('VCP variance summary is outside the value scale')


def validate_vcp(row, config, agent):
    if agent not in AGENTS:
        raise ValueError('unregistered VCP arm')
    heads = 2 if '_k2' in agent else 4
    temperature = 0 if agent.endswith('_uniform') else config.get('agent_settings', {}).get(agent, {}).get('temperature', 1)
    if not 0 <= temperature <= 24 or (temperature == 0) != agent.endswith('_uniform'):
        raise ValueError('unregistered VCP temperature')
    parameters = row['agent_parameters']
    expected = dict(replay=True, heads=heads, temperature=temperature, prioritized=temperature > 0,
                    replay_capacity=10000, replay_warmup=1, replay_updates_per_step=3,
                    strategy='future', admissibility='EveryTransition', candidate_filter=False,
                    copy_update='Full', relabeled_proportion=2 / 3, truncation='Bootstrap',
                    sampler_truncation='Bootstrap', sampler_stream=7, acceptance_stream=12,
                    extra_head_stream_start=8, variance_maximum=250000.0, priority_epsilon=1e-6,
                    variance_maximum_definition='reward_scale_bound', importance_weights='none',
                    priority_clock='all_draws_before_any_learning_in_episode',
                    acting_head='episode_index_modulo_head_count', learning_head='acting_head')
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError('VCP rules differ from the registered preset')
    steps, updates = row['actual_steps'], row['replay_updates']
    kinds = row['learning_update_counts']
    if kinds != dict(full=updates, values_only=0, skipped_warmup_steps=0) or updates != 3 * steps:
        raise ValueError('VCP must make three unweighted full applications per joint interaction')
    diagnostics = row['replay_diagnostics']
    priority = diagnostics['priority']
    revised = 'training_variance_measured' in priority
    if config.get('vcp_diagnostics') == 'evaluation_disagreement' and not revised:
        raise ValueError('VCP row lacks the registered revised diagnostics')
    if revised and priority['training_variance_measured'] is not (temperature > 0):
        raise ValueError('VCP training variance measurement differs')
    proposals, accepted = priority['proposals'], priority['accepted_draws']
    if accepted != updates or not proposals >= accepted > 0 or diagnostics['draws'] != accepted:
        raise ValueError('VCP draw/proposal counts disagree')
    if priority['value_queries'] != (0 if revised and temperature == 0 else heads * proposals):
        raise ValueError('VCP priority queries disagree with actual proposals')
    if row['match_formations_train'] != steps + 3 * updates + priority['value_queries']:
        raise ValueError('VCP independent match-formation identity failed')
    if priority['proposals_per_draw'] != proposals / accepted or priority['acceptance_share'] != accepted / proposals:
        raise ValueError('VCP acceptance accounting differs')
    if temperature == 0 and (proposals != accepted or priority['acceptance_coins'] != 0):
        raise ValueError('uniform ensemble must accept every proposal without randomness')
    if priority['variance_maximum'] != 250000.0 or not 0 < priority['mean_acceptance_probability'] <= 1:
        raise ValueError('VCP acceptance envelope differs')
    if revised and temperature == 0:
        if any(priority[name] is not None for name in ['proposal_variance', 'accepted_variance', 'candidate_acceptance_sums']):
            raise ValueError('uniform VCP must not report an unmeasured training variance')
        if priority['avoided_value_queries'] != heads * proposals or priority['avoided_classifier_perception_tests'] < 0:
            raise ValueError('uniform VCP removed-query accounting differs')
    else:
        for name, count in [('proposal_variance', proposals), ('accepted_variance', accepted)]:
            validate_variance(priority[name], count)
            if revised:
                validate_logarithmic(priority[name]['logarithmic'], count)
    routes = diagnostics['routes']
    if diagnostics['original'] + diagnostics['relabeled'] != accepted or diagnostics['failed_draws'] or diagnostics['fallbacks']:
        raise ValueError('VCP accepted routes disagree')
    if diagnostics['realized_relabeled_share'] != diagnostics['relabeled'] / accepted:
        raise ValueError('VCP realized copy share differs')
    for name in ['original', 'relabeled']:
        route = routes[name]
        if route['draws'] != diagnostics[name] or route['reward_sum'] != 1000 * route['done']:
            raise ValueError('VCP objective reward/termination or route counts differ')
        if route['mean_reward'] != (route['reward_sum'] / route['draws'] if route['draws'] else None):
            raise ValueError('VCP route mean reward differs')
    for key in ['done', 'already_reached', 'after_counterfactual_end', 'outside_candidates']:
        if diagnostics[key] != sum(route[key] for route in routes.values()):
            raise ValueError('VCP pooled routes disagree')
    if diagnostics['proposal_routes']['draws'] != proposals:
        raise ValueError('VCP proposal routes disagree')
    for key in ['reward_evaluations', 'reach_evaluations', 'objective_evaluations']:
        if diagnostics[key] != sum(value[key] for value in diagnostics['objective_evaluations_by_purpose'].values()):
            raise ValueError('VCP objective purposes disagree')
        if diagnostics[key] != diagnostics['proposal_routes'][key]:
            raise ValueError('VCP objective work must include rejected proposals')
    ensemble = row['ensemble']
    details = ensemble['heads']
    if ensemble['head_count'] != heads or len(details) != heads:
        raise ValueError('VCP head population accounting differs')
    for key in ['population_classifiers', 'population_numerosity', 'population_logical_bytes',
                'population_mark_entries', 'population_known_bytes_lower_bound']:
        if row[key] != sum(head[key] for head in details):
            raise ValueError('VCP memory must count every head: ' + key)
    if steps != sum(head['environment_steps'] for head in details) or updates != sum(head['full'] for head in details):
        raise ValueError('VCP per-head interaction/update counts differ')
    episodes = [head['episodes'] for head in details]
    if sum(episodes) != row['episodes'] or episodes != [row['episodes'] // heads + int(head < row['episodes'] % heads) for head in range(heads)]:
        raise ValueError('VCP acting head does not rotate')
    if [head['random_stream'] for head in details] != [1, 8, 9, 10][:heads]:
        raise ValueError('VCP head streams differ')
    if not math.isclose(row['knowledge'], sum(head['knowledge'] for head in details) / heads, abs_tol=1e-12, rel_tol=0):
        raise ValueError('VCP knowledge must average every head')
    for block in row['evaluation_blocks'].values():
        if len(block['heads']) != heads or not math.isclose(block['success'], sum(head['success'] for head in block['heads']) / heads, rel_tol=0, abs_tol=1e-12):
            raise ValueError('VCP evaluation must average every head')
        if revised:
            summary = block['head_disagreement']
            count = summary['count']
            validate_variance(summary, count)
            validate_logarithmic(summary['logarithmic'], count)
            if (not 0 < count <= row['knowledge_transition_goal_pairs']
                    or summary['value_queries'] != heads * count
                    or summary['exact_zero_share'] != summary['logarithmic']['exact_zero_share']
                    or summary['variance_maximum'] != 250000
                    or summary['cost_scope'] != 'evaluation_only_training_counters_paused'
                    or summary['reference_set'] != 'non_identity_knowledge_transitions_from_evaluated_block_start_goal_pairs_equal_transition_weights'
                    or summary['quantile_rule'] != 'nearest_rank_ceil_n_times_percentile'):
                raise ValueError('VCP evaluation disagreement reference or cost differs')
            quantiles = [summary['quantiles'][name] for name in ['p25', 'p50', 'p75', 'p90', 'p99']]
            if quantiles != sorted(quantiles) or not summary['minimum'] <= quantiles[0] <= quantiles[-1] <= summary['maximum']:
                raise ValueError('VCP evaluation disagreement quantiles differ')
    if revised and ensemble['agent_and_measurement_known_bytes_lower_bound'] != ensemble['agent_known_bytes_lower_bound'] + ensemble['measurement_diagnostics_inline_bytes']:
        raise ValueError('VCP added diagnostic memory accounting differs')
