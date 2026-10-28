from agent_names import agent_settings, canonical_run_name

SCHEDULES = ('constant', '3to1at30000', '3to1at100000', '3to1at300000')
ORDERS = ('independent', 'backward', 'random-pass')
EPISODIC = ('acs2er_episodic', 'acs2her', 'acs2her_nontrivial', 'acs2her_shortened',
            'acs2her_values', 'acs2her_values_shortened')
AXES = {'replay_schedule', 'replay_order'}
PASS_PARAMETERS = dict(draws='length_weighted_episode_passes', strategy='episode',
    pass_goal='one_goal_per_pass_admitted_at_start_then_per_transition',
    pass_cut='backward_suffix_before_ordering_exact_episode_budget')


def registered_axes(config, agent):
    settings = agent_settings(config, agent)
    present = set(settings) & AXES
    if not present:
        return None
    if (canonical_run_name(agent) not in EPISODIC or present != AXES
            or set(settings) - AXES - {'relabeled_proportion'}):
        raise ValueError('replay axes require complete explicit episodic settings')
    schedule, order = settings['replay_schedule'], settings['replay_order']
    if (schedule not in SCHEDULES or order not in ORDERS
            or (schedule != 'constant' and order != 'independent')
            or (schedule == 'constant' and order == 'independent')):
        raise ValueError('unregistered or combined replay axes')
    if canonical_run_name(agent) == 'acs2er_episodic' and 'relabeled_proportion' in settings:
        raise ValueError('episodic ER fixes its copy share at zero')
    return schedule, order


def parameters(config, agent):
    axes = registered_axes(config, agent)
    if axes is None:
        return {}
    schedule, order = axes
    result = dict(replay_schedule=dict(kind='constant'), replay_order=order.replace('-', '_'))
    if schedule != 'constant':
        result['replay_schedule'] = dict(kind='three_to_one', switch_steps=int(schedule.split('at')[1]),
            early_rate=3, late_rate=1, clock='episode_end_environment_steps', early_inclusive=True)
    if order != 'independent':
        result.update(PASS_PARAMETERS)
        if order == 'random-pass':
            result['order_stream'] = 9
    return result


def expected_draws(row, config, agent):
    axes = registered_axes(config, agent)
    diagnostics = row.get('replay_diagnostics') or {}
    if axes is None:
        if ('draw_schedule' in diagnostics or 'passes' in diagnostics
                or AXES & set(row.get('agent_parameters', {}))):
            raise ValueError('default rows must not carry replay-axis fields')
        return 3 * row['actual_steps']
    schedule, order = axes
    if schedule == 'constant':
        if 'draw_schedule' in diagnostics:
            raise ValueError('constant replay cannot report a switched schedule')
        return 3 * row['actual_steps']
    if 'passes' in diagnostics:
        raise ValueError('independent replay cannot report episode passes')
    clock = diagnostics.get('draw_schedule', {})
    if set(clock) != {'early_steps', 'late_steps', 'first_late_episode_end'}:
        raise ValueError('missing switched-schedule counts')
    early, late, first = clock['early_steps'], clock['late_steps'], clock['first_late_episode_end']
    threshold = int(schedule.split('at')[1])
    steps, cap = row['actual_steps'], config['cap']
    if (type(early) is not int or type(late) is not int or min(early, late) < 0
            or early + late != steps or early > threshold):
        raise ValueError('invalid episode-end schedule volume')
    if steps <= threshold:
        if early != steps or late or first is not None:
            raise ValueError('schedule switched before its inclusive boundary')
    elif (type(first) is not int or not threshold - cap < early <= threshold
            or not threshold < first <= early + cap or first > steps):
        raise ValueError('schedule does not switch on a complete episode ending after T')
    return 3 * early + late


def validate_passes(row, config, agent):
    axes = registered_axes(config, agent)
    if axes is None or axes[1] == 'independent':
        return
    counts = row['replay_diagnostics'].get('passes', {})
    if (set(counts) != {'passes', 'requested_relabel', 'fallbacks', 'cut_passes', 'available_steps', 'emitted_steps'}
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError('missing or invalid episode-pass counts')
    if (not 0 < counts['passes'] <= counts['emitted_steps'] == row['replay_updates']
            or not counts['fallbacks'] <= counts['requested_relabel'] <= counts['passes']
            or counts['cut_passes'] > counts['passes']
            or counts['available_steps'] < counts['emitted_steps']
            or counts['available_steps'] > counts['passes'] * config['cap']
            or (canonical_run_name(agent) == 'acs2er_episodic' and counts['requested_relabel'])):
        raise ValueError('episode-pass counters do not reconcile')
