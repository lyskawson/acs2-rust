from fractions import Fraction


def intervals(rows):
    previous_steps = previous_updates = 0
    result = []
    for row in rows:
        steps, updates = row['actual_steps'], row['replay_updates']
        if steps <= previous_steps or updates < previous_updates:
            raise ValueError('checkpoint counters are not monotone')
        result.append(dict(nominal_step=row['nominal_step'], actual_steps=steps,
            interval_interactions=steps - previous_steps, interval_updates=updates - previous_updates,
            updates_per_interaction=float(Fraction(updates - previous_updates, steps - previous_steps)),
            updates_per_interaction_exact=str(Fraction(updates - previous_updates, steps - previous_steps))))
        previous_steps, previous_updates = steps, updates
    return result


def value_gaps(block):
    diagnostics = block['value_diagnostics']
    count = round(block['success'] * block['evaluation_pairs'])
    pairs = block['evaluation_pairs']
    all_value = diagnostics['all']['mean_estimate_minus_return']
    successful = diagnostics['successful']['mean_estimate_minus_return'] if count else None
    failed = (pairs * all_value - count * successful) / (pairs - count) if 0 < count < pairs else all_value if count == 0 else None
    return dict(all=all_value, successful=successful, failed=failed)


def measured(row):
    diagnostics = row['replay_diagnostics']
    copies = diagnostics['routes']['relabeled']
    draws = diagnostics['draws']
    copied = copies['draws']
    walls = diagnostics.get('wall_bumps')
    ratio = lambda value, count: value / count if count else None
    result = dict(actual_steps=row['actual_steps'], replay_updates=row['replay_updates'],
        realized_updates_per_interaction=ratio(row['replay_updates'], row['actual_steps']),
        realized_copy_share=ratio(copied, draws), fallback_share=ratio(diagnostics['fallbacks'], draws),
        trivial_copy_share=ratio(copies['already_reached'], copied), post_end_copy_share=ratio(copies['after_counterfactual_end'], copied),
        rewarded_copy_share=ratio(copies['done'], copied),
        wall_copy_share=ratio(walls['unchanged_state'], copied) if walls else None,
        pass_counts=diagnostics.get('passes'), value_gaps={name: value_gaps(block) for name, block in row['evaluation_blocks'].items()},
        knowledge=row['knowledge'], population=row['population_classifiers'], numerosity=row['population_numerosity'],
        memory_lower_bound=row['population_known_bytes_lower_bound'] + row['replay_logical_bytes'] + row['trajectory_logical_bytes'],
        wall_seconds_train=row['wall_seconds_train'], wall_seconds_eval=row['wall_seconds_eval'],
        objective_evaluations_by_purpose=diagnostics['objective_evaluations_by_purpose'])
    return result
