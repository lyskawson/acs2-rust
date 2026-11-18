use acs2_agents::acs2tuher::{
    ByRoute, Measurements, UtilityConfiguration, CANDIDATE_STREAM, ROUTE_STREAM,
};
use serde_json::{json, Value};

pub(super) fn parameters(config: UtilityConfiguration) -> Value {
    json!({
        "measure": config.measure.name(),
        "candidates": config.candidates,
        "refresh_steps": config.refresh_steps,
        "route_stream": ROUTE_STREAM,
        "candidate_stream": CANDIDATE_STREAM,
        "route_probabilities": {"original": 1.0 / 3.0, "hindsight": 1.0 / 3.0, "trajectory_utility": 1.0 / 3.0}
    })
}

fn by_route(by_route: ByRoute) -> Value {
    json!({
        "original": by_route.original,
        "hindsight": by_route.hindsight,
        "trajectory_utility": by_route.trajectory_utility
    })
}

pub(super) fn measurements(stats: Measurements) -> Value {
    let cost = stats.cost;
    let draws = stats.draw_requests;
    let total = draws.original + draws.hindsight + draws.trajectory_utility;
    let hindsight = draws.hindsight - stats.hindsight_fallbacks;
    let utility = draws.trajectory_utility - stats.tu_fallbacks;
    let share = |count| (total > 0).then(|| count as f64 / total as f64);
    json!({
        "cost": {
            "cpu_seconds": cost.cpu_nanoseconds as f64 / 1e9,
            "scoring_cpu_seconds": cost.scoring_cpu_nanoseconds as f64 / 1e9,
            "insertion_cpu_seconds": cost.insertion_cpu_nanoseconds as f64 / 1e9,
            "insertion_scoring_cpu_seconds": cost.insertion_scoring_cpu_nanoseconds as f64 / 1e9,
            "evaluations": cost.insertions + cost.refreshes,
            "insertion_evaluations": cost.insertions,
            "refresh_evaluations": cost.refreshes,
            "candidates_sampled": cost.candidates_sampled,
            "candidates_scored": cost.candidates_scored,
            "candidate_slots_at_four": cost.candidates_at_four,
            "insertion_candidates": cost.insertion_candidates,
            "insertion_candidate_slots_at_four": cost.insertion_candidates_at_four,
            "match_sets": cost.match_sets,
            "observation_match_sets": cost.observation_match_sets,
            "value_reads": cost.value_reads,
            "trial_value_updates": cost.trial_value_updates,
            "trial_reward_entries": cost.trial_reward_entries,
            "buffer_entries": stats.buffer_entries,
            "buffer_logical_bytes": stats.buffer_logical_bytes
        },
        "diagnostics": {
            "draw_requests": by_route(draws),
            "hindsight_fallbacks": stats.hindsight_fallbacks,
            "tu_fallbacks": stats.tu_fallbacks,
            "reward_updates": by_route(stats.reward_updates),
            "realized_copy_shares": {"total": share(hindsight + utility), "hindsight": share(hindsight),
                "trajectory_utility": share(utility)},
            "assignments": stats.assigned,
            "assignments_reached_in_trajectory": stats.assigned_goals_reached,
            "assigned_goal_reached_share": (stats.assigned > 0)
                .then(|| stats.assigned_goals_reached as f64 / stats.assigned as f64),
            "distinct_assigned_goals": stats.distinct_assigned_goals,
            "updates_per_goal": {"minimum": stats.minimum_updates_per_goal,
                "median": stats.median_updates_per_goal, "maximum": stats.maximum_updates_per_goal},
            "mean_post_update_level": stats.mean_post_update_level
        }
    })
}
