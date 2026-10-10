use acs2_core::goal::{Goal, GoalOutcome, GoalStep};
use acs2_trajectory::{GoalEvaluator, ObjectiveCost, ReplayCounters, RouteCounters};
use serde_json::{json, Value};

use crate::runner::MeasuredEnvironment;

pub struct MeasuredGoalEvaluator<'a, E, const S: usize, const G: usize, const M: usize> {
    environment: &'a E,
}

impl<'a, E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    MeasuredGoalEvaluator<'a, E, S, G, M>
{
    pub fn new(environment: &'a E) -> Self {
        Self { environment }
    }
}

impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    GoalEvaluator<S, G> for MeasuredGoalEvaluator<'_, E, S, G, M>
{
    fn outcome(&self, step: &GoalStep<S, G>, desired: &Goal<G>) -> GoalOutcome {
        self.environment.relabel(step, desired)
    }
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool {
        self.environment.goal_reached(achieved, desired)
    }
}

fn cost_json(cost: ObjectiveCost) -> Value {
    json!({
        "objective_evaluations": cost.total(), "reward_evaluations": cost.reward_evaluations,
        "reach_evaluations": cost.reach_evaluations
    })
}

fn route_json(route: &RouteCounters) -> Value {
    json!({
        "draws": route.draws, "already_reached": route.already_reached,
        "after_counterfactual_end": route.after_counterfactual_end, "done": route.done,
        "outside_candidates": route.outside_candidates, "reward_sum": route.reward,
        "mean_reward": route.given_route().map(|means| means.reward)
    })
}

pub fn replay_diagnostics(counters: &ReplayCounters) -> Value {
    let pooled = counters.pooled();
    let total = counters.cost.total();
    json!({
        "draws": pooled.draws, "failed_draws": counters.failed_draws, "original": counters.original.draws,
        "relabeled": counters.relabeled.draws, "fallbacks": counters.fallbacks, "already_reached": pooled.already_reached,
        "after_counterfactual_end": pooled.after_counterfactual_end, "done": pooled.done,
        "outside_candidates": pooled.outside_candidates,
        "strategies": {"original": counters.strategies[0], "final": counters.strategies[1],
            "future": counters.strategies[2], "episode": counters.strategies[3], "uniform_real": counters.strategies[4]},
        "objective_evaluations": total.total(), "reward_evaluations": total.reward_evaluations,
        "reach_evaluations": total.reach_evaluations,
        "objective_evaluations_by_purpose": {"scoring": cost_json(counters.cost.scoring),
            "selection": cost_json(counters.cost.selection), "provenance": cost_json(counters.cost.provenance)},
        "routes": {"original": route_json(&counters.original), "relabeled": route_json(&counters.relabeled)}
    })
}
