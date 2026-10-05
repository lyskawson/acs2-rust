use acs2_core::goal::{Goal, GoalOutcome, GoalStep};
use acs2_trajectory::{GoalEvaluator, ReplayCounters};
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

pub fn replay_diagnostics(counters: &ReplayCounters) -> Value {
    json!({
        "draws": counters.draws, "failed_draws": counters.failed_draws, "original": counters.original, "relabeled": counters.relabeled,
        "fallbacks": counters.fallbacks, "already_reached": counters.already_reached,
        "after_counterfactual_end": counters.after_counterfactual_end, "done": counters.done,
        "outside_candidates": counters.outside_candidates,
        "strategies": {"original": counters.strategies[0], "final": counters.strategies[1],
            "future": counters.strategies[2], "episode": counters.strategies[3], "uniform_real": counters.strategies[4]},
        "objective_evaluations": counters.cost.total(), "reward_evaluations": counters.cost.reward_evaluations,
        "reach_evaluations": counters.cost.reach_evaluations
    })
}
