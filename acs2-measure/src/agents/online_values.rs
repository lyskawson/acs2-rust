use crate::runner::{GoalAgent, MeasuredEnvironment, Preset};
use acs2_agents::acs2_per_goal_reward::Agent;
use acs2_core::ports::goal::GoalLayout;
use acs2_core::representation::{classifier::Classifier, perception::Perception};
use acs2_core::support::{
    knowledge::{anticipation_fraction, Transition},
    rng::RandomSource,
};
use serde_json::{json, Value};
use std::mem::size_of;

pub struct OnlineGoalAgent<const S: usize, const G: usize, const M: usize, R: RandomSource> {
    pub agent: Agent<S, G, M, R>,
    pub preset: Preset,
    pub steps: u64,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource> GoalAgent<S, G, M>
    for OnlineGoalAgent<S, G, M, R>
{
    fn name(&self) -> &'static str {
        "acs2_per_goal_reward"
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.agent.train_episode(env, time);
        self.steps = env.measured_steps();
    }
    fn declared_policy(&self) -> &'static str {
        "greedy_change_anticipating_population"
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        self.agent.greedy_action(state, rng)
    }
    fn first_decision(&self, head: usize, state: &Perception<M>, action: usize) -> Option<Value> {
        assert_eq!(head, 0);
        let goal = GoalLayout::<S, G, M>::split(state).1;
        Some(crate::first_decision::profile_with_rewards::<S, G, M>(
            self.agent.population(),
            state,
            action,
            |reference| self.agent.rewards().entry(reference, &goal).r,
        ))
    }
    fn online_updates(&self) -> u64 {
        self.steps
    }
    fn replay_updates(&self) -> u64 {
        0
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn population_classifiers(&self) -> usize {
        self.agent.population().len()
    }
    fn population_numerosity(&self) -> u32 {
        self.agent.population().numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        self.agent.population().len() * size_of::<Classifier<M>>()
            + self.agent.rewards().logical_bytes()
    }
    fn population_mark_entries(&self) -> usize {
        self.agent
            .population()
            .classifiers()
            .iter()
            .map(|c| {
                c.mark
                    .attributes
                    .iter()
                    .map(|attribute| attribute.len())
                    .sum::<usize>()
            })
            .sum()
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        Some(anticipation_fraction(
            self.agent.population(),
            self.agent.config().theta_r,
            transitions
                .iter()
                .map(|t| Transition::new(t.p0, t.action, t.p1)),
        ))
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        let rewards = self.agent.rewards();
        let updates = rewards.updates();
        Some(
            json!({"per_goal_rewards": {"goals": rewards.goals().len(), "existing_pairs": rewards.existing_pairs(),
            "updated_entries": rewards.updated_entries(), "reward_updates": {"original": updates.original, "relabeled": updates.relabeled}}}),
        )
    }
    fn agent_parameters(&self) -> Value {
        let p = self.preset;
        json!({"replay": false, "replay_capacity": p.replay_capacity, "replay_warmup": p.replay_warmup,
            "replay_updates_per_step": p.replay_updates_per_step,
            "reward_values": "per_classifier_per_goal", "reward_goal_count": self.agent.rewards().goals().len(),
            "reward_initialization": "base_creation_values_alp_parent_inheritance", "genetic_algorithm": false,
            "restore_supported": false})
    }
}
