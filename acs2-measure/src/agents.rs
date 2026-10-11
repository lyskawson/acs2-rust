use std::mem::size_of;

use acs2_agents::replay::{EpisodicAgent, EpisodicConfiguration, ReplayEnvironment};
use acs2_core::agents::trial::LearningAgent;
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::ports::goal::{Goal, GoalOutcome, GoalStart, GoalStep};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::perception::Perception;
use acs2_core::support::knowledge::{anticipation_fraction, Transition};
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::GoalEvaluator;
use serde_json::{json, Value};

use crate::runner::{CoreAgent, GoalAgent, MeasuredEnvironment, Preset};

pub fn episodic_configuration(preset: Preset) -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity: preset.replay_capacity,
        warmup: preset.replay_warmup,
        updates_per_step: preset.replay_updates_per_step,
        truncation: preset.truncation,
    }
}

struct EpisodeAdapter<'a, E, const M: usize>(&'a mut E);

impl<E: Environment<M>, const M: usize> Environment<M> for EpisodeAdapter<'_, E, M> {
    fn reset(&mut self) -> Perception<M> {
        self.0.reset()
    }
    fn step(&mut self, action: usize) -> StepOutcome<M> {
        self.0.step(action)
    }
}

impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    GoalEvaluator<S, G> for EpisodeAdapter<'_, E, M>
{
    fn outcome(&self, step: &GoalStep<S, G>, goal: &Goal<G>) -> GoalOutcome {
        self.0.relabel(step, goal)
    }
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool {
        self.0.goal_reached(achieved, desired)
    }
}

impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize>
    ReplayEnvironment<S, G, M> for EpisodeAdapter<'_, E, M>
{
    fn episode_start(&self) -> GoalStart<S, G> {
        self.0
            .episode_start()
            .expect("reset supplies a raw goal start")
    }
    fn last_step(&self) -> GoalStep<S, G> {
        self.0
            .last_transition()
            .expect("step supplies a raw goal transition")
            .step
    }
}

pub struct EpisodicGoalAgent<A>(pub A);

impl<A: EpisodicAgent<S, G, M>, const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M>
    for EpisodicGoalAgent<A>
{
    fn name(&self) -> &'static str {
        self.0.name()
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.0
            .learner_mut()
            .train_episode(&mut EpisodeAdapter::<_, M>(env), time);
    }
    fn declared_policy(&self) -> &'static str {
        "greedy_change_anticipating_population"
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        self.0.learner().greedy_action(state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        let counts = self.0.learner().counts();
        counts.full + counts.values_only
    }
    fn learning_update_counts(&self) -> Option<Value> {
        let counts = self.0.learner().counts();
        Some(
            json!({"full": counts.full, "values_only": counts.values_only,
            "skipped_warmup_steps": counts.skipped_warmup_steps}),
        )
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.0.learner().store().logical_bytes()
    }
    fn population_classifiers(&self) -> usize {
        self.0.learner().population().len()
    }
    fn population_numerosity(&self) -> u32 {
        self.0.learner().population().numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        self.population_classifiers() * size_of::<Classifier<M>>()
    }
    fn population_mark_entries(&self) -> usize {
        self.0
            .learner()
            .population()
            .classifiers()
            .iter()
            .map(|classifier| {
                classifier
                    .mark
                    .attributes
                    .iter()
                    .map(|attribute| attribute.len())
                    .sum::<usize>()
            })
            .sum()
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        let learner = self.0.learner();
        Some(anticipation_fraction(
            learner.population(),
            learner.config().theta_r,
            transitions
                .iter()
                .map(|transition| Transition::new(transition.p0, transition.action, transition.p1)),
        ))
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        let learner = self.0.learner();
        let counters = learner.sampler().counters();
        let mut diagnostics = crate::trajectory::replay_diagnostics(counters);
        let draws = counters.pooled().draws;
        diagnostics["realized_relabeled_share"] =
            json!((draws > 0).then(|| counters.relabeled.draws as f64 / draws as f64));
        diagnostics["stored_steps"] = json!(learner.store().len());
        diagnostics["stored_episodes"] = json!(learner.store().episodes().count());
        Some(diagnostics)
    }
    fn agent_parameters(&self) -> Value {
        let learner = self.0.learner();
        let replay = learner.replay_configuration();
        let sampler = learner.sampler().configuration();
        json!({"replay": true, "schedule": "after_complete_episode", "replay_capacity": replay.capacity,
            "replay_warmup": replay.warmup, "warmup_rule": "retained_steps_after_episode_insertion_at_least_warmup_no_backpay",
            "replay_updates_per_step": replay.updates_per_step, "draws": "uniform_transitions_with_replacement",
            "strategy": sampler.selection.strategy.name(), "admissibility": format!("{:?}", sampler.selection.admissibility),
            "candidate_filter": sampler.selection.candidate_filter, "relabeled_proportion": sampler.relabeled_proportion,
            "copy_update": format!("{:?}", learner.copy_update()), "truncation": format!("{:?}", replay.truncation),
            "sampler_truncation": format!("{:?}", sampler.truncation), "sampler_stream": acs2_trajectory::SAMPLER_STREAM,
            "learning_clock": "episode_end_environment_steps"})
    }
}

pub struct DiagnosticAdapter<A, const M: usize>(pub CoreAgent<A, M>);

impl<A: LearningAgent<M>, const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M>
    for DiagnosticAdapter<A, M>
{
    fn name(&self) -> &'static str {
        "acs2er_u1"
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.0.train_episode(env, time)
    }
    fn declared_policy(&self) -> &'static str {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::declared_policy(&self.0)
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::eval_action(&self.0, state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        self.0.updates
    }
    fn learning_update_counts(&self) -> Option<Value> {
        Some(
            json!({"full": self.0.updates, "values_only": 0, "skipped_warmup_steps": self.0.steps - self.0.updates}),
        )
    }
    fn replay_samples(&self) -> usize {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::replay_samples(&self.0)
    }
    fn population_classifiers(&self) -> usize {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::population_classifiers(&self.0)
    }
    fn population_numerosity(&self) -> u32 {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::population_numerosity(&self.0)
    }
    fn population_logical_bytes(&self) -> usize {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::population_logical_bytes(&self.0)
    }
    fn population_mark_entries(&self) -> usize {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::population_mark_entries(&self.0)
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::knowledge(&self.0, transitions)
    }
    fn replay_logical_bytes(&self) -> usize {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::replay_logical_bytes(&self.0)
    }
    fn agent_parameters(&self) -> Value {
        <CoreAgent<A, M> as GoalAgent<S, G, M>>::agent_parameters(&self.0)
    }
}
