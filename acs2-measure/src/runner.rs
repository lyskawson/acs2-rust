use std::mem::size_of;
use std::time::{Duration, Instant};

use acs2_core::acs2er::{Acs2ErAgent, ReplayConfiguration, ReplaySample};
use acs2_core::action_selection::{ActionSelector, BestAction, EpsilonGreedy};
use acs2_core::agent::Agent;
use acs2_core::classifier::Classifier;
use acs2_core::config::{AlpGenVariant, Configuration};
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::goal::{Goal, GoalEnvironment, GoalLayout, GoalOutcome, GoalStep};
use acs2_core::measurement::{
    read_match_counters, start_match_counting, stop_match_counting, MatchCounters,
};
use acs2_core::perception::Perception;
use acs2_core::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::{LearningAgent, TruncationMode};
use serde_json::{json, Value};

use crate::reference::{reference, Reference};
use crate::task::{Pair, Task};

pub const AGENT_STREAM: u64 = 1;
pub const ENVIRONMENT_STREAM: u64 = 2;
pub const EVALUATION_STREAM: u64 = 3;
pub const POOL_STREAM: u64 = 4;
pub const EVALUATION_ENVIRONMENT_STREAM: u64 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentKind {
    Acs2,
    Acs2Er,
}

impl AgentKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Acs2 => "acs2",
            Self::Acs2Er => "acs2er",
        }
    }
}

#[derive(Clone, Copy)]
pub struct Preset {
    pub replay_capacity: usize,
    pub replay_warmup: usize,
    pub replay_updates_per_step: usize,
    pub truncation: TruncationMode,
}

impl Preset {
    pub fn thesis() -> Self {
        Self {
            replay_capacity: 10_000,
            replay_warmup: 1,
            replay_updates_per_step: 3,
            truncation: TruncationMode::Bootstrap,
        }
    }
    pub fn config(self, actions: usize) -> Configuration {
        Configuration {
            number_of_possible_actions: actions,
            beta: 0.05,
            gamma: 0.95,
            theta_i: 0.1,
            theta_r: 0.9,
            theta_exp: 20,
            theta_as: 20,
            theta_ga: 100,
            mu: 0.3,
            chi: 0.8,
            u_max: 100_000,
            epsilon: 0.8,
            initial_q: 0.5,
            initial_r: 0.5,
            initial_ir: 0.0,
            do_ga: false,
            do_pee: false,
            do_action_planning: false,
            do_subsumption: true,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        }
    }
    pub fn json(self, actions: usize) -> Value {
        let c = self.config(actions);
        json!({
            "number_of_possible_actions": c.number_of_possible_actions, "beta": c.beta, "gamma": c.gamma,
            "theta_i": c.theta_i, "theta_r": c.theta_r, "theta_exp": c.theta_exp, "theta_as": c.theta_as,
            "theta_ga": c.theta_ga, "mu": c.mu, "chi": c.chi, "u_max": c.u_max, "epsilon": c.epsilon,
            "initial_q": c.initial_q, "initial_r": c.initial_r, "initial_ir": c.initial_ir,
            "do_ga": c.do_ga, "do_pee": c.do_pee, "do_action_planning": c.do_action_planning,
            "do_subsumption": c.do_subsumption, "alp_gen_variant": format!("{:?}", c.alp_gen_variant),
            "truncation": format!("{:?}", self.truncation), "exploration": "EpsilonGreedy",
            "bootstrap": "MaxFitnessBootstrap", "replay_capacity": self.replay_capacity,
            "replay_warmup": self.replay_warmup, "replay_updates_per_step": self.replay_updates_per_step,
            "random_streams": {"agent": AGENT_STREAM, "environment": ENVIRONMENT_STREAM, "evaluation": EVALUATION_STREAM, "restricted_pool": POOL_STREAM, "evaluation_environment": EVALUATION_ENVIRONMENT_STREAM, "evaluation_sample": 5}
        })
    }
}

pub trait GoalAgent<const S: usize, const G: usize, const M: usize> {
    fn name(&self) -> &'static str;
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64);
    fn declared_policy(&self) -> &'static str;
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64);
    fn online_updates(&self) -> u64;
    fn replay_updates(&self) -> u64;
    fn replay_samples(&self) -> usize;
    fn population_classifiers(&self) -> usize;
    fn population_numerosity(&self) -> u32;
    fn population_logical_bytes(&self) -> usize;
    fn population_mark_entries(&self) -> usize;
    fn replay_logical_bytes(&self) -> usize;
    fn trajectory_logical_bytes(&self) -> usize {
        0
    }
    fn agent_parameters(&self) -> Value;
}

pub struct CoreAgent<A, const M: usize> {
    pub agent: A,
    pub preset: Preset,
    pub replay: bool,
    pub steps: u64,
    pub updates: u64,
}

impl<A: LearningAgent<M>, const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M>
    for CoreAgent<A, M>
{
    fn name(&self) -> &'static str {
        if self.replay {
            "acs2er"
        } else {
            "acs2"
        }
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        let selector = EpsilonGreedy {
            number_of_possible_actions: self.agent.config().number_of_possible_actions,
            epsilon: self.agent.config().epsilon,
        };
        self.agent
            .run_explore_trial(env, &selector, &MaxFitnessBootstrap, time);
        let after = env.measured_steps();
        if self.replay {
            for step in self.steps + 1..=after {
                let available = (step as usize).min(self.preset.replay_capacity);
                if available >= self.preset.replay_warmup {
                    self.updates += available.min(self.preset.replay_updates_per_step) as u64;
                }
            }
        }
        self.steps = after;
    }
    fn declared_policy(&self) -> &'static str {
        "greedy_change_anticipating_population"
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        let population = self.agent.population();
        let matched = population.form_match_set(state);
        let action = BestAction {
            number_of_possible_actions: self.agent.config().number_of_possible_actions,
        }
        .select(population, &matched, rng);
        let action_set = population.form_action_set(&matched, action);
        (
            action,
            MaxFitnessBootstrap.estimate(population, &action_set),
        )
    }
    fn online_updates(&self) -> u64 {
        if self.replay {
            0
        } else {
            self.steps
        }
    }
    fn replay_updates(&self) -> u64 {
        self.updates
    }
    fn replay_samples(&self) -> usize {
        if self.replay {
            (self.steps as usize).min(self.preset.replay_capacity)
        } else {
            0
        }
    }
    fn population_classifiers(&self) -> usize {
        self.agent.population().len()
    }
    fn population_numerosity(&self) -> u32 {
        self.agent.population().numerosity()
    }
    fn population_logical_bytes(&self) -> usize {
        self.agent.population().len() * size_of::<Classifier<M>>()
    }
    fn population_mark_entries(&self) -> usize {
        self.agent
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
    fn replay_logical_bytes(&self) -> usize {
        if self.replay {
            (self.steps as usize).min(self.preset.replay_capacity) * size_of::<ReplaySample<M>>()
        } else {
            0
        }
    }
    fn agent_parameters(&self) -> Value {
        json!({ "replay": self.replay, "replay_capacity": self.preset.replay_capacity, "replay_warmup": self.preset.replay_warmup, "replay_updates_per_step": self.preset.replay_updates_per_step })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalTransition<const S: usize, const G: usize> {
    pub step: GoalStep<S, G>,
    pub desired: Goal<G>,
    pub outcome: GoalOutcome,
}

pub trait MeasuredEnvironment<const S: usize, const G: usize, const M: usize>:
    Environment<M>
{
    fn measured_steps(&self) -> u64;
    fn desired_goal(&self) -> Option<Goal<G>>;
    fn last_transition(&self) -> Option<GoalTransition<S, G>>;
    fn relabel(&self, step: &GoalStep<S, G>, desired: &Goal<G>) -> GoalOutcome;
}

pub struct TrainingEnvironment<'a, T, const S: usize, const G: usize, const M: usize>
where
    T: Task<S, G, M>,
{
    pub task: &'a T,
    pub inner: T::Env,
    pub pool_rng: ChaChaRandomSource,
    pub steps: u64,
    pub episodes: u64,
    desired: Option<Goal<G>>,
    last_transition: Option<GoalTransition<S, G>>,
    active_call: bool,
    resets_in_call: u32,
    episode_finished: bool,
}

impl<T, const S: usize, const G: usize, const M: usize> TrainingEnvironment<'_, T, S, G, M>
where
    T: Task<S, G, M>,
{
    pub fn new(
        task: &T,
        environment: T::Env,
        pool_rng: ChaChaRandomSource,
    ) -> TrainingEnvironment<'_, T, S, G, M> {
        TrainingEnvironment {
            task,
            inner: environment,
            pool_rng,
            steps: 0,
            episodes: 0,
            desired: None,
            last_transition: None,
            active_call: false,
            resets_in_call: 0,
            episode_finished: false,
        }
    }
    pub fn begin_episode(&mut self) {
        assert!(!self.active_call);
        self.active_call = true;
        self.resets_in_call = 0;
        self.episode_finished = false;
        self.last_transition = None;
    }
    pub fn end_episode(&mut self) {
        assert!(
            self.active_call && self.resets_in_call == 1 && self.episode_finished,
            "an agent must finish exactly one episode per call"
        );
        self.active_call = false;
    }
}

impl<T, const S: usize, const G: usize, const M: usize> Environment<M>
    for TrainingEnvironment<'_, T, S, G, M>
where
    T: Task<S, G, M>,
{
    fn reset(&mut self) -> Perception<M> {
        assert!(
            self.active_call && self.resets_in_call == 0,
            "one reset per training call"
        );
        self.resets_in_call += 1;
        let start = match self.task.training_goal(&mut self.pool_rng) {
            Some(goal) => self.inner.reset_with_goal(goal),
            None => self.inner.reset(),
        };
        self.desired = Some(start.desired);
        GoalLayout::<S, G, M>::join(&start.observation, &start.desired)
    }
    fn step(&mut self, action: usize) -> StepOutcome<M> {
        assert!(
            self.active_call && self.resets_in_call == 1 && !self.episode_finished,
            "step requires an active episode"
        );
        let desired = self.desired.expect("an active episode has a desired goal");
        let step = self.inner.step(action);
        let goal_outcome = step.outcome(self.inner.objective(), &desired);
        self.steps += 1;
        self.last_transition = Some(GoalTransition {
            step,
            desired,
            outcome: goal_outcome,
        });
        let outcome = StepOutcome {
            observation: GoalLayout::<S, G, M>::join(&step.observation, &desired),
            reward: goal_outcome.reward,
            terminated: goal_outcome.terminated,
            truncated: goal_outcome.truncated,
            info: (),
        };
        if outcome.terminated || outcome.truncated {
            self.episode_finished = true;
            self.episodes += 1;
            self.desired = None;
        }
        outcome
    }
}

impl<T, const S: usize, const G: usize, const M: usize> MeasuredEnvironment<S, G, M>
    for TrainingEnvironment<'_, T, S, G, M>
where
    T: Task<S, G, M>,
{
    fn measured_steps(&self) -> u64 {
        self.steps
    }
    fn desired_goal(&self) -> Option<Goal<G>> {
        self.desired
    }
    fn last_transition(&self) -> Option<GoalTransition<S, G>> {
        self.last_transition
    }
    fn relabel(&self, step: &GoalStep<S, G>, desired: &Goal<G>) -> GoalOutcome {
        step.outcome(self.inner.objective(), desired)
    }
}

#[derive(Default)]
pub struct EvalResult {
    pub success: f64,
    pub successful_steps: Option<f64>,
    pub successful_step_ratio: Option<f64>,
    pub estimated_first_action_value: f64,
    pub discounted_return: f64,
    pub value_gap: f64,
    pub successful_estimated_first_action_value: Option<f64>,
    pub successful_discounted_return: Option<f64>,
    pub successful_value_gap: Option<f64>,
    pub starts: Option<Vec<Value>>,
}

pub fn evaluate<T, A, const S: usize, const G: usize, const M: usize>(
    task: &T,
    agent: &A,
    pairs: &[Pair<T::State, G>],
    seed: u64,
    gamma: f64,
    record_starts: bool,
) -> EvalResult
where
    T: Task<S, G, M>,
    A: GoalAgent<S, G, M>,
{
    let mut env = task.environment(ChaChaRandomSource::from_seed_and_stream(
        seed,
        EVALUATION_ENVIRONMENT_STREAM,
    ));
    let mut rng = ChaChaRandomSource::from_seed_and_stream(seed, EVALUATION_STREAM);
    let mut result = EvalResult {
        starts: record_starts.then(Vec::new),
        ..EvalResult::default()
    };
    let mut successful_steps = 0.0;
    let mut successful_step_ratio = 0.0;
    let mut successful_estimate = 0.0;
    let mut successful_return = 0.0;
    for &(start, goal, weight) in pairs {
        let initial = task.reset_at(&mut env, start, goal);
        let mut state = GoalLayout::<S, G, M>::join(&initial.observation, &goal);
        let mut discounted_return = 0.0;
        let mut discount = 1.0;
        let mut steps = 0u32;
        let mut first_action = 0usize;
        let mut estimate = 0.0;
        let success = loop {
            let (action, value) = agent.eval_action(&state, &mut rng);
            if steps == 0 {
                first_action = action;
                estimate = value;
            }
            let step = env.step(action);
            steps += 1;
            let outcome = step.outcome(env.objective(), &goal);
            discounted_return += discount * outcome.reward;
            discount *= gamma;
            if outcome.terminated || outcome.truncated {
                break outcome.terminated;
            }
            state = GoalLayout::<S, G, M>::join(&step.observation, &goal);
        };
        let distance = task.distance(start, &goal);
        result.estimated_first_action_value += weight * estimate;
        result.discounted_return += weight * discounted_return;
        if success {
            result.success += weight;
            successful_steps += weight * f64::from(steps);
            successful_estimate += weight * estimate;
            successful_return += weight * discounted_return;
            if let Some(distance) = distance {
                successful_step_ratio += weight * f64::from(steps) / f64::from(distance);
            }
        }
        if let Some(starts) = &mut result.starts {
            starts.push(json!({"start": format!("{:?}", start), "goal": format!("{:?}", goal), "weight": weight,
                "first_action": first_action, "estimated_first_action_value": estimate,
                "discounted_return": discounted_return, "success": success, "steps": steps, "shortest_distance": distance}));
        }
    }
    result.value_gap = result.estimated_first_action_value - result.discounted_return;
    if result.success > 0.0 {
        result.successful_steps = Some(successful_steps / result.success);
        result.successful_step_ratio = Some(successful_step_ratio / result.success);
        result.successful_estimated_first_action_value = Some(successful_estimate / result.success);
        result.successful_discounted_return = Some(successful_return / result.success);
        result.successful_value_gap =
            Some((successful_estimate - successful_return) / result.success);
    }
    result
}

pub struct RunOutput {
    pub rows: Vec<Value>,
    pub final_population: String,
    pub final_rng: String,
}

pub struct RunMetadata<'a> {
    pub commit: &'a str,
    pub source_state: &'a str,
    pub host: &'a str,
    pub cpu_model: &'a str,
    pub record_starts: bool,
}

#[derive(Clone, Copy)]
pub struct RunSettings {
    pub evaluate_points: bool,
    pub capture_final_state: bool,
}

impl<'a> RunMetadata<'a> {
    pub fn test(commit: &'a str) -> Self {
        Self {
            commit,
            source_state: "test",
            host: "test",
            cpu_model: "test",
            record_starts: true,
        }
    }
}

pub fn run<T, const S: usize, const G: usize, const M: usize>(
    task: &T,
    kind: AgentKind,
    seed: u64,
    targets: &[u64],
    commit: &str,
    evaluate_points: bool,
) -> RunOutput
where
    T: Task<S, G, M>,
{
    let mut rows = Vec::new();
    let (final_population, final_rng) = run_with_sink(
        task,
        kind,
        seed,
        targets,
        &RunMetadata::test(commit),
        RunSettings {
            evaluate_points,
            capture_final_state: true,
        },
        &mut |row| rows.push(row),
    )
    .expect("state capture enabled");
    RunOutput {
        rows,
        final_population,
        final_rng,
    }
}

pub fn run_with_sink<T, F, const S: usize, const G: usize, const M: usize>(
    task: &T,
    kind: AgentKind,
    seed: u64,
    targets: &[u64],
    metadata: &RunMetadata<'_>,
    settings: RunSettings,
    emit: &mut F,
) -> Option<(String, String)>
where
    T: Task<S, G, M>,
    F: FnMut(Value),
{
    let preset = Preset::thesis();
    let config = preset.config(task.actions());
    let agent_rng = ChaChaRandomSource::from_seed_and_stream(seed, AGENT_STREAM);
    match kind {
        AgentKind::Acs2 => {
            let mut agent = CoreAgent {
                agent: Agent::<M, _>::new(config, agent_rng)
                    .with_truncation_mode(preset.truncation),
                preset,
                replay: false,
                steps: 0,
                updates: 0,
            };
            run_goal_agent_with_sink(
                task,
                &mut agent,
                seed,
                targets,
                metadata,
                settings.evaluate_points,
                emit,
            );
            settings.capture_final_state.then(|| {
                let state = acs2_core::checkpoint::Checkpointed::capture(&agent.agent);
                (
                    format!("{:?}", state.population),
                    format!("{:?}", state.rng),
                )
            })
        }
        AgentKind::Acs2Er => {
            let replay = ReplayConfiguration {
                buffer_size: preset.replay_capacity,
                min_samples: preset.replay_warmup,
                samples_number: preset.replay_updates_per_step,
            };
            let mut agent = CoreAgent {
                agent: Acs2ErAgent::<M, _>::new(config, replay, agent_rng)
                    .with_truncation_mode(preset.truncation),
                preset,
                replay: true,
                steps: 0,
                updates: 0,
            };
            run_goal_agent_with_sink(
                task,
                &mut agent,
                seed,
                targets,
                metadata,
                settings.evaluate_points,
                emit,
            );
            settings.capture_final_state.then(|| {
                let state = acs2_core::checkpoint::Checkpointed::capture(&agent.agent);
                (
                    format!("{:?}", state.population),
                    format!("{:?}", state.rng),
                )
            })
        }
    }
}

pub fn run_goal_agent<T, A, const S: usize, const G: usize, const M: usize>(
    task: &T,
    agent: &mut A,
    seed: u64,
    targets: &[u64],
    commit: &str,
    evaluate_points: bool,
) -> Vec<Value>
where
    T: Task<S, G, M>,
    A: GoalAgent<S, G, M>,
{
    let mut rows = Vec::new();
    run_goal_agent_with_sink(
        task,
        agent,
        seed,
        targets,
        &RunMetadata::test(commit),
        evaluate_points,
        &mut |row| rows.push(row),
    );
    rows
}

pub fn run_goal_agent_with_sink<T, A, F, const S: usize, const G: usize, const M: usize>(
    task: &T,
    agent: &mut A,
    seed: u64,
    targets: &[u64],
    metadata: &RunMetadata<'_>,
    evaluate_points: bool,
    emit: &mut F,
) where
    T: Task<S, G, M>,
    A: GoalAgent<S, G, M>,
    F: FnMut(Value),
{
    let environment = task.environment(ChaChaRandomSource::from_seed_and_stream(
        seed,
        ENVIRONMENT_STREAM,
    ));
    let mut env = TrainingEnvironment::new(
        task,
        environment,
        ChaChaRandomSource::from_seed_and_stream(seed, POOL_STREAM),
    );
    let pairs = task.pairs();
    let refs = reference(task, &pairs);
    let plan = RunPlan {
        pairs: &pairs,
        refs,
        seed,
        targets,
        metadata,
        evaluate_points,
    };
    run_inner(task, agent, &mut env, plan, emit)
}

pub fn assert_evaluation_read_only<
    T,
    A,
    Make,
    Snapshot,
    const S: usize,
    const G: usize,
    const M: usize,
>(
    task: &T,
    seed: u64,
    targets: &[u64],
    make_agent: Make,
    snapshot: Snapshot,
) where
    T: Task<S, G, M>,
    A: GoalAgent<S, G, M>,
    Make: Fn() -> A,
    Snapshot: Fn(&A) -> String,
{
    let mut evaluated = make_agent();
    let mut control = make_agent();
    let metadata = RunMetadata::test("test");
    run_goal_agent_with_sink(
        task,
        &mut evaluated,
        seed,
        targets,
        &metadata,
        true,
        &mut |_| {},
    );
    run_goal_agent_with_sink(
        task,
        &mut control,
        seed,
        targets,
        &metadata,
        false,
        &mut |_| {},
    );
    assert_eq!(snapshot(&evaluated), snapshot(&control));
}

struct RunPlan<'a, State, const G: usize> {
    pairs: &'a [Pair<State, G>],
    refs: Reference,
    seed: u64,
    targets: &'a [u64],
    metadata: &'a RunMetadata<'a>,
    evaluate_points: bool,
}

struct MatchCountGuard;

impl MatchCountGuard {
    fn new() -> Self {
        start_match_counting();
        Self
    }
}

impl Drop for MatchCountGuard {
    fn drop(&mut self) {
        stop_match_counting();
    }
}

fn run_inner<T, A, F, const S: usize, const G: usize, const M: usize>(
    task: &T,
    agent: &mut A,
    env: &mut TrainingEnvironment<'_, T, S, G, M>,
    plan: RunPlan<'_, T::State, G>,
    emit: &mut F,
) where
    T: Task<S, G, M>,
    A: GoalAgent<S, G, M>,
    F: FnMut(Value),
{
    let mut cumulative = MatchCounters::default();
    let mut train_elapsed = Duration::ZERO;
    let mut eval_elapsed = Duration::ZERO;
    let _counter = MatchCountGuard::new();
    for &target in plan.targets {
        let train_started = Instant::now();
        while env.steps < target {
            env.begin_episode();
            agent.train_episode(env, env.steps);
            env.end_episode();
        }
        train_elapsed += train_started.elapsed();
        let interval = read_match_counters().unwrap();
        cumulative.formations += interval.formations;
        cumulative.classifier_tests += interval.classifier_tests;
        stop_match_counting();
        let eval_started = Instant::now();
        let eval = if plan.evaluate_points {
            evaluate(
                task,
                agent,
                plan.pairs,
                plan.seed,
                Preset::thesis().config(task.actions()).gamma,
                plan.metadata.record_starts,
            )
        } else {
            EvalResult::default()
        };
        eval_elapsed += eval_started.elapsed();
        emit(json!({
            "schema": 2, "commit": plan.metadata.commit, "source_state": plan.metadata.source_state,
            "host": plan.metadata.host, "cpu_model": plan.metadata.cpu_model,
            "task": task.name(), "cap": task.cap(), "goal_encoding": task.encoding(), "goal_pool": task.pool_label(),
            "agent": agent.name(), "evaluated_policy": agent.declared_policy(), "seed": plan.seed, "nominal_step": target,
            "actual_steps": env.steps, "max_overshoot_exclusive": task.cap(), "episodes": env.episodes,
            "evaluation_distribution": if task.sampled_evaluation() { "fixed_seed_sample" } else { "exhaustive_exact_weights" },
            "evaluation_sample_seed": if task.sampled_evaluation() { Some(0u64) } else { None },
            "evaluation_sample_stream": if task.sampled_evaluation() { Some(5u64) } else { None },
            "evaluation_pairs": plan.pairs.len(),
            "evaluation_standard_error": if task.sampled_evaluation() { (eval.success * (1.0 - eval.success) / plan.pairs.len() as f64).sqrt() } else { 0.0 },
            "success": eval.success, "mean_success_steps": eval.successful_steps, "mean_success_steps_over_shortest": eval.successful_step_ratio,
            "value_diagnostics": {"all": {"mean_first_action_estimate": eval.estimated_first_action_value,
                "mean_discounted_return": eval.discounted_return, "mean_estimate_minus_return": eval.value_gap},
                "successful": if eval.success > 0.0 { Some(json!({"mean_first_action_estimate": eval.successful_estimated_first_action_value,
                    "mean_discounted_return": eval.successful_discounted_return,
                    "mean_estimate_minus_return": eval.successful_value_gap})) } else { None }},
            "starts": eval.starts, "reference_distribution": if task.sampled_evaluation() { "fixed_seed_sample" } else { "task_start_goal_distribution" },
            "random_floor": plan.refs.random_success, "reachable_within_cap": plan.refs.reachable_within_cap,
            "reachable_after_cap": plan.refs.reachable_after_cap, "unreachable_or_ambiguous": plan.refs.unreachable_or_ambiguous,
            "online_updates": agent.online_updates(), "replay_updates": agent.replay_updates(),
            "match_formations_train": cumulative.formations, "classifier_perception_tests_train": cumulative.classifier_tests,
            "population_classifiers": agent.population_classifiers(), "population_numerosity": agent.population_numerosity(),
            "population_logical_bytes": agent.population_logical_bytes(),
            "population_mark_entries": agent.population_mark_entries(),
            "population_known_bytes_lower_bound": agent.population_logical_bytes() + agent.population_mark_entries() * size_of::<Symbol>(),
            "classifier_size_bytes": size_of::<Classifier<M>>(), "replay_sample_size_bytes": size_of::<ReplaySample<M>>(),
            "replay_samples": agent.replay_samples(), "replay_logical_bytes": agent.replay_logical_bytes(),
            "trajectory_logical_bytes": agent.trajectory_logical_bytes(), "wall_seconds_train": train_elapsed.as_secs_f64(),
            "wall_seconds_eval": eval_elapsed.as_secs_f64(), "wall_seconds_total": (train_elapsed + eval_elapsed).as_secs_f64(),
            "preset": Preset::thesis().json(task.actions()), "agent_parameters": agent.agent_parameters()
        }));
        start_match_counting();
    }
}
