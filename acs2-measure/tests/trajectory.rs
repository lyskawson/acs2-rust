use acs2_core::acs2er::{Acs2ErAgent, ReplayConfiguration};
use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::goal::GoalLayout;
use acs2_core::perception::Perception;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::trial::{LearningAgent, TruncationMode};
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::{HandEye4, HandEye5};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::goal::taxi::passenger_goal;
use acs2_envs::maze::geometries::pyalcs::{MAZE4, MAZE6, MAZE7, MAZEF3};
use acs2_measure::runner::{
    assert_evaluation_read_only, run_goal_agent, GoalAgent, MeasuredEnvironment, Preset,
    TrainingEnvironment, AGENT_STREAM, ENVIRONMENT_STREAM, POOL_STREAM,
};
use acs2_measure::task::{BitTask, HandEyeTask, MazeTask, Task, TaxiTask};
use acs2_measure::trajectory::{replay_diagnostics, MeasuredGoalEvaluator};
use acs2_trajectory::*;
use serde_json::{json, Value};

fn maze() -> MazeTask<Coordinates, 2> {
    MazeTask::new(
        "maze4".to_owned(),
        &MAZE4,
        vec![(2, 5), (5, 5), (6, 3), (6, 4)],
        5,
        "coordinates",
    )
}
fn environment<'a, T: Task<S, G, M>, const S: usize, const G: usize, const M: usize>(
    task: &'a T,
) -> TrainingEnvironment<'a, T, S, G, M> {
    TrainingEnvironment::new(
        task,
        task.environment(ChaChaRandomSource::from_seed_and_stream(
            42,
            ENVIRONMENT_STREAM,
        )),
        ChaChaRandomSource::from_seed_and_stream(42, POOL_STREAM),
    )
}

struct RecordingEnvironment<'a, E, const S: usize, const G: usize, const M: usize> {
    env: &'a mut E,
    store: &'a mut TrajectoryStore<S, G>,
    raw: Option<Episode<S, G>>,
}
impl<E: MeasuredEnvironment<S, G, M>, const S: usize, const G: usize, const M: usize> Environment<M>
    for RecordingEnvironment<'_, E, S, G, M>
{
    fn reset(&mut self) -> Perception<M> {
        let joined = self.env.reset();
        self.raw = Some(self.store.begin_episode(self.env.episode_start().unwrap()));
        joined
    }
    fn step(&mut self, action: usize) -> StepOutcome<M> {
        let outcome = self.env.step(action);
        self.raw
            .as_mut()
            .unwrap()
            .push(action, self.env.last_transition().unwrap().step);
        if outcome.terminated || outcome.truncated {
            self.store.insert(self.raw.take().unwrap()).unwrap();
        }
        outcome
    }
}

fn replay_equivalence<T: Task<S, G, M>, const S: usize, const G: usize, const M: usize>(task: &T) {
    for truncation in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        let preset = Preset::thesis();
        let mut env = environment(task);
        let mut store = TrajectoryStore::new(10_000);
        let mut agent = Acs2ErAgent::<M, _>::new(
            preset.config(task.actions()),
            ReplayConfiguration {
                buffer_size: 10_000,
                min_samples: 1,
                samples_number: 3,
            },
            ChaChaRandomSource::from_seed_and_stream(42, AGENT_STREAM),
        )
        .with_truncation_mode(truncation);
        let policy = EpsilonGreedy {
            number_of_possible_actions: task.actions(),
            epsilon: 0.8,
        };
        for _ in 0..20 {
            let time = env.steps;
            env.begin_episode();
            agent.run_explore_trial(
                &mut RecordingEnvironment {
                    env: &mut env,
                    store: &mut store,
                    raw: None,
                },
                &policy,
                &MaxFitnessBootstrap,
                time,
            );
            env.end_episode();
        }
        let evaluator = MeasuredGoalEvaluator::<_, S, G, M>::new(&env);
        let actual: Vec<_> = store
            .episodes()
            .flat_map(|episode| {
                (0..episode.len()).map(|t| {
                    build_sample::<S, G, M>(
                        episode,
                        t,
                        &episode.start().desired,
                        &evaluator,
                        truncation,
                    )
                    .unwrap()
                    .sample
                })
            })
            .collect();
        let expected: Vec<_> = agent.replay_memory().samples().copied().collect();
        assert_eq!(actual, expected);
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.reward.to_bits(), b.reward.to_bits());
        }
        assert_eq!(actual.len(), env.steps as usize);
        for episode in store.episodes() {
            for rule in [
                Admissibility::EveryTransition,
                Admissibility::FromNonGoalState,
                Admissibility::CounterfactualEpisode,
            ] {
                let sequence = relabel_episode::<S, G, M>(
                    episode,
                    &episode.start().desired,
                    rule,
                    &evaluator,
                    truncation,
                )
                .unwrap();
                assert_eq!(sequence.samples.len(), episode.len());
                for (t, sample) in sequence.samples.iter().enumerate() {
                    assert_eq!(sample.transition, t);
                }
            }
        }
    }
}

#[test]
fn measured_original_goals_reproduce_acs2er_samples_bit_for_bit_and_in_order() {
    replay_equivalence::<_, 8, 2, 10>(&maze());
    let template = HandEye4::new(50, Box::new(ChaChaRandomSource::from_seed(0)));
    replay_equivalence::<_, 17, 2, 19>(&HandEyeTask::<4, 17>::new(
        "handeye4".to_owned(),
        50,
        template.goal_pool().to_vec(),
    ));
    replay_equivalence::<_, 3, 1, 4>(&TaxiTask::new(200, (0..4).map(passenger_goal).collect()));
}

fn research_rewards<T: Task<S, G, M>, const S: usize, const G: usize, const M: usize>(task: &T) {
    let mut env = environment(task);
    let mut store = TrajectoryStore::new(10_000);
    let mut rng = ChaChaRandomSource::from_seed_and_stream(42, AGENT_STREAM);
    let mut desired_history = Vec::new();
    for _ in 0..12 {
        env.begin_episode();
        env.reset();
        let start = env.episode_start().unwrap();
        desired_history.push(start.desired);
        let mut raw = store.begin_episode(start);
        loop {
            let action = rng.gen_range(task.actions());
            let outcome = env.step(action);
            raw.push(action, env.last_transition().unwrap().step);
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
        env.end_episode();
        store.insert(raw).unwrap();
    }
    assert!(store
        .candidates()
        .iter()
        .all(|goal| desired_history.contains(goal)));
    let evaluator = MeasuredGoalEvaluator::<_, S, G, M>::new(&env);
    for episode in store.episodes() {
        for t in 0..episode.len() {
            for strategy in [
                GoalStrategy::Original,
                GoalStrategy::Final,
                GoalStrategy::Future,
                GoalStrategy::Episode,
                GoalStrategy::UniformReal,
            ] {
                let choices = goal_distribution(
                    episode,
                    t,
                    Selection {
                        strategy,
                        admissibility: Admissibility::EveryTransition,
                        candidate_filter: false,
                    },
                    store.candidates(),
                    &evaluator,
                );
                for choice in choices.choices {
                    let scored = build_sample::<S, G, M>(
                        episode,
                        t,
                        &choice.goal,
                        &evaluator,
                        TruncationMode::Bootstrap,
                    )
                    .unwrap();
                    assert!(scored.sample.reward >= 0.0);
                    assert_eq!(
                        GoalLayout::<S, G, M>::split(&scored.sample.state).1,
                        GoalLayout::<S, G, M>::split(&scored.sample.next_state).1
                    );
                    assert_eq!(scored.sample.done, scored.outcome.terminated);
                }
            }
        }
    }
}

#[test]
fn every_research_family_and_strategy_has_non_negative_objective_rewards() {
    research_rewards::<_, 8, 2, 10>(&maze());
    for (name, geometry, cap) in [
        ("maze6", &MAZE6, 10),
        ("maze7", &MAZE7, 10),
        ("mazef3", &MAZEF3, 5),
    ] {
        let topology = acs2_envs::maze::topology::MazeTopology::new(geometry).unwrap();
        research_rewards::<_, 8, 2, 10>(&MazeTask::<Coordinates, 2>::new(
            name.to_owned(),
            geometry,
            topology.walkable_cells().to_vec(),
            cap,
            "coordinates",
        ));
    }
    let handeye4 = HandEye4::new(50, Box::new(ChaChaRandomSource::from_seed(0)));
    research_rewards::<_, 17, 2, 19>(&HandEyeTask::<4, 17>::new(
        "handeye4".to_owned(),
        50,
        handeye4.goal_pool().to_vec(),
    ));
    let handeye5 = HandEye5::new(50, Box::new(ChaChaRandomSource::from_seed(0)));
    research_rewards::<_, 26, 2, 28>(&HandEyeTask::<5, 26>::new(
        "handeye5".to_owned(),
        50,
        handeye5.goal_pool().to_vec(),
    ));
    research_rewards::<_, 3, 1, 4>(&TaxiTask::new(200, (0..4).map(passenger_goal).collect()));
    let bits = BitFlipping::<8>::new(Box::new(ChaChaRandomSource::from_seed(0)))
        .goal_pool()
        .collect();
    research_rewards::<_, 8, 8, 16>(&BitTask::<8>::new(8, bits));
}

struct RecordOnly {
    store: TrajectoryStore<8, 2>,
    policy_rng: ChaChaRandomSource,
    sampler: Sampler<ChaChaRandomSource>,
}
impl RecordOnly {
    fn new() -> Self {
        Self {
            store: TrajectoryStore::new(100),
            policy_rng: ChaChaRandomSource::from_seed_and_stream(42, AGENT_STREAM),
            sampler: Sampler::new(
                SamplerConfiguration {
                    selection: Selection {
                        strategy: GoalStrategy::Future,
                        admissibility: Admissibility::CounterfactualEpisode,
                        candidate_filter: true,
                    },
                    relabeled_proportion: 0.8,
                    truncation: TruncationMode::Bootstrap,
                },
                ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
            ),
        }
    }
    fn snapshot(&self) -> String {
        format!(
            "{:?};{:?};{:?};{:?}",
            self.store,
            self.policy_rng.capture_state(),
            self.sampler.counters(),
            self.sampler.random_source().capture_state()
        )
    }
}
impl GoalAgent<8, 2, 10> for RecordOnly {
    fn name(&self) -> &'static str {
        "trajectory_recorder"
    }
    fn train_episode<E: MeasuredEnvironment<8, 2, 10>>(&mut self, env: &mut E, _time: u64) {
        env.reset();
        let mut raw = self.store.begin_episode(env.episode_start().unwrap());
        loop {
            let action = self.policy_rng.gen_range(8);
            let outcome = env.step(action);
            raw.push(action, env.last_transition().unwrap().step);
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
        self.store.insert(raw).unwrap();
        self.sampler
            .draw::<8, 2, 10>(&self.store, 3, &MeasuredGoalEvaluator::new(env))
            .unwrap();
    }
    fn declared_policy(&self) -> &'static str {
        "uniform_random"
    }
    fn eval_action(&self, _state: &Perception<10>, rng: &mut dyn RandomSource) -> (usize, f64) {
        (rng.gen_range(8), 0.0)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        0
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn population_classifiers(&self) -> usize {
        0
    }
    fn population_numerosity(&self) -> u32 {
        0
    }
    fn population_logical_bytes(&self) -> usize {
        0
    }
    fn population_mark_entries(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.store.logical_bytes()
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        Some(replay_diagnostics(self.sampler.counters()))
    }
    fn agent_parameters(&self) -> Value {
        json!({"sampler_stream": SAMPLER_STREAM})
    }
}

#[test]
fn recorder_passes_read_only_evaluation_and_reports_optional_provenance_and_storage() {
    let task = maze();
    assert_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
        &task,
        42,
        &[50, 100],
        RecordOnly::new,
        RecordOnly::snapshot,
    );
    let mut agent = RecordOnly::new();
    let rows = run_goal_agent(&task, &mut agent, 42, &[50, 100], "test", true);
    assert_eq!(rows[1]["schema"], 3);
    assert_eq!(
        rows[1]["trajectory_logical_bytes"],
        agent.store.logical_bytes()
    );
    assert_eq!(
        rows[1]["replay_diagnostics"],
        replay_diagnostics(agent.sampler.counters())
    );
    assert!(rows[1]["replay_diagnostics"]["draws"].as_u64().unwrap() > 0);
    assert_eq!(rows[1]["online_updates"], 0);
    assert_eq!(rows[1]["replay_updates"], 0);
}
