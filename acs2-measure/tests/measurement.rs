use std::cell::Cell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Duration;

use acs2_core::checkpoint::Checkpointed;
use acs2_core::environment::Environment;
use acs2_core::goal::{Goal, GoalEnvironment, GoalLayout, GoalObjective};
use acs2_core::measurement::{read_match_counters, start_match_counting, stop_match_counting};
use acs2_core::perception::Perception;
use acs2_core::population::Population;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;
use acs2_envs::goal::hand_eye::{position_goal, HandEye4, HandEyeState};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::goal::taxi::passenger_goal;
use acs2_envs::maze::geometries::pyalcs::{MAZE4, MAZEF3};
use acs2_envs::roles::ResearchTask;
use acs2_measure::reference::reference;
use acs2_measure::runner::{
    assert_evaluation_read_only, evaluate, run, run_goal_agent, run_goal_agent_with_sink,
    AgentKind, CoreAgent, GoalAgent, MeasuredEnvironment, Preset, RunMetadata, RunSettings,
    TrainingEnvironment, ENVIRONMENT_STREAM,
};
use acs2_measure::task::{BitTask, HandEyeTask, MazeTask, Task, TaxiTask};

fn maze(pool: Option<Vec<(usize, usize)>>) -> MazeTask<Coordinates, 2> {
    let full = acs2_envs::maze::topology::MazeTopology::new(&MAZE4)
        .unwrap()
        .walkable_cells()
        .to_vec();
    MazeTask::new(
        "maze4".to_owned(),
        &MAZE4,
        pool.unwrap_or(full),
        5,
        "coordinates",
    )
}

#[test]
fn exact_goal_pool_floors_and_caps_match_independent_oracles() {
    let full = maze(None);
    let restricted = maze(Some(vec![(2, 5), (5, 5), (6, 3), (6, 4)]));
    let full_pairs = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::pairs(&full);
    let restricted_pairs = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::pairs(&restricted);
    assert_eq!(full_pairs.len(), 702);
    assert_eq!(restricted_pairs.len(), 104);
    let full_ref = reference::<_, 8, 2, 10>(&full, &full_pairs);
    let restricted_ref = reference::<_, 8, 2, 10>(&restricted, &restricted_pairs);
    assert!((full_ref.random_success - 2833.0 / 32768.0).abs() < 1e-12);
    assert!((restricted_ref.random_success - 334385.0 / 3407872.0).abs() < 1e-12);
    assert!((restricted_ref.reachable_within_cap - 1.0).abs() < 1e-12);
    assert!(full_ref.reachable_within_cap < 1.0);
    assert!((full_pairs.iter().map(|pair| pair.2).sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((restricted_pairs.iter().map(|pair| pair.2).sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn full_goal_pools_are_compact_and_restricted_pools_are_explicit() {
    let full_maze = maze(None);
    let restricted_maze = maze(Some(vec![(2, 5)]));
    assert_eq!(
        <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::pool_label(&full_maze),
        "full"
    );
    assert_ne!(
        <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::pool_label(&restricted_maze),
        "full"
    );

    let handeye_goals = HandEye4::new(5, Box::new(ChaChaRandomSource::from_seed(0)))
        .goal_pool()
        .to_vec();
    let full_handeye = HandEyeTask::<4, 17>::new("handeye4".to_owned(), 5, handeye_goals);
    assert_eq!(
        <HandEyeTask<4, 17> as Task<17, 2, 19>>::pool_label(&full_handeye),
        "full"
    );

    let full_taxi = TaxiTask::new(5, (0..4).map(passenger_goal).collect());
    assert_eq!(<TaxiTask as Task<3, 1, 4>>::pool_label(&full_taxi), "full");

    let bit_goals = acs2_envs::goal::bit_flipping::BitFlipping::<8>::new(Box::new(
        ChaChaRandomSource::from_seed(0),
    ))
    .goal_pool()
    .collect();
    let full_bit = BitTask::<8>::new(5, bit_goals);
    assert_eq!(
        <BitTask<8> as Task<8, 8, 16>>::pool_label(&full_bit),
        "full"
    );
    let restricted_bit = BitTask::<8>::new(5, vec![Goal::new([Symbol::Token(b'0'); 8])]);
    assert_ne!(
        <BitTask<8> as Task<8, 8, 16>>::pool_label(&restricted_bit),
        "full"
    );
}

#[test]
fn handeye_half_held_and_taxi_distribution_are_explicit() {
    let full = HandEye4::new(
        50,
        Box::new(acs2_core::rng::ChaChaRandomSource::from_seed(0)),
    );
    let handeye = HandEyeTask::<4, 17>::new("handeye4".to_owned(), 50, full.goal_pool().to_vec());
    let pairs = <HandEyeTask<4, 17> as Task<17, 2, 19>>::pairs(&handeye);
    assert_eq!(pairs.len(), 4080);
    assert!((pairs.iter().map(|pair| pair.2).sum::<f64>() - 1.0).abs() < 1e-12);
    let refs = reference::<_, 17, 2, 19>(&handeye, &pairs);
    assert!((refs.random_success - 0.131_961_639_209_121_46).abs() < 1e-12);
    assert!((refs.reachable_within_cap - 1.0).abs() < 1e-12);
    let restricted = HandEyeTask::<4, 17>::new(
        "handeye4".to_owned(),
        50,
        vec![position_goal((0, 0)), position_goal((3, 3))],
    );
    let restricted_pairs = <HandEyeTask<4, 17> as Task<17, 2, 19>>::pairs(&restricted);
    assert!((restricted_pairs.iter().map(|pair| pair.2).sum::<f64>() - 1.0).abs() < 1e-12);
    assert_eq!(restricted_pairs.len(), 510);
    let taxi = TaxiTask::new(
        200,
        vec![
            passenger_goal(0),
            passenger_goal(1),
            passenger_goal(2),
            passenger_goal(3),
        ],
    );
    let taxi_pairs = <TaxiTask as Task<3, 1, 4>>::pairs(&taxi);
    assert_eq!(taxi_pairs.len(), 300);
    assert!((taxi_pairs.iter().map(|pair| pair.2).sum::<f64>() - 1.0).abs() < 1e-12);
    let taxi_refs = reference::<_, 3, 1, 4>(&taxi, &taxi_pairs);
    assert!((taxi_refs.random_success - 0.046_733_164_577_065_03).abs() < 1e-12);
    assert!((taxi_refs.reachable_within_cap - 1.0).abs() < 1e-12);
}

#[test]
fn evaluation_cannot_change_training_or_the_agent_stream() {
    let task = maze(None);
    for kind in [AgentKind::Acs2, AgentKind::Acs2Er] {
        let with = run::<_, 8, 2, 10>(&task, kind, 42, &[55, 110], "test", true);
        let without = run::<_, 8, 2, 10>(&task, kind, 42, &[110], "test", false);
        assert_eq!(with.final_population, without.final_population);
        assert_eq!(with.final_rng, without.final_rng);
        assert_eq!(
            with.rows.last().unwrap()["actual_steps"],
            without.rows[0]["actual_steps"]
        );
        assert_eq!(
            with.rows.last().unwrap()["episodes"],
            without.rows[0]["episodes"]
        );
        assert_eq!(
            with.rows.last().unwrap()["match_formations_train"],
            without.rows[0]["match_formations_train"]
        );
    }
}

#[test]
fn environment_budget_policy_and_update_volume_are_recorded() {
    let task = maze(None);
    for kind in [AgentKind::Acs2, AgentKind::Acs2Er] {
        let output = run::<_, 8, 2, 10>(&task, kind, 42, &[50, 100], "test", true);
        for row in &output.rows {
            let actual = row["actual_steps"].as_u64().unwrap();
            let nominal = row["nominal_step"].as_u64().unwrap();
            assert!(actual >= nominal && actual - nominal < 5);
            assert_eq!(
                row["evaluated_policy"],
                "greedy_change_anticipating_population"
            );
            assert_eq!(row["evaluation_pairs"], 702);
            assert_eq!(row["evaluation_standard_error"], 0.0);
            assert_eq!(row["starts"].as_array().unwrap().len(), 702);
            if kind == AgentKind::Acs2Er {
                assert_eq!(row["replay_updates"], 3 * actual - 3);
                assert_eq!(row["online_updates"], 0);
            } else {
                assert_eq!(row["online_updates"], actual);
                assert_eq!(row["replay_updates"], 0);
            }
            assert!(row["population_logical_bytes"].as_u64().unwrap() > 0);
            assert!(row["match_formations_train"].as_u64().unwrap() > 0);
            assert!(row["classifier_perception_tests_train"].as_u64().unwrap() > 0);
        }
    }
}

#[test]
fn seed_and_streams_reproduce_every_non_wall_field() {
    let task = maze(None);
    let a = run::<_, 8, 2, 10>(&task, AgentKind::Acs2, 43, &[80], "test", true);
    let b = run::<_, 8, 2, 10>(&task, AgentKind::Acs2, 43, &[80], "test", true);
    let mut left = a.rows[0].clone();
    let mut right = b.rows[0].clone();
    left.as_object_mut().unwrap().remove("wall_seconds_train");
    left.as_object_mut().unwrap().remove("wall_seconds_eval");
    left.as_object_mut().unwrap().remove("wall_seconds_total");
    right.as_object_mut().unwrap().remove("wall_seconds_train");
    right.as_object_mut().unwrap().remove("wall_seconds_eval");
    right.as_object_mut().unwrap().remove("wall_seconds_total");
    assert_eq!(left, right);
    assert_eq!(a.final_population, b.final_population);
    assert_eq!(a.final_rng, b.final_rng);
}

#[test]
fn match_formation_counts_population_tests_without_touching_learning() {
    let population = Population::<2>::new();
    let state = Perception::new([Symbol::Token(b'0'), Symbol::Token(b'1')]);
    start_match_counting();
    population.form_match_set(&state);
    population.form_match_set(&state);
    assert_eq!(read_match_counters().unwrap().formations, 2);
    assert_eq!(read_match_counters().unwrap().classifier_tests, 0);
    assert_eq!(stop_match_counting().unwrap().formations, 2);
    assert!(read_match_counters().is_none());
}

#[test]
fn registry_and_preset_name_all_factors() {
    assert!(matches!(
        ResearchTask::named("maze4"),
        Some(ResearchTask::GoalMaze(_))
    ));
    assert!(matches!(
        ResearchTask::named("handeye5"),
        Some(ResearchTask::HandEye(5))
    ));
    assert!(matches!(
        ResearchTask::named("taxi"),
        Some(ResearchTask::Taxi)
    ));
    assert!(matches!(
        ResearchTask::named("bitflip16"),
        Some(ResearchTask::BitFlipping(16))
    ));
    assert!(ResearchTask::named("unknown").is_none());
    let preset = Preset::thesis().json(8);
    assert_eq!(preset["epsilon"], 0.8);
    assert_eq!(preset["replay_updates_per_step"], 3);
    assert_eq!(preset["random_streams"]["evaluation"], 3);
    assert_eq!(preset["truncation"], "Bootstrap");
}

#[test]
fn large_bit_task_samples_evaluation_but_has_analytical_references() {
    let template = acs2_envs::goal::bit_flipping::BitFlipping::<16>::new(Box::new(
        acs2_core::rng::ChaChaRandomSource::from_seed(0),
    ));
    let task = BitTask::<16>::new(1, template.goal_pool().collect());
    let pairs = <BitTask<16> as Task<16, 16, 32>>::pairs(&task);
    assert_eq!(pairs.len(), 8_192);
    assert!(<BitTask<16> as Task<16, 16, 32>>::sampled_evaluation(&task));
    let refs = reference::<_, 16, 16, 32>(&task, &pairs);
    assert_eq!(
        pairs
            .iter()
            .filter(
                |pair| <BitTask<16> as Task<16, 16, 32>>::distance(&task, pair.0, &pair.1)
                    == Some(1)
            )
            .count(),
        1
    );
    assert!((refs.random_success - 1.0 / 131_072.0).abs() < 1e-15);
    assert!((refs.reachable_within_cap - 1.0 / 8_192.0).abs() < 1e-15);
    assert!((refs.reachable_after_cap - 8_191.0 / 8_192.0).abs() < 1e-12);
}

#[test]
fn a_training_call_cannot_hide_multiple_episodes() {
    let task = maze(None);
    let environment = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::environment(
        &task,
        acs2_core::rng::ChaChaRandomSource::from_seed(1),
    );
    let mut measured = TrainingEnvironment::<_, 8, 2, 10>::new(
        &task,
        environment,
        acs2_core::rng::ChaChaRandomSource::from_seed(2),
    );
    measured.begin_episode();
    measured.reset();
    let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| measured.reset()));
    assert!(attempt.is_err());
}

#[test]
fn restricted_training_uses_only_real_goals_and_conditioned_starts() {
    let pool = vec![position_goal((0, 0)), position_goal((3, 3))];
    let task = HandEyeTask::<4, 17>::new("handeye4".to_owned(), 5, pool.clone());
    let environment = <HandEyeTask<4, 17> as Task<17, 2, 19>>::environment(
        &task,
        acs2_core::rng::ChaChaRandomSource::from_seed(42),
    );
    let mut measured = TrainingEnvironment::<_, 17, 2, 19>::new(
        &task,
        environment,
        acs2_core::rng::ChaChaRandomSource::from_seed(44),
    );
    for _ in 0..30 {
        measured.begin_episode();
        measured.reset();
        let goal = measured.desired_goal().unwrap();
        assert!(pool.contains(&goal));
        assert_ne!(
            measured.inner.state().block,
            match goal.symbols {
                [Symbol::Token(x), Symbol::Token(y)] => (x as usize, y as usize),
                _ => unreachable!(),
            }
        );
        loop {
            let step = measured.step(0);
            if step.terminated || step.truncated {
                break;
            }
        }
        measured.end_episode();
    }
    assert_eq!(measured.episodes, 30);
    assert!(std::panic::catch_unwind(|| HandEyeTask::<4, 17>::new(
        "bad".to_owned(),
        5,
        vec![pool[0], pool[0]]
    ))
    .is_err());
}

struct LastActionPolicy;

impl GoalAgent<8, 2, 10> for LastActionPolicy {
    fn name(&self) -> &'static str {
        "last_action"
    }
    fn train_episode<E: MeasuredEnvironment<8, 2, 10>>(&mut self, env: &mut E, _time: u64) {
        env.reset();
        loop {
            let outcome = env.step(0);
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
    }
    fn declared_policy(&self) -> &'static str {
        "always_action_7"
    }
    fn eval_action(
        &self,
        _state: &Perception<10>,
        _rng: &mut dyn acs2_core::rng::RandomSource,
    ) -> (usize, f64) {
        (7, 17.0)
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
    fn agent_parameters(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[test]
fn evaluation_uses_the_policy_the_agent_declares() {
    let task = maze(None);
    let rows =
        run_goal_agent::<_, _, 8, 2, 10>(&task, &mut LastActionPolicy, 42, &[6], "test", true);
    assert_eq!(rows[0]["agent"], "last_action");
    assert_eq!(rows[0]["evaluated_policy"], "always_action_7");
    for start in rows[0]["starts"].as_array().unwrap() {
        assert_eq!(start["first_action"], 7);
        assert_eq!(start["estimated_first_action_value"], 17.0);
    }
}

struct ProbeAgent<const G: usize, const M: usize> {
    steps_seen: u64,
    eval_calls: Cell<u64>,
    mutate_on_eval: bool,
    train_delay: Duration,
    eval_delay: Duration,
    panic_at_step: Option<u64>,
    first_action: usize,
    handeye_policy: bool,
    relabel: Option<(Goal<G>, Goal<G>, Goal<G>, bool, f64)>,
    relabel_checked: bool,
}

impl<const G: usize, const M: usize> ProbeAgent<G, M> {
    fn new() -> Self {
        Self {
            steps_seen: 0,
            eval_calls: Cell::new(0),
            mutate_on_eval: false,
            train_delay: Duration::ZERO,
            eval_delay: Duration::ZERO,
            panic_at_step: None,
            first_action: 0,
            handeye_policy: false,
            relabel: None,
            relabel_checked: false,
        }
    }
}

impl<const S: usize, const G: usize, const M: usize> GoalAgent<S, G, M> for ProbeAgent<G, M> {
    fn name(&self) -> &'static str {
        "probe"
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, _time: u64) {
        env.reset();
        if self.train_delay > Duration::ZERO {
            std::thread::sleep(self.train_delay);
        }
        let mut first = true;
        let mut first_transition = None;
        loop {
            let outcome = env.step(if first { self.first_action } else { 0 });
            self.steps_seen += 1;
            if self.panic_at_step == Some(self.steps_seen) {
                panic!("probe interruption");
            }
            if first {
                if let Some((achieved, original, relabeled_goal, reached, reward)) = self.relabel {
                    let transition = env.last_transition().expect("step transition");
                    assert_eq!(transition.step.achieved, achieved);
                    assert_eq!(transition.desired, original);
                    assert_eq!(transition.outcome.terminated, outcome.terminated);
                    assert_eq!(transition.outcome.truncated, outcome.truncated);
                    assert_eq!(transition.outcome.reward, outcome.reward);
                    assert_eq!(transition.outcome.reward > 0.0, achieved == original);
                    first_transition = Some((transition, relabeled_goal, reached, reward));
                }
            }
            first = false;
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
        if let Some((transition, desired, reached, reward)) = first_transition {
            assert_eq!(
                env.relabel(&transition.step, &transition.desired),
                transition.outcome
            );
            let relabeled = env.relabel(&transition.step, &desired);
            assert_eq!(relabeled.reward, reward);
            assert_eq!(relabeled.terminated, reached);
            assert_eq!(
                relabeled.truncated,
                transition.step.time_limit_reached && !reached
            );
            self.relabel_checked = true;
        }
    }
    fn declared_policy(&self) -> &'static str {
        "probe_action_zero"
    }
    fn eval_action(
        &self,
        state: &Perception<M>,
        _rng: &mut dyn acs2_core::rng::RandomSource,
    ) -> (usize, f64) {
        if self.mutate_on_eval {
            self.eval_calls.set(self.eval_calls.get() + 1);
        }
        if self.eval_delay > Duration::ZERO {
            std::thread::sleep(self.eval_delay);
        }
        if self.handeye_policy {
            let held = state.symbols[9] == Symbol::Token(b'2');
            return (if held { 1 } else { 4 }, if held { 10.0 } else { 2.0 });
        }
        (0, 1.5)
    }
    fn online_updates(&self) -> u64 {
        self.steps_seen
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
    fn agent_parameters(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

#[test]
fn environment_steps_equal_independently_observed_agent_steps() {
    let task = maze(None);
    let mut agent = ProbeAgent::<2, 10>::new();
    agent.panic_at_step = Some(200);
    let rows = run_goal_agent(&task, &mut agent, 42, &[20, 40], "test", false);
    assert_eq!(rows[1]["actual_steps"].as_u64().unwrap(), agent.steps_seen);
    assert_eq!(
        rows[1]["online_updates"].as_u64().unwrap(),
        agent.steps_seen
    );
    assert!(rows[1]["episodes"].as_u64().unwrap() > 1);
}

#[test]
fn replay_match_formations_cross_check_update_count() {
    let task = maze(None);
    let output = run::<_, 8, 2, 10>(&task, AgentKind::Acs2Er, 42, &[50], "test", false);
    let row = &output.rows[0];
    let steps = row["actual_steps"].as_u64().unwrap();
    let replay = row["replay_updates"].as_u64().unwrap();
    let formations = row["match_formations_train"].as_u64().unwrap();
    assert_eq!(formations, steps + 3 * replay);
}

#[test]
fn completed_rows_are_flushed_before_the_next_episode_panics() {
    let task = maze(None);
    let mut agent = ProbeAgent::<2, 10>::new();
    agent.panic_at_step = Some(6);
    let path = std::env::temp_dir().join(format!("acs2-interrupted-{}.jsonl", std::process::id()));
    let mut writer = BufWriter::new(File::create(&path).unwrap());
    let metadata = RunMetadata::test("test");
    let interrupted = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_goal_agent_with_sink(
            &task,
            &mut agent,
            42,
            &[1, 100],
            &metadata,
            true,
            &mut |row| {
                writeln!(writer, "{row}").unwrap();
                writer.flush().unwrap();
            },
        );
    }));
    assert!(interrupted.is_err());
    let contents = std::fs::read_to_string(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    let rows: Vec<serde_json::Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["nominal_step"], 1);
    assert!(read_match_counters().is_none());
}

#[test]
fn unequal_handeye_start_weights_change_value_aggregates() {
    let goal = position_goal((1, 0));
    let task = HandEyeTask::<3, 10>::new("handeye3".to_owned(), 2, vec![goal]);
    let pairs = <HandEyeTask<3, 10> as Task<10, 2, 12>>::pairs(&task);
    let pair = |held| {
        pairs
            .iter()
            .copied()
            .find(|(state, desired, _)| {
                *state
                    == HandEyeState {
                        gripper: (0, 0),
                        block: (0, 0),
                        held,
                    }
                    && *desired == goal
            })
            .unwrap()
    };
    let held = pair(true);
    let unheld = pair(false);
    assert_eq!(held.2 / unheld.2, 9.0);
    let total = held.2 + unheld.2;
    let selected = [
        (held.0, held.1, held.2 / total),
        (unheld.0, unheld.1, unheld.2 / total),
    ];
    let mut agent = ProbeAgent::<2, 12>::new();
    agent.handeye_policy = true;
    let result = evaluate::<_, _, 10, 2, 12>(&task, &agent, &selected, 42, 0.95, true);
    let expected_estimate = selected[0].2 * 10.0 + selected[1].2 * 2.0;
    let expected_return = selected[0].2 * 1000.0 + selected[1].2 * 950.0;
    assert!((result.success - 1.0).abs() < 1e-12);
    assert!((result.estimated_first_action_value - expected_estimate).abs() < 1e-12);
    assert!((result.discounted_return - expected_return).abs() < 1e-9);
    assert!((result.value_gap - (expected_estimate - expected_return)).abs() < 1e-9);
    assert!(
        (result.successful_estimated_first_action_value.unwrap() - expected_estimate).abs() < 1e-12
    );
    assert!((result.successful_discounted_return.unwrap() - expected_return).abs() < 1e-9);
    assert!(
        (result.successful_value_gap.unwrap() - (expected_estimate - expected_return)).abs() < 1e-9
    );
    assert_eq!(result.starts.unwrap().len(), 2);
}

#[test]
fn value_diagnostics_are_weighted_and_optional_per_start() {
    let task = maze(None);
    let output = run::<_, 8, 2, 10>(&task, AgentKind::Acs2, 42, &[20], "test", true);
    let row = &output.rows[0];
    let starts = row["starts"].as_array().unwrap();
    let estimate: f64 = starts
        .iter()
        .map(|start| {
            start["weight"].as_f64().unwrap()
                * start["estimated_first_action_value"].as_f64().unwrap()
        })
        .sum();
    let realized: f64 = starts
        .iter()
        .map(|start| {
            start["weight"].as_f64().unwrap() * start["discounted_return"].as_f64().unwrap()
        })
        .sum();
    assert!(
        (row["value_diagnostics"]["all"]["mean_first_action_estimate"]
            .as_f64()
            .unwrap()
            - estimate)
            .abs()
            < 1e-9
    );
    assert!(
        (row["value_diagnostics"]["all"]["mean_discounted_return"]
            .as_f64()
            .unwrap()
            - realized)
            .abs()
            < 1e-9
    );
    assert!(
        (row["value_diagnostics"]["all"]["mean_estimate_minus_return"]
            .as_f64()
            .unwrap()
            - (estimate - realized))
            .abs()
            < 1e-9
    );
    let success = row["success"].as_f64().unwrap();
    let successful_estimate: f64 = starts
        .iter()
        .filter(|start| start["success"] == true)
        .map(|start| {
            start["weight"].as_f64().unwrap()
                * start["estimated_first_action_value"].as_f64().unwrap()
        })
        .sum::<f64>()
        / success;
    assert!(
        (row["value_diagnostics"]["successful"]["mean_first_action_estimate"]
            .as_f64()
            .unwrap()
            - successful_estimate)
            .abs()
            < 1e-9
    );
    let metadata = RunMetadata {
        record_starts: false,
        ..RunMetadata::test("test")
    };
    let mut compact = Vec::new();
    acs2_measure::runner::run_with_sink::<_, _, 8, 2, 10>(
        &task,
        AgentKind::Acs2,
        42,
        &[20],
        &metadata,
        RunSettings {
            evaluate_points: true,
            capture_final_state: false,
        },
        &mut |row| compact.push(row),
    );
    assert!(compact[0]["starts"].is_null());
    assert_eq!(compact[0]["value_diagnostics"], row["value_diagnostics"]);
}

#[test]
fn training_and_evaluation_timers_measure_separate_work() {
    let task = BitTask::<1>::new(
        1,
        vec![
            Goal::new([Symbol::Token(b'0')]),
            Goal::new([Symbol::Token(b'1')]),
        ],
    );
    let mut agent = ProbeAgent::<1, 2>::new();
    agent.train_delay = Duration::from_millis(20);
    agent.eval_delay = Duration::from_millis(10);
    let rows = run_goal_agent(&task, &mut agent, 42, &[1], "test", true);
    let row = &rows[0];
    let training = row["wall_seconds_train"].as_f64().unwrap();
    let evaluation = row["wall_seconds_eval"].as_f64().unwrap();
    assert!(training >= 0.02);
    assert!(evaluation >= 0.02);
    assert!((row["wall_seconds_total"].as_f64().unwrap() - training - evaluation).abs() < 1e-9);
}

#[test]
fn unsuccessful_conditional_means_are_absent() {
    let task = TaxiTask::new(
        1,
        vec![
            passenger_goal(0),
            passenger_goal(1),
            passenger_goal(2),
            passenger_goal(3),
        ],
    );
    let mut agent = ProbeAgent::<1, 4>::new();
    let rows = run_goal_agent(&task, &mut agent, 42, &[1], "test", true);
    let row = &rows[0];
    assert_eq!(row["success"], 0.0);
    assert!(row["mean_success_steps"].is_null());
    assert!(row["mean_success_steps_over_shortest"].is_null());
    assert!(row["value_diagnostics"]["successful"].is_null());
}

#[test]
fn read_only_check_catches_interior_mutation() {
    let task = maze(None);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        assert_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
            &task,
            42,
            &[5, 10],
            || {
                let mut agent = ProbeAgent::<2, 10>::new();
                agent.mutate_on_eval = true;
                agent
            },
            |agent| format!("{}:{}", agent.steps_seen, agent.eval_calls.get()),
        );
    }));
    assert!(failure.is_err());
}

#[test]
fn core_agents_pass_reusable_read_only_check() {
    let task = maze(None);
    let preset = Preset::thesis();
    assert_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
        &task,
        42,
        &[55, 110],
        || CoreAgent {
            agent: acs2_core::agent::Agent::<10, _>::new(
                preset.config(8),
                ChaChaRandomSource::from_seed_and_stream(42, 1),
            )
            .with_truncation_mode(preset.truncation),
            preset,
            replay: false,
            steps: 0,
            updates: 0,
        },
        |agent| {
            format!(
                "{:?};{};{}",
                agent.agent.capture(),
                agent.steps,
                agent.updates
            )
        },
    );
    assert_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
        &task,
        42,
        &[55, 110],
        || CoreAgent {
            agent: acs2_core::acs2er::Acs2ErAgent::<10, _>::new(
                preset.config(8),
                acs2_core::acs2er::ReplayConfiguration {
                    buffer_size: preset.replay_capacity,
                    min_samples: preset.replay_warmup,
                    samples_number: preset.replay_updates_per_step,
                },
                ChaChaRandomSource::from_seed_and_stream(42, 1),
            )
            .with_truncation_mode(preset.truncation),
            preset,
            replay: true,
            steps: 0,
            updates: 0,
        },
        |agent| {
            format!(
                "{:?};{};{}",
                agent.agent.capture(),
                agent.steps,
                agent.updates
            )
        },
    );
}

#[test]
fn relabeling_agent_distinguishes_maze_f3_coordinate_twins() {
    let task = MazeTask::<Coordinates, 2>::new(
        "mazef3".to_owned(),
        &MAZEF3,
        vec![(3, 3)],
        5,
        "coordinates",
    );
    assert_eq!(
        task.template.topology().perception_at((3, 3)),
        task.template.topology().perception_at((1, 4))
    );
    let desired = task.template.goal_at((3, 3));
    let relabeled_goal = task.template.goal_at((1, 4));
    let twin = task.template.goal_at((3, 3));
    let run_probe = |start: (usize, usize), action: usize, achieved: Goal<2>, reached: bool| {
        let seed = (0..10_000)
            .find(|&seed| {
                let mut environment = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::environment(
                    &task,
                    ChaChaRandomSource::from_seed_and_stream(seed, ENVIRONMENT_STREAM),
                );
                environment.reset();
                environment.position() == start
            })
            .expect("reachable training start");
        let mut agent = ProbeAgent::<2, 10>::new();
        agent.first_action = action;
        agent.relabel = Some((
            achieved,
            desired,
            relabeled_goal,
            reached,
            task.template.objective().reward(&achieved, &relabeled_goal),
        ));
        let rows = run_goal_agent(&task, &mut agent, seed, &[1], "test", false);
        assert!(agent.relabel_checked);
        assert_eq!(rows[0]["actual_steps"].as_u64().unwrap(), agent.steps_seen);
    };
    run_probe((3, 2), 2, twin, false);
    let (start, action) = task
        .template
        .topology()
        .walkable_cells()
        .iter()
        .find_map(|&cell| {
            (0..8)
                .find(|&action| {
                    cell != (1, 4) && task.template.topology().next_cell(cell, action) == (1, 4)
                })
                .map(|action| (cell, action))
        })
        .expect("a predecessor of the goal");
    run_probe(start, action, relabeled_goal, true);
}

#[test]
fn episode_start_exposes_the_exact_achieved_goal_before_the_first_step() {
    let task = MazeTask::<Coordinates, 2>::new(
        "mazef3".to_owned(),
        &MAZEF3,
        vec![(1, 4)],
        5,
        "coordinates",
    );
    let seed = (0..10_000)
        .find(|&seed| {
            let mut environment = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::environment(
                &task,
                ChaChaRandomSource::from_seed_and_stream(seed, ENVIRONMENT_STREAM),
            );
            environment.reset();
            environment.position() == (3, 3)
        })
        .expect("the perception twin is a possible start");
    let environment = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::environment(
        &task,
        ChaChaRandomSource::from_seed_and_stream(seed, ENVIRONMENT_STREAM),
    );
    let mut measured = TrainingEnvironment::new(
        &task,
        environment,
        ChaChaRandomSource::from_seed_and_stream(seed, 4),
    );
    measured.begin_episode();
    assert_eq!(measured.episode_start(), None);
    let observation = measured.reset();
    let start = measured.episode_start().unwrap();
    assert_eq!(start.achieved, task.template.goal_at((3, 3)));
    assert_eq!(start.desired, task.template.goal_at((1, 4)));
    assert_ne!(start.achieved, start.desired);
    assert_eq!(
        start.observation,
        task.template.topology().perception_at((1, 4))
    );
    assert_eq!(
        observation,
        GoalLayout::<8, 2, 10>::join(&start.observation, &start.desired)
    );
}
