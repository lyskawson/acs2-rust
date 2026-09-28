use acs2_core::environment::Environment;
use acs2_core::measurement::{read_match_counters, start_match_counting, stop_match_counting};
use acs2_core::perception::Perception;
use acs2_core::population::Population;
use acs2_core::symbol::Symbol;
use acs2_envs::goal::hand_eye::{position_goal, HandEye4};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::goal::taxi::passenger_goal;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_envs::roles::ResearchTask;
use acs2_measure::reference::reference;
use acs2_measure::runner::{
    run, run_goal_agent, AgentKind, GoalAgent, MeasuredEnvironment, Preset, TrainingEnvironment,
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
    left.as_object_mut()
        .unwrap()
        .remove("wall_seconds_train_and_eval");
    right
        .as_object_mut()
        .unwrap()
        .remove("wall_seconds_train_and_eval");
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
    let task = BitTask::<16>::new(16, template.goal_pool().collect());
    let pairs = <BitTask<16> as Task<16, 16, 32>>::pairs(&task);
    assert_eq!(pairs.len(), 8_192);
    assert!(<BitTask<16> as Task<16, 16, 32>>::sampled_evaluation(&task));
    let refs = reference::<_, 16, 16, 32>(&task, &pairs);
    assert!((refs.random_success - 0.000_227_857_554_436_878_88).abs() < 1e-15);
    assert_eq!(refs.reachable_within_cap, 1.0);
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
        let goal = measured.inner.desired().unwrap();
        assert!(pool.contains(&goal));
        assert_ne!(
            measured.inner.environment().state().block,
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

impl GoalAgent<10> for LastActionPolicy {
    fn name(&self) -> &'static str {
        "last_action"
    }
    fn train_episode<E: MeasuredEnvironment<10>>(&mut self, env: &mut E, _time: u64) {
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
