use acs2_agents::acs2_per_goal_reward::Agent;
use acs2_core::ports::goal::Goal;
use acs2_core::representation::{classifier::Classifier, population::Population, symbol::Symbol};
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_envs::roles::ResearchTask;
use acs2_measure::agents::online_values::OnlineGoalAgent;
use acs2_measure::runner::{self, AgentKind, CoreAgent, GoalAgent, Preset};
use acs2_measure::task::MazeTask;

#[test]
fn one_goal_rows_and_per_pair_records_equal_the_base_through_the_new_online_driver() {
    let task =
        MazeTask::<Coordinates, 2>::new("one-goal".into(), &MAZE4, vec![(1, 1)], 5, "coordinates");
    let base = runner::run::<_, 8, 2, 10>(
        &task,
        AgentKind::Acs2,
        42,
        &[10, 50, 100, 300],
        "test",
        true,
    );
    let actual = runner::run::<_, 8, 2, 10>(
        &task,
        AgentKind::Acs2PerGoalReward,
        42,
        &[10, 50, 100, 300],
        "test",
        true,
    );
    for (mut a, mut b) in actual.rows.into_iter().zip(base.rows) {
        assert!(a["evaluation_blocks"]["full_pool"]["starts"].is_array() || a["starts"].is_array());
        for field in [
            "agent",
            "agent_parameters",
            "replay_diagnostics",
            "population_logical_bytes",
            "population_known_bytes_lower_bound",
            "wall_seconds_train",
            "wall_seconds_eval",
            "wall_seconds_total",
        ] {
            a.as_object_mut().unwrap().remove(field);
            b.as_object_mut().unwrap().remove(field);
        }
        for (key, value) in a.as_object().unwrap() {
            assert_eq!(value, &b[key], "field {key}");
        }
    }
    let preset = Preset::thesis();
    let config = preset.config(8);
    let goal = Goal::new([Symbol::Token(1), Symbol::Token(1)]);
    let mut selected = Classifier::<10>::general(Some(0), &config);
    selected.q = 1.0;
    selected.r = 10.0;
    selected.ir = 7.0;
    selected.num = 3;
    selected.effect.symbols[0] = Symbol::Token(1);
    let mut other = selected.clone();
    other.action = Some(1);
    other.num = 1;
    other.r = 20.0;
    other.ir = 13.0;
    let classifiers = vec![selected, other];
    let base = CoreAgent {
        agent: acs2_agents::acs2::Agent::with_population(
            config.clone(),
            ChaChaRandomSource::from_seed(42),
            Population::from_classifiers(classifiers.clone()),
        ),
        preset,
        replay: false,
        steps: 0,
        updates: 0,
    };
    let mut actual = Agent::<8, 2, 10, _>::new(config.clone(), &[goal], ChaChaRandomSource::from_seed(42))
        .with_population(Population::from_classifiers(classifiers));
    for (reference, r, ir) in [(0, 10.0, 7.0), (1, 20.0, 13.0)] {
        let entry = actual.rewards_mut().entry_mut(reference, &goal);
        entry.r = r;
        entry.ir = ir;
    }
    let actual = OnlineGoalAgent {
        agent: actual,
        preset,
        steps: 0,
    };
    let pairs = [((1, 2), goal, 1.0)];
    let base = runner::evaluate::<_, _, 8, 2, 10>(&task, &base, &pairs, 42, config.gamma, true);
    let actual = runner::evaluate::<_, _, 8, 2, 10>(&task, &actual, &pairs, 42, config.gamma, true);
    assert_eq!(
        base.estimated_first_action_value.to_bits(),
        10.0_f64.to_bits()
    );
    assert_eq!(
        actual.estimated_first_action_value.to_bits(),
        base.estimated_first_action_value.to_bits()
    );
    let actual_starts = actual.starts.unwrap();
    let base_starts = base.starts.unwrap();
    assert_eq!(actual_starts[0]["first_action"], 0);
    assert_eq!(actual_starts[0]["estimated_first_action_value"], 10.0);
    assert_eq!(actual_starts, base_starts);
}

#[test]
fn online_evaluation_is_read_only_including_values_and_greedy_at_epsilon_point_eight() {
    let task = MazeTask::<Coordinates, 2>::new(
        "maze4".into(),
        &MAZE4,
        vec![(1, 1), (2, 2)],
        5,
        "coordinates",
    );
    let goals = [
        Goal::new([Symbol::Token(1), Symbol::Token(1)]),
        Goal::new([Symbol::Token(2), Symbol::Token(2)]),
    ];
    let preset = Preset::thesis();
    runner::assert_evaluation_read_only(
        &task,
        42,
        &[10, 50],
        || OnlineGoalAgent {
            agent: Agent::<8, 2, 10, _>::new(
                preset.config(8),
                &goals,
                ChaChaRandomSource::from_seed_and_stream(42, 1),
            ),
            preset,
            steps: 0,
        },
        |a| a.agent.snapshot(),
    );
    let config = preset.config(8);
    assert_eq!(config.epsilon, 0.8);
    let mut c = Classifier::<10>::general(Some(0), &config);
    c.q = 1.0;
    c.effect.symbols[0] = Symbol::Token(1);
    c.r = -1e250;
    c.ir = 1e250;
    let mut d = c.clone();
    d.action = Some(1);
    d.r = 1e250;
    let mut agent = Agent::<8, 2, 10, _>::new(config, &goals, ChaChaRandomSource::from_seed(42))
        .with_population(Population::from_classifiers(vec![c, d]));
    for (g, r0, r1) in [(goals[0], 20.0, 1.0), (goals[1], 2.0, 30.0)] {
        agent.rewards_mut().entry_mut(0, &g).r = r0;
        agent.rewards_mut().entry_mut(1, &g).r = r1;
    }
    let wrapped = OnlineGoalAgent {
        agent,
        preset,
        steps: 0,
    };
    let before = wrapped.agent.snapshot();
    for seed in 42..92 {
        for (goal, action, value) in [(goals[0], 0, 20.0), (goals[1], 1, 30.0)] {
            let state = acs2_core::ports::goal::GoalLayout::<8, 2, 10>::join(
                &acs2_core::representation::perception::Perception::new([Symbol::Token(0); 8]),
                &goal,
            );
            let (a, v) = wrapped.eval_action(&state, &mut ChaChaRandomSource::from_seed(seed));
            assert_eq!(a, action);
            assert_eq!(v, value);
            assert_eq!(
                wrapped.first_decision(0, &state, a).unwrap()["winners"][0]["r"],
                value
            );
        }
    }
    assert_eq!(wrapped.agent.snapshot(), before);
}

#[test]
fn every_six_arm_driver_runs_on_every_registered_alcs_task_with_its_own_partition() {
    let arms = [
        AgentKind::Acs2,
        AgentKind::Acs2PerGoalReward,
        AgentKind::Acs2ErEpisodic,
        AgentKind::Acs2HerShortened {
            proportion: 2.0 / 3.0,
        },
        AgentKind::Acs2ErEpisodicPerGoalReward,
        AgentKind::Acs2HerShortenedPerGoalReward,
    ];
    for &(name, _) in acs2_envs::maze::geometries::research::ALCS_TASKS {
        let Some(ResearchTask::GoalMaze(g)) = ResearchTask::named(name) else {
            panic!()
        };
        let c = acs2_measure::research_pairs::configuration(name);
        let distribution = acs2_measure::research_pairs::registered::<Coordinates, 2>(
            name,
            g,
            c["cap"].as_u64().unwrap() as u32,
            "coordinates",
        );
        for arm in arms {
            let mut rows = Vec::new();
            runner::run_protocol_with_sink::<_, _, 8, 2, 10>(
                &acs2_measure::research_pairs::protocol(&distribution),
                arm,
                42,
                &[2],
                &runner::RunMetadata::test("test"),
                runner::RunSettings {
                    evaluate_points: false,
                    capture_final_state: false,
                },
                &mut |row| rows.push(row),
            );
            assert_eq!(rows.len(), 1);
            assert_eq!(rows[0]["agent"], arm.name());
            assert_eq!(
                rows[0]["goal_protocol"]["id"],
                acs2_measure::research_pairs::ID
            );
        }
    }
}
