use acs2_agents::acs2tuher::{
    Agent, Measure, UtilityConfiguration, UtilityRandom, CANDIDATE_STREAM, ROUTE_STREAM,
};
use acs2_agents::episodic_replay::{EpisodicAgent, EpisodicReplay};
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_goal_agent_protocol_with_sink, AgentKind, GoalAgent,
    Preset, RunMetadata, AGENT_STREAM,
};
use acs2_measure::task::Task;
use serde_json::Value;

const SEED: u64 = 42;
const MEASURES: [Measure; 3] = [
    Measure::MeanValue,
    Measure::TrialUpdate,
    Measure::RandomGoal,
];

fn agent<const S: usize, const G: usize, const M: usize>(
    actions: usize,
    goals: &[Goal<G>],
    measure: Option<Measure>,
    share: f64,
) -> Agent<S, G, M, ChaChaRandomSource, ChaChaRandomSource> {
    let preset = Preset::thesis();
    Agent::new(
        preset.config(actions),
        episodic_configuration(preset),
        goals,
        ChaChaRandomSource::from_seed_and_stream(SEED, AGENT_STREAM),
        ChaChaRandomSource::from_seed_and_stream(SEED, acs2_trajectory::SAMPLER_STREAM),
        measure.map(|measure| {
            (
                UtilityConfiguration { tu_share: share, ..UtilityConfiguration::pilot(measure) },
                UtilityRandom {
                    routes: ChaChaRandomSource::from_seed_and_stream(SEED, ROUTE_STREAM),
                    candidates: ChaChaRandomSource::from_seed_and_stream(SEED, CANDIDATE_STREAM),
                },
            )
        }),
    )
}

fn without_timing(mut row: Value) -> Value {
    for key in [
        "wall_seconds_train",
        "wall_seconds_eval",
        "wall_seconds_total",
    ] {
        row.as_object_mut().unwrap().remove(key);
    }
    row
}

#[test]
fn disabled_utility_reproduces_s_plus_one_rows_pair_records_and_state() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let protocol = distributions.protocol();
    let goals = <_ as Task<8, 2, 10>>::training_goals(protocol.training).unwrap();
    let preset = Preset::thesis();
    let mut s_plus_one =
        EpisodicGoalAgent(
            acs2_agents::acs2her::per_goal_reward::Agent::<8, 2, 10, _, _>::new(
                preset.config(8),
                episodic_configuration(preset),
                goals,
                ChaChaRandomSource::from_seed_and_stream(SEED, AGENT_STREAM),
                ChaChaRandomSource::from_seed_and_stream(SEED, acs2_trajectory::SAMPLER_STREAM),
            ),
        );
    let mut disabled = EpisodicGoalAgent(agent::<8, 2, 10>(8, goals, Some(Measure::MeanValue), 0.0));
    let mut reference = Vec::new();
    let mut actual = Vec::new();
    let targets = [30, 300, 1_000];
    let metadata = RunMetadata::test("identity");
    run_goal_agent_protocol_with_sink(
        &protocol,
        &mut s_plus_one,
        SEED,
        &targets,
        &metadata,
        true,
        &mut |row| reference.push(without_timing(row).to_string()),
    );
    run_goal_agent_protocol_with_sink(
        &protocol,
        &mut disabled,
        SEED,
        &targets,
        &metadata,
        true,
        &mut |row| actual.push(without_timing(row).to_string()),
    );
    assert_eq!(actual, reference);
    assert_eq!(disabled.0.snapshot(), s_plus_one.0.learner().snapshot());
    assert!(disabled.0.trajectory_utility().is_none());
    assert!(actual[2].contains("\"winners\""));
}

#[test]
fn all_utility_presets_evaluate_read_only_on_the_registered_pair_blocks() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let protocol = distributions.protocol();
    let goals = <_ as Task<8, 2, 10>>::training_goals(protocol.training).unwrap();
    for measure in MEASURES {
        assert_protocol_evaluation_read_only(
            &protocol,
            SEED,
            &[30, 120],
            || EpisodicGoalAgent(agent::<8, 2, 10>(8, goals, Some(measure), 0.5)),
            |agent| agent.0.snapshot(),
        );
    }
}

#[test]
fn all_utility_presets_act_greedily_with_exploration_point_eight() {
    let first = Goal::new([Symbol::Token(8)]);
    let second = Goal::new([Symbol::Token(9)]);
    for (measure, share) in MEASURES.into_iter().map(|measure| (measure, 0.5)).chain(
        [Measure::MeanValue, Measure::RandomGoal].into_iter().flat_map(|measure|
            [0.25, 0.75, 1.0].into_iter().map(move |share| (measure, share))
        )
    ) {
        let mut agent = agent::<2, 1, 3>(2, &[first, second], Some(measure), share);
        let learner = agent.learner();
        let config = learner.config().clone();
        assert_eq!(config.epsilon, 0.8);
        let mut left = Classifier::general(Some(0), &config);
        left.q = 0.5;
        left.effect.symbols[0] = Symbol::Token(1);
        let mut right = left.clone();
        right.action = Some(1);
        right.q = 1.0;
        *agent.learner_mut() = EpisodicReplay::new(
            config,
            learner.replay_configuration(),
            learner.sampler().configuration(),
            learner.copy_update(),
            ChaChaRandomSource::from_seed(7),
            ChaChaRandomSource::from_seed(13),
        )
        .with_population(Population::from_classifiers(vec![left, right]))
        .with_per_goal_rewards(&[first, second]);
        let rewards = agent.learner_mut().per_goal_rewards_mut().unwrap();
        rewards.entry_mut(0, &first).r = 10.0;
        rewards.entry_mut(1, &first).r = 2.0;
        rewards.entry_mut(0, &second).r = 0.0;
        rewards.entry_mut(1, &second).r = 20.0;
        let measured = EpisodicGoalAgent(agent);
        let before = measured.0.snapshot();
        for seed in 42..92 {
            for (goal, expected) in [(first, (0, 5.0)), (second, (1, 20.0))] {
                let state =
                    GoalLayout::<2, 1, 3>::join(&Perception::new([Symbol::Token(0); 2]), &goal);
                assert_eq!(
                    measured.eval_action(&state, &mut ChaChaRandomSource::from_seed(seed)),
                    expected
                );
            }
        }
        assert_eq!(measured.0.snapshot(), before);
    }
}

#[test]
fn measurements_partition_draws_and_keep_learning_and_utility_work_separate() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let protocol = distributions.protocol();
    let goals = <_ as Task<8, 2, 10>>::training_goals(protocol.training).unwrap();
    for measure in MEASURES {
        let mut agent = EpisodicGoalAgent(agent::<8, 2, 10>(8, goals, Some(measure), 0.5));
        let mut rows = Vec::new();
        let metadata = RunMetadata {
            record_starts: false,
            ..RunMetadata::test("cost")
        };
        run_goal_agent_protocol_with_sink(
            &protocol,
            &mut agent,
            SEED,
            &[100, 6_000],
            &metadata,
            false,
            &mut |row| rows.push(row),
        );
        let row = &rows[1];
        let steps = row["actual_steps"].as_u64().unwrap();
        let learning_draws = row["replay_updates"].as_u64().unwrap();
        assert_eq!(learning_draws, 3 * steps);
        assert_eq!(row["match_formations_train"], steps + 3 * learning_draws);
        assert_eq!(row["learning_update_counts"]["full"], learning_draws);
        let stats = agent.0.trajectory_utility().unwrap();
        let draws = stats.draw_requests;
        assert_eq!(
            draws.original + draws.hindsight + draws.trajectory_utility,
            learning_draws
        );
        for count in [draws.original, draws.hindsight, draws.trajectory_utility] {
            assert!((count as f64 / learning_draws as f64 - 1.0 / 3.0).abs() < 0.013);
        }
        let cost = stats.cost;
        assert_eq!(
            row["trajectory_utility"]["cost"]["evaluations"],
            cost.insertions + cost.refreshes
        );
        assert_eq!(row["episodes"], cost.insertions);
        assert!(cost.refreshes > 0 && cost.cpu_nanoseconds > 0);
        assert!(cost.candidates_sampled < 8 * (cost.insertions + cost.refreshes));
        match measure {
            Measure::RandomGoal => {
                assert_eq!(cost.match_sets, 0);
                assert_eq!(cost.value_reads, 0);
                assert_eq!(cost.candidates_scored, 0);
            }
            Measure::MeanValue | Measure::TrialUpdate => {
                assert!(cost.match_sets > 0 && cost.value_reads > 0);
                assert_eq!(cost.candidates_scored, cost.candidates_sampled);
            }
        }
        assert_eq!(
            cost.trial_value_updates > 0,
            measure == Measure::TrialUpdate
        );
        assert_eq!(
            stats.mean_post_update_level.is_some(),
            measure == Measure::TrialUpdate
        );
        assert_eq!(
            row["replay_diagnostics"]["fallbacks"],
            stats.tu_fallbacks + stats.hindsight_fallbacks
        );
        assert_eq!(
            row["replay_diagnostics"]["strategies"]["trajectory_utility"],
            draws.trajectory_utility - stats.tu_fallbacks
        );
        assert_eq!(
            row["replay_diagnostics"]["per_goal_rewards"]["reward_updates"]["relabeled"],
            stats.reward_updates.hindsight + stats.reward_updates.trajectory_utility
        );
        assert_eq!(
            agent.0.learner().store().episodes().count(),
            stats.buffer_entries
        );
        let mut route_rng = ChaChaRandomSource::from_seed_and_stream(SEED, ROUTE_STREAM);
        let mut expected = [0; 3];
        for _ in 0..learning_draws {
            expected[route_rng.gen_range(3)] += 1;
        }
        assert_eq!(
            [draws.original, draws.hindsight, draws.trajectory_utility],
            expected
        );
    }
}

#[test]
fn cli_names_fix_the_copy_share() {
    for measure in MEASURES {
        let preset = AgentKind::from_run_name(measure.name(), 2.0 / 3.0).unwrap();
        assert_eq!(
            preset,
            AgentKind::Acs2TuHer(UtilityConfiguration::pilot(measure))
        );
        assert_eq!(preset.name(), measure.name());
        assert!(AgentKind::from_run_name(measure.name(), 0.5).is_err());
    }
}

#[test]
fn share_and_second_test_cap_preserve_state_and_all_registered_fields() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let mut protocol = distributions.protocol();
    let extended = distributions.test.with_cap(100);
    let goals = <_ as Task<8, 2, 10>>::training_goals(protocol.training).unwrap();
    let targets = [30, 120, 500];
    for measure in [Measure::MeanValue, Measure::RandomGoal] {
        for share in [0.25, 0.5, 0.75, 1.0] {
            let make = || EpisodicGoalAgent(agent::<8, 2, 10>(8, goals, Some(measure), share));
            protocol.second_test = Some(&extended);
            assert_protocol_evaluation_read_only(
                &protocol, SEED, &targets, make, |agent| agent.0.snapshot()
            );
            let run = |protocol| {
                let mut agent = make();
                let mut rows = Vec::new();
                run_goal_agent_protocol_with_sink(
                    protocol, &mut agent, SEED, &targets, &RunMetadata::test("test"), true,
                    &mut |row| rows.push(row),
                );
                (rows, agent.0.snapshot())
            };
            let (actual, actual_state) = run(&protocol);
            protocol.second_test = None;
            let (expected, expected_state) = run(&protocol);
            assert_eq!(actual_state, expected_state);
            for (mut actual, mut expected) in actual.into_iter().zip(expected) {
                let caps = actual.as_object_mut().unwrap().remove("test_caps").unwrap();
                let test = &expected["evaluation_blocks"]["test"];
                assert_eq!(caps["registered"]["success"], test["success"]);
                assert!(caps["extended"]["success"].as_f64().unwrap() >= test["success"].as_f64().unwrap());
                for (short, long) in test["starts"].as_array().unwrap().iter().zip(
                    caps["extended"]["starts"].as_array().unwrap()
                ) {
                    if short["success"] == true {
                        assert_eq!(short["success"], long["success"]);
                        assert_eq!(short["steps"], long["steps"]);
                    }
                }
                for row in [&mut actual, &mut expected] {
                    let cost = row["trajectory_utility"]["cost"].as_object_mut().unwrap();
                    cost.retain(|key, _| !key.ends_with("cpu_seconds"));
                }
                assert_eq!(without_timing(actual).to_string(), without_timing(expected).to_string());
            }
        }
    }
}
