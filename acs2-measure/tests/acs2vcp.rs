use acs2_agents::{acs2her, acs2vcp};
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, vcp::VcpGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, evaluate, run_goal_agent_protocol_with_sink,
    run_protocol_with_sink, AgentKind, GoalAgent, Preset, RunMetadata, RunSettings,
};
use acs2_measure::task::{MazeTask, Task};
use serde_json::Value;

type MeasuredVcp =
    VcpGoalAgent<8, 2, 10, ChaChaRandomSource, ChaChaRandomSource, ChaChaRandomSource>;

fn make(preset: acs2vcp::Preset) -> MeasuredVcp {
    VcpGoalAgent {
        agent: acs2vcp::Agent::seeded(
            Preset::thesis().config(8),
            episodic_configuration(Preset::thesis()),
            preset.parameters(),
            42,
            1,
        ),
        preset,
    }
}

fn bits(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            Value::String(format!("f64:{:016x}", number.as_f64().unwrap().to_bits()))
        }
        Value::Array(values) => Value::Array(values.iter().map(bits).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), bits(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[test]
fn one_head_at_zero_and_positive_temperature_reproduces_every_pinned_hindsight_field() {
    let oracle: Value =
        serde_json::from_str(include_str!("../../fixtures/her_pair_regression.json")).unwrap();
    let mut checked = 0;
    for configuration in oracle["configurations"].as_array().unwrap() {
        let archived = configuration["agent"].as_str().unwrap();
        let hindsight = match archived {
            "er_episodic" => acs2her::HindsightConfiguration {
                relabeled_proportion: 0.0,
                ..acs2her::Preset::Acs2Her.configuration()
            },
            "her_full" => acs2her::Preset::Acs2Her.configuration(),
            "her_values" => acs2her::Preset::Acs2HerValues {
                proportion: 2.0 / 3.0,
            }
            .configuration(),
            "her_values_shortened" => acs2her::Preset::Acs2HerValuesShortened {
                proportion: 2.0 / 3.0,
            }
            .configuration(),
            _ => continue,
        };
        let task = configuration["task"].as_str().unwrap();
        let acs2_envs::roles::ResearchTask::GoalMaze(geometry) =
            acs2_envs::roles::ResearchTask::named(task).unwrap()
        else {
            unreachable!()
        };
        let distributions = MazePairDistributions::<Coordinates, 2>::registered(
            task,
            geometry,
            configuration["cap"].as_u64().unwrap() as u32,
            "coordinates",
        );
        for temperature in [0, 24] {
            let mut agent = VcpGoalAgent {
                agent: acs2vcp::Agent::<8, 2, 10, _, _, _>::seeded(
                    Preset::thesis().config(8),
                    episodic_configuration(Preset::thesis()),
                    acs2vcp::ConsistencyConfiguration {
                        heads: 1,
                        temperature,
                        hindsight,
                        reward_on_reach: 1000.0,
                    },
                    42,
                    1,
                ),
                preset: acs2vcp::Preset::UniformTwo,
            };
            let mut rows = vec![];
            run_goal_agent_protocol_with_sink(
                &distributions.protocol(),
                &mut agent,
                42,
                &[1000, 3000],
                &RunMetadata {
                    record_starts: false,
                    ..RunMetadata::test("test")
                },
                true,
                &mut |row| rows.push(row),
            );
            for (mut row, expected) in rows
                .into_iter()
                .zip(configuration["rows"].as_array().unwrap())
            {
                for key in oracle["excluded"].as_array().unwrap() {
                    row.as_object_mut().unwrap().remove(key.as_str().unwrap());
                }
                for key in ["priority", "proposal_routes"] {
                    row["replay_diagnostics"]
                        .as_object_mut()
                        .unwrap()
                        .remove(key);
                }
                let mut expected = expected.clone();
                for key in ["agent", "agent_parameters", "ensemble"] {
                    row.as_object_mut().unwrap().remove(key);
                    expected.as_object_mut().unwrap().remove(key);
                }
                assert_eq!(bits(&row), expected, "{task} {archived} T={temperature}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 96);
}

#[test]
fn all_registered_ensembles_preserve_every_head_store_stream_and_counter_during_evaluation() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for preset in [
        acs2vcp::Preset::PrioritizedTwo { temperature: 24 },
        acs2vcp::Preset::PrioritizedFour { temperature: 24 },
        acs2vcp::Preset::UniformTwo,
        acs2vcp::Preset::UniformFour,
    ] {
        assert_protocol_evaluation_read_only(
            &distributions.protocol(),
            42,
            &[20, 60, 100],
            || make(preset),
            |agent| agent.agent.snapshot(),
        );
    }
}

#[test]
fn ensemble_rows_count_joint_interactions_updates_queries_routes_and_total_memory() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let baseline_steps = |kind| {
        let mut steps = vec![];
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[20, 60, 100],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: false,
                capture_final_state: false,
            },
            &mut |row| steps.push(row["actual_steps"].as_u64().unwrap()),
        );
        steps
    };
    let reference = baseline_steps(AgentKind::Acs2Her);
    for preset in [
        acs2vcp::Preset::PrioritizedTwo { temperature: 4 },
        acs2vcp::Preset::PrioritizedFour { temperature: 4 },
        acs2vcp::Preset::UniformTwo,
        acs2vcp::Preset::UniformFour,
    ] {
        let mut rows = vec![];
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            AgentKind::Acs2Vcp(preset),
            42,
            &[20, 60, 100],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        for (point, row) in rows.iter().enumerate() {
            let steps = row["actual_steps"].as_u64().unwrap();
            assert!(steps >= [20, 60, 100][point] && steps < [20, 60, 100][point] + 5);
            assert!((steps as i64 - reference[point] as i64).abs() < 5);
            let counts = &row["learning_update_counts"];
            let full = counts["full"].as_u64().unwrap();
            let values = counts["values_only"].as_u64().unwrap();
            assert_eq!(full + values, 3 * steps);
            let priority = &row["replay_diagnostics"]["priority"];
            let queries = priority["value_queries"].as_u64().unwrap();
            assert_eq!(
                row["match_formations_train"],
                steps + 3 * full + 2 * values + queries
            );
            assert_eq!(priority["accepted_draws"], full + values);
            assert_eq!(
                queries,
                priority["proposals"].as_u64().unwrap() * preset.parameters().heads as u64
            );
            let heads = row["ensemble"]["heads"].as_array().unwrap();
            assert_eq!(heads.len(), preset.parameters().heads);
            for field in [
                "population_classifiers",
                "population_numerosity",
                "population_logical_bytes",
                "population_mark_entries",
                "population_known_bytes_lower_bound",
            ] {
                assert_eq!(
                    row[field],
                    heads
                        .iter()
                        .map(|head| head[field].as_u64().unwrap())
                        .sum::<u64>(),
                    "{field}"
                );
            }
            assert_eq!(
                steps,
                heads
                    .iter()
                    .map(|head| head["environment_steps"].as_u64().unwrap())
                    .sum::<u64>()
            );
            assert_eq!(
                full,
                heads
                    .iter()
                    .map(|head| head["full"].as_u64().unwrap())
                    .sum::<u64>()
            );
            let mean_knowledge = heads
                .iter()
                .map(|head| head["knowledge"].as_f64().unwrap())
                .sum::<f64>()
                / heads.len() as f64;
            assert_eq!(
                row["knowledge"].as_f64().unwrap().to_bits(),
                mean_knowledge.to_bits()
            );
            for block in ["training", "validation", "test"] {
                let evaluated = &row["evaluation_blocks"][block];
                let per_head = evaluated["heads"].as_array().unwrap();
                assert_eq!(per_head.len(), heads.len());
                let mean = per_head
                    .iter()
                    .map(|head| head["success"].as_f64().unwrap() / heads.len() as f64)
                    .sum::<f64>();
                assert_eq!(
                    evaluated["success"].as_f64().unwrap().to_bits(),
                    mean.to_bits()
                );
            }
        }
    }
}

struct GreedyRandom;
impl RandomSource for GreedyRandom {
    fn gen_bool(&mut self, _: f64) -> bool {
        panic!("evaluation must ignore epsilon 0.8")
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("unexpected unit draw")
    }
    fn gen_range(&mut self, _: usize) -> usize {
        0
    }
}

#[test]
fn every_head_uses_the_goal_in_greedy_evaluation_and_exploration_and_evaluation_averages_different_heads(
) {
    let preset = Preset::thesis();
    let config = preset.config(8);
    assert_eq!(config.epsilon, 0.8);
    let task = MazeTask::<Coordinates, 2>::new(
        "maze4".to_owned(),
        &MAZE4,
        vec![(1, 2), (1, 1)],
        1,
        "coordinates",
    );
    let mut env = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::environment(
        &task,
        ChaChaRandomSource::from_seed(42),
    );
    let goal = Goal::new([Symbol::Token(1), Symbol::Token(2)]);
    let start =
        <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::reset_at(&task, &mut env, (1, 1), goal);
    let state = GoalLayout::<8, 2, 10>::join(&start.observation, &goal);
    let populations = [2, 6]
        .into_iter()
        .map(|action| {
            let mut classifier = Classifier::<10>::general(Some(action), &config);
            classifier.effect.symbols[0] = Symbol::Token(0);
            classifier.q = 1.0;
            classifier.r = 100.0;
            Population::from_classifiers(vec![classifier])
        })
        .collect();
    let mut agent = make(acs2vcp::Preset::UniformTwo);
    agent.agent = agent.agent.with_populations(populations);
    assert_eq!(agent.eval_head_action(0, &state, &mut GreedyRandom).0, 2);
    assert_eq!(agent.eval_head_action(1, &state, &mut GreedyRandom).0, 6);
    let before = agent.agent.snapshot();
    let result = evaluate::<_, _, 8, 2, 10>(
        &task,
        &agent,
        &[((1, 1), goal, 1.0)],
        42,
        config.gamma,
        true,
    );
    let heads = result.heads.unwrap();
    assert_eq!(heads[0].success, 1.0);
    assert_eq!(heads[1].success, 0.0);
    assert_eq!(result.success, 0.5);
    assert_eq!(agent.agent.snapshot(), before);
    let mut rules = vec![];
    for desired in [7, 9] {
        for action in [0, 1] {
            let mut classifier = Classifier::general(Some(action), &config);
            classifier.condition.symbols[1] = Symbol::Token(desired);
            classifier.effect.symbols[0] = Symbol::Token(1);
            classifier.r = if (desired == 7 && action == 0) || (desired == 9 && action == 1) {
                100.0
            } else {
                1.0
            };
            rules.push(classifier);
        }
    }
    let mut config = preset.config(2);
    config.epsilon = 0.0;
    let mut learner = acs2vcp::Agent::<1, 1, 2, _, _, _>::seeded(
        config,
        episodic_configuration(preset),
        acs2vcp::Preset::UniformFour.parameters(),
        42,
        1,
    )
    .with_populations(
        (0..4)
            .map(|_| Population::from_classifiers(rules.clone()))
            .collect(),
    );
    for (desired, action) in [(7, 0), (9, 1)] {
        let state = Perception {
            symbols: [Symbol::Token(0), Symbol::Token(desired)],
        };
        for head in 0..4 {
            assert_eq!(
                learner.greedy_action(head, &state, &mut GreedyRandom).0,
                action
            );
        }
        assert_eq!(learner.explore_action(&state), action);
    }
}

#[test]
fn registered_vcp_arms_reproduce_the_pinned_pilot_rows_bit_for_bit_on_the_measured_native_layouts()
{
    use std::mem::size_of;
    let oracle: Value =
        serde_json::from_str(include_str!("../../fixtures/vcp_pair_regression.json")).unwrap();
    let head_bytes = size_of::<acs2vcp::Head<10, ChaChaRandomSource>>();
    let agent_bytes = size_of::<
        acs2vcp::Agent<8, 2, 10, ChaChaRandomSource, ChaChaRandomSource, ChaChaRandomSource>,
    >() - size_of::<acs2_trajectory::TrajectoryStore<8, 2>>();
    assert!(
        oracle["inline_memory_layouts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|layout| layout["head_inline_bytes"] == head_bytes
                && layout["agent_inline_without_store_bytes"] == agent_bytes),
        "unmeasured native inline memory layout"
    );
    let mut arms = std::collections::BTreeSet::new();
    for configuration in oracle["configurations"].as_array().unwrap() {
        let name = configuration["agent"].as_str().unwrap();
        arms.insert(name);
        let temperature = configuration["temperature"].as_u64().unwrap() as u32;
        let kind = match name {
            "acs2vcp_k2" => acs2vcp::Preset::PrioritizedTwo { temperature },
            "acs2vcp_k4" => acs2vcp::Preset::PrioritizedFour { temperature },
            "acs2vcp_k2_uniform" => acs2vcp::Preset::UniformTwo,
            "acs2vcp_k4_uniform" => acs2vcp::Preset::UniformFour,
            _ => panic!("unregistered VCP arm"),
        };
        let task = configuration["task"].as_str().unwrap();
        let acs2_envs::roles::ResearchTask::GoalMaze(geometry) =
            acs2_envs::roles::ResearchTask::named(task).unwrap()
        else {
            unreachable!()
        };
        let distributions = MazePairDistributions::<Coordinates, 2>::registered(
            task,
            geometry,
            configuration["cap"].as_u64().unwrap() as u32,
            "coordinates",
        );
        let full = distributions.protocol();
        let protocol = acs2_measure::goal_protocol::TaskProtocol {
            training: full.training,
            evaluations: full
                .evaluations
                .into_iter()
                .filter(|block| block.name == "validation")
                .collect(),
            registration: full.registration,
        };
        let targets: Vec<_> = oracle["nominal_points"]
            .as_array()
            .unwrap()
            .iter()
            .map(|point| point.as_u64().unwrap())
            .collect();
        let mut rows = vec![];
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &protocol,
            AgentKind::Acs2Vcp(kind),
            42,
            &targets,
            &RunMetadata {
                record_starts: false,
                ..RunMetadata::test("test")
            },
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        assert_eq!(rows.len(), configuration["rows"].as_array().unwrap().len());
        for (mut row, expected) in rows
            .into_iter()
            .zip(configuration["rows"].as_array().unwrap())
        {
            for key in oracle["excluded"].as_array().unwrap() {
                row.as_object_mut().unwrap().remove(key.as_str().unwrap());
            }
            let mut expected = expected.clone();
            let mut total = 0;
            for head in expected["ensemble"]["heads"].as_array_mut().unwrap() {
                let portable = head["population_known_bytes_lower_bound"].as_u64().unwrap();
                assert_eq!(
                    head["head_known_bytes_lower_bound"],
                    portable
                        + oracle["source_inline_memory_layout"]["head_inline_bytes"]
                            .as_u64()
                            .unwrap()
                );
                head["head_known_bytes_lower_bound"] =
                    serde_json::json!(portable + head_bytes as u64);
                total += portable + head_bytes as u64;
            }
            expected["ensemble"]["heads_known_bytes_lower_bound"] = serde_json::json!(total);
            expected["ensemble"]["agent_known_bytes_lower_bound"] = serde_json::json!(
                total + expected["trajectory_logical_bytes"].as_u64().unwrap() + agent_bytes as u64
            );
            assert_eq!(bits(&row), expected, "{task} {name} T={temperature}");
        }
    }
    assert_eq!(
        arms,
        [
            "acs2vcp_k2",
            "acs2vcp_k4",
            "acs2vcp_k2_uniform",
            "acs2vcp_k4_uniform"
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn vcp_cli_refuses_unregistered_hindsight_names_and_invalid_temperature_combinations() {
    let output_path =
        std::env::temp_dir().join(format!("acs2vcp-cli-{}.jsonl", std::process::id()));
    let base = [
        "--task",
        "maze4",
        "--cap",
        "5",
        "--protocol",
        "maze-pairs-v1",
        "--seeds",
        "42",
        "--targets",
        "20",
        "--evaluation-blocks",
        "validation",
    ];
    for (name, option, value) in [
        ("acs2vcp_k2_values", "--vcp-temperature", "1"),
        ("acs2vcp_k2", "--vcp-temperature", "0"),
        ("acs2vcp_k4", "--vcp-temperature", "25"),
        ("acs2vcp_k2_uniform", "--vcp-temperature", "1"),
        ("acs2vcp_k2", "--relabel-proportion", "0.5"),
    ] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
            .args(base)
            .args(["--agents", name, option, value, "--out"])
            .arg(&output_path)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{name} {option} {value}");
    }
    assert!(!output_path.exists());
}
