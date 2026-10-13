use acs2_agents::episodic_replay::{EpisodicAgent, EpisodicReplay};
use acs2_agents::acs2er::episodic;
use acs2_agents::acs2her;
use acs2_agents::acs2er::Agent as Acs2ErAgent;
use acs2_core::checkpoint::Checkpointed;
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agent_names::{canonical_row, canonical_run_name};
use acs2_measure::agents::{episodic_configuration, OneUpdateAdapter, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_protocol_with_sink, AgentKind, CoreAgent, GoalAgent,
    Preset, RunMetadata, RunSettings,
};
use serde_json::Value;

fn random(stream: u64) -> ChaChaRandomSource {
    ChaChaRandomSource::from_seed_and_stream(42, stream)
}

struct EvaluationRandom;

impl RandomSource for EvaluationRandom {
    fn gen_bool(&mut self, _: f64) -> bool {
        panic!("greedy evaluation must not draw an epsilon coin")
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("greedy evaluation needs no unit draw")
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        0
    }
}

fn check_greedy_evaluation<A>(mut agent: A)
where
    A: EpisodicAgent<
        1,
        1,
        2,
        PolicyRandom = ChaChaRandomSource,
        SamplerRandom = ChaChaRandomSource,
    >,
{
    let config = Preset::thesis().config(2);
    assert_eq!(config.epsilon.to_bits(), 0.8_f64.to_bits());
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
    *agent.learner_mut() = EpisodicReplay::new(
        config,
        agent.learner().replay_configuration(),
        agent.learner().sampler().configuration(),
        agent.learner().copy_update(),
        random(1),
        random(7),
    )
    .with_population(Population::from_classifiers(rules));
    let measured = EpisodicGoalAgent(agent);
    let before = measured.0.learner().snapshot();
    for (desired, action) in [(7, 0), (9, 1)] {
        let state = Perception {
            symbols: [Symbol::Token(0), Symbol::Token(desired)],
        };
        assert_eq!(
            measured.eval_action(&state, &mut EvaluationRandom).0,
            action
        );
    }
    assert_eq!(measured.0.learner().snapshot(), before);
}

#[test]
fn episodic_evaluation_ignores_the_thesis_epsilon_of_point_eight() {
    let preset = Preset::thesis();
    check_greedy_evaluation(episodic::Agent::<1, 1, 2, _, _>::new(
        preset.config(2),
        episodic_configuration(preset),
        random(1),
        random(7),
    ));
    for variant in [
        acs2her::Preset::Acs2Her,
        acs2her::Preset::Acs2HerValues {
            proportion: 2.0 / 3.0,
        },
        acs2her::Preset::Acs2HerValuesShortened {
            proportion: 2.0 / 3.0,
        },
    ] {
        check_greedy_evaluation(acs2her::Agent::<1, 1, 2, _, _>::new(
            preset.config(2),
            episodic_configuration(preset),
            variant,
            random(1),
            random(7),
        ));
    }
}

#[test]
fn acs2er_one_update_performs_exactly_one_replay_per_environment_step() {
    let preset = Preset::thesis();
    assert_eq!(
        acs2_agents::acs2er::one_update::OneUpdateConfiguration {
            capacity: preset.replay_capacity,
            warmup: preset.replay_warmup,
        }
        .replay()
        .samples_number,
        1
    );
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    run_protocol_with_sink::<_, _, 8, 2, 10>(
        &distributions.protocol(),
        AgentKind::Acs2ErOneUpdate,
        42,
        &[20, 60, 100],
        &RunMetadata::test("test"),
        RunSettings {
            evaluate_points: false,
            capture_final_state: false,
        },
        &mut |row| {
            let steps = row["actual_steps"].as_u64().unwrap();
            assert_eq!(row["match_formations_train"], 4 * steps);
            assert_eq!(row["replay_updates"], steps);
        },
    );
}

#[test]
fn acs2her_factory_uses_sampler_stream_seven_independently_of_agent_stream() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for kind in [
        AgentKind::Acs2ErEpisodic,
        AgentKind::Acs2Her,
        AgentKind::Acs2HerValues {
            proportion: 2.0 / 3.0,
        },
        AgentKind::Acs2HerValuesShortened {
            proportion: 2.0 / 3.0,
        },
    ] {
        let (_, snapshot) = run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[20],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: false,
                capture_final_state: true,
            },
            &mut |_| {},
        )
        .unwrap();
        let sampler_random = snapshot.rsplit('|').nth(2).unwrap();
        let agent_random = snapshot.rsplit('|').nth(4).unwrap();
        assert!(sampler_random.contains("stream: 7,"), "{sampler_random}");
        assert!(agent_random.contains("stream: 1,"), "{agent_random}");
    }
}

#[test]
fn all_episodic_agents_preserve_population_store_sampler_rng_and_counters_under_evaluation() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let protocol = distributions.protocol();
    let preset = Preset::thesis();
    assert_protocol_evaluation_read_only(
        &protocol,
        42,
        &[10, 40, 100],
        || {
            EpisodicGoalAgent(episodic::Agent::<8, 2, 10, _, _>::new(
                preset.config(8),
                episodic_configuration(preset),
                random(1),
                random(7),
            ))
        },
        |agent| agent.0.learner().snapshot(),
    );
    for variant in [
        acs2her::Preset::Acs2Her,
        acs2her::Preset::Acs2HerValues {
            proportion: 2.0 / 3.0,
        },
        acs2her::Preset::Acs2HerValuesShortened {
            proportion: 2.0 / 3.0,
        },
    ] {
        assert_protocol_evaluation_read_only(
            &protocol,
            42,
            &[10, 40, 100],
            || {
                EpisodicGoalAgent(acs2her::Agent::<8, 2, 10, _, _>::new(
                    preset.config(8),
                    episodic_configuration(preset),
                    variant,
                    random(1),
                    random(7),
                ))
            },
            |agent| agent.0.learner().snapshot(),
        );
    }
}

#[test]
fn acs2er_one_update_preserves_the_unchanged_core_replay_and_rng_during_evaluation() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    let replay = acs2_agents::acs2er::one_update::OneUpdateConfiguration {
        capacity: preset.replay_capacity,
        warmup: preset.replay_warmup,
    };
    assert_protocol_evaluation_read_only(
        &distributions.protocol(),
        42,
        &[10, 40, 100],
        || {
            OneUpdateAdapter(CoreAgent {
                agent: acs2_agents::acs2er::one_update::Agent::<10, _>::new(
                    preset.config(8),
                    replay,
                    preset.truncation,
                    random(1),
                ),
                preset: Preset {
                    replay_updates_per_step: 1,
                    ..preset
                },
                replay: true,
                steps: 0,
                updates: 0,
            })
        },
        |agent| {
            format!(
                "{:?}|{}|{}",
                agent.0.agent.snapshot(),
                agent.0.steps,
                agent.0.updates
            )
        },
    );
}

#[test]
fn replay_rows_report_realized_routes_cost_memory_and_counts_by_update_kind() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for kind in [
        AgentKind::Acs2ErEpisodic,
        AgentKind::Acs2Her,
        AgentKind::Acs2HerValues {
            proportion: 2.0 / 3.0,
        },
        AgentKind::Acs2HerValuesShortened {
            proportion: 2.0 / 3.0,
        },
        AgentKind::Acs2ErOneUpdate,
    ] {
        let mut rows = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[20, 60, 100],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        for row in rows {
            let steps = row["actual_steps"].as_u64().unwrap();
            let updates = row["learning_update_counts"]["full"].as_u64().unwrap()
                + row["learning_update_counts"]["values_only"]
                    .as_u64()
                    .unwrap();
            assert_eq!(row["online_updates"], 0);
            assert_eq!(row["replay_updates"], updates);
            assert_eq!(
                updates,
                if kind == AgentKind::Acs2ErOneUpdate {
                    steps
                } else {
                    3 * steps
                }
            );
            assert!(row["knowledge"].is_number());
            assert!(row["population_logical_bytes"].is_number());
            assert!(row["evaluation_blocks"]["test"]["success"].is_number());
            if kind == AgentKind::Acs2ErOneUpdate {
                assert_eq!(row["agent_parameters"]["replay_updates_per_step"], 1);
                assert_eq!(row["replay_diagnostics"]["original"], updates);
                assert_eq!(row["replay_diagnostics"]["relabeled"], 0);
                assert_eq!(row["replay_diagnostics"]["objective_evaluations"], 0);
                continue;
            }
            let diagnostics = &row["replay_diagnostics"];
            let relabeled = diagnostics["relabeled"].as_u64().unwrap();
            assert_eq!(diagnostics["draws"], updates);
            assert_eq!(
                diagnostics["original"].as_u64().unwrap() + relabeled,
                updates
            );
            assert_eq!(
                diagnostics["realized_relabeled_share"].as_f64().unwrap(),
                relabeled as f64 / updates as f64
            );
            assert_eq!(
                diagnostics["objective_evaluations"].as_u64().unwrap(),
                ["scoring", "selection", "provenance"]
                    .iter()
                    .map(
                        |purpose| diagnostics["objective_evaluations_by_purpose"][purpose]
                            ["objective_evaluations"]
                            .as_u64()
                            .unwrap()
                    )
                    .sum::<u64>()
            );
            assert!(row["trajectory_logical_bytes"].as_u64().unwrap() > 0);
            assert_eq!(row["agent_parameters"]["truncation"], "Bootstrap");
            assert_eq!(
                row["agent_parameters"]["sampler_truncation"],
                row["agent_parameters"]["truncation"]
            );
            match kind {
                AgentKind::Acs2ErEpisodic => {
                    assert_eq!(relabeled, 0);
                    assert_eq!(row["learning_update_counts"]["full"], updates);
                }
                AgentKind::Acs2Her => {
                    assert!(relabeled > 0);
                    assert_eq!(row["learning_update_counts"]["full"], updates);
                }
                _ => {
                    assert_eq!(row["learning_update_counts"]["values_only"], relabeled);
                    assert_eq!(row["learning_update_counts"]["full"], updates - relabeled);
                }
            }
        }
    }
}

#[test]
fn pair_baselines_reproduce_archived_seed42_learning_fields() {
    let oracle: Value =
        serde_json::from_str(include_str!("../../fixtures/pair_baseline_regression.json")).unwrap();
    for configuration in oracle["configurations"].as_array().unwrap() {
        let geometry =
            acs2_envs::roles::ResearchTask::named(configuration["task"].as_str().unwrap()).unwrap();
        let acs2_envs::roles::ResearchTask::GoalMaze(geometry) = geometry else {
            unreachable!()
        };
        let distributions = MazePairDistributions::<Coordinates, 2>::registered(
            configuration["task"].as_str().unwrap(),
            geometry,
            configuration["cap"].as_u64().unwrap() as u32,
            "coordinates",
        );
        let kind = AgentKind::from_run_name(
            canonical_run_name(configuration["agent"].as_str().unwrap()),
            2.0 / 3.0,
        )
        .unwrap();
        let targets: Vec<_> = configuration["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["nominal_step"].as_u64().unwrap())
            .collect();
        let mut actual = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
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
            &mut |row| actual.push(row),
        );
        for (mut row, expected) in actual
            .into_iter()
            .zip(configuration["rows"].as_array().unwrap())
        {
            for key in [
                "commit",
                "source_state",
                "host",
                "cpu_model",
                "wall_seconds_train",
                "wall_seconds_eval",
                "wall_seconds_total",
            ] {
                row.as_object_mut().unwrap().remove(key);
            }
            assert_eq!(
                row, canonical_row(expected),
                "{} {}",
                configuration["task"], configuration["agent"]
            );
        }
    }
}

fn floating_field_bits(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            Value::String(format!("f64:{:016x}", number.as_f64().unwrap().to_bits()))
        }
        Value::Array(values) => Value::Array(values.iter().map(floating_field_bits).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), floating_field_bits(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[test]
fn new_replay_arms_reproduce_pinned_seed42_pilot_fields_bit_for_bit() {
    let oracle: Value =
        serde_json::from_str(include_str!("../../fixtures/her_pair_regression.json")).unwrap();
    assert_eq!(oracle["configurations"].as_array().unwrap().len(), 30);
    for configuration in oracle["configurations"].as_array().unwrap() {
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
        let kind = AgentKind::from_run_name(
            canonical_run_name(configuration["agent"].as_str().unwrap()),
            2.0 / 3.0,
        )
        .unwrap();
        let expected = configuration["rows"].as_array().unwrap();
        let mut actual = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[1000, 3000],
            &RunMetadata {
                record_starts: false,
                ..RunMetadata::test("test")
            },
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| actual.push(row),
        );
        assert_eq!(actual.len(), expected.len());
        for (mut row, pinned) in actual.into_iter().zip(expected) {
            for key in oracle["excluded"].as_array().unwrap() {
                row.as_object_mut().unwrap().remove(key.as_str().unwrap());
            }
            assert_eq!(
                floating_field_bits(&row),
                canonical_row(pinned),
                "{} {} at {}",
                task,
                configuration["agent"],
                row["nominal_step"]
            );
        }
    }
}

#[test]
fn cli_can_measure_validation_only_and_refuses_tuning_the_full_reference() {
    let path = std::env::temp_dir().join(format!("acs2her-cli-{}.jsonl", std::process::id()));
    let args = [
        "--task",
        "maze4",
        "--cap",
        "5",
        "--protocol",
        "maze-pairs-v1",
        "--agents",
        "acs2her_values",
        "--seeds",
        "42",
        "--targets",
        "20",
        "--evaluation-blocks",
        "validation",
        "--relabel-proportion",
        "0.9",
        "--out",
    ];
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .args(args)
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let row: Value = serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
    assert_eq!(
        row["evaluation_blocks"]
            .as_object()
            .unwrap()
            .keys()
            .collect::<Vec<_>>(),
        ["validation"]
    );
    assert_eq!(row["agent_parameters"]["relabeled_proportion"], 0.9);
    let mut invalid = args;
    invalid[7] = "acs2her";
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .args(invalid)
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("only values variants"));
    std::fs::remove_file(path).unwrap();
}

#[test]
fn u1_observer_preserves_the_exact_core_state_and_counts_actual_replayed_endings() {
    use acs2_measure::runner::run_goal_agent_protocol_with_sink;
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset {
        replay_updates_per_step: 1,
        ..Preset::thesis()
    };
    let parameters = acs2_agents::acs2er::one_update::OneUpdateConfiguration {
        capacity: preset.replay_capacity,
        warmup: preset.replay_warmup,
    };
    let mut plain = CoreAgent {
        agent: Acs2ErAgent::<10, _>::new(preset.config(8), parameters.replay(), random(1))
            .with_truncation_mode(preset.truncation),
        preset,
        replay: true,
        steps: 0,
        updates: 0,
    };
    let mut observed = OneUpdateAdapter(CoreAgent {
        agent: acs2_agents::acs2er::one_update::Agent::<10, _>::new(
            preset.config(8),
            parameters,
            preset.truncation,
            random(1),
        ),
        preset,
        replay: true,
        steps: 0,
        updates: 0,
    });
    let mut plain_rows = Vec::new();
    let mut observed_rows = Vec::new();
    let targets = [100, 1000, 10000];
    run_goal_agent_protocol_with_sink(
        &distributions.protocol(),
        &mut plain,
        42,
        &targets,
        &RunMetadata::test("test"),
        true,
        &mut |row| plain_rows.push(row),
    );
    run_goal_agent_protocol_with_sink(
        &distributions.protocol(),
        &mut observed,
        42,
        &targets,
        &RunMetadata::test("test"),
        true,
        &mut |row| observed_rows.push(row),
    );
    assert_eq!(
        format!("{:?}", plain.agent.capture()),
        format!("{:?}", observed.0.agent.captured_state())
    );
    for (mut plain, mut observed) in plain_rows.into_iter().zip(observed_rows) {
        for key in [
            "agent",
            "replay_diagnostics",
            "learning_update_counts",
            "replay_logical_bytes",
            "wall_seconds_train",
            "wall_seconds_eval",
            "wall_seconds_total",
        ] {
            plain.as_object_mut().unwrap().remove(key);
            observed.as_object_mut().unwrap().remove(key);
        }
        assert_eq!(plain, observed);
    }
    let counters = observed.0.agent.counters();
    assert_eq!(counters.original.draws, observed.0.updates);
    assert!(counters.original.done > 0);
    assert_eq!(
        counters.original.reward,
        1000.0 * counters.original.done as f64
    );
    assert_eq!(counters.relabeled.draws, 0);
    assert_eq!(counters.cost.total().total(), 0);
}
