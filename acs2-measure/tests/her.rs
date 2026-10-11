use acs2_agents::replay::EpisodicAgent;
use acs2_agents::{er_episodic, her};
use acs2_core::agents::acs2er::Acs2ErAgent;
use acs2_core::checkpoint::Checkpointed;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, DiagnosticAdapter, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_protocol_with_sink, AgentKind, CoreAgent, Preset,
    RunMetadata, RunSettings,
};
use serde_json::Value;

fn random(stream: u64) -> ChaChaRandomSource {
    ChaChaRandomSource::from_seed_and_stream(42, stream)
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
            EpisodicGoalAgent(er_episodic::Agent::<8, 2, 10, _, _>::new(
                preset.config(8),
                episodic_configuration(preset),
                random(1),
                random(7),
            ))
        },
        |agent| agent.0.learner().snapshot(),
    );
    for variant in [
        her::Variant::Full,
        her::Variant::Values {
            proportion: 2.0 / 3.0,
        },
        her::Variant::ValuesShortened {
            proportion: 2.0 / 3.0,
        },
    ] {
        assert_protocol_evaluation_read_only(
            &protocol,
            42,
            &[10, 40, 100],
            || {
                EpisodicGoalAgent(her::Agent::<8, 2, 10, _, _>::new(
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
fn diagnostic_u1_preserves_the_unchanged_core_replay_and_rng_during_evaluation() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    let replay = acs2_agents::acs2er_u1::DiagnosticConfiguration {
        capacity: preset.replay_capacity,
        warmup: preset.replay_warmup,
    }
    .replay();
    assert_protocol_evaluation_read_only(
        &distributions.protocol(),
        42,
        &[10, 40, 100],
        || {
            DiagnosticAdapter(CoreAgent {
                agent: Acs2ErAgent::<10, _>::new(preset.config(8), replay, random(1))
                    .with_truncation_mode(preset.truncation),
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
                agent.0.agent.capture(),
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
        AgentKind::ErEpisodic,
        AgentKind::HerFull,
        AgentKind::HerValues {
            proportion: 2.0 / 3.0,
        },
        AgentKind::HerValuesShortened {
            proportion: 2.0 / 3.0,
        },
        AgentKind::Acs2ErU1,
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
                if kind == AgentKind::Acs2ErU1 {
                    steps
                } else {
                    3 * steps
                }
            );
            assert!(row["knowledge"].is_number());
            assert!(row["population_logical_bytes"].is_number());
            assert!(row["evaluation_blocks"]["test"]["success"].is_number());
            if kind == AgentKind::Acs2ErU1 {
                assert_eq!(row["agent_parameters"]["replay_updates_per_step"], 1);
                assert!(row["replay_diagnostics"].is_null());
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
                AgentKind::ErEpisodic => {
                    assert_eq!(relabeled, 0);
                    assert_eq!(row["learning_update_counts"]["full"], updates);
                }
                AgentKind::HerFull => {
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
        let kind = if configuration["agent"] == "acs2" {
            AgentKind::Acs2
        } else {
            AgentKind::Acs2Er
        };
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
                &row, expected,
                "{} {}",
                configuration["task"], configuration["agent"]
            );
        }
    }
}

#[test]
fn cli_can_measure_validation_only_and_refuses_tuning_the_full_reference() {
    let path = std::env::temp_dir().join(format!("her-cli-{}.jsonl", std::process::id()));
    let args = ["--task", "maze4", "--cap", "5", "--protocol", "maze-pairs-v1", "--agents", "her_values", "--seeds", "42", "--targets", "20", "--evaluation-blocks", "validation", "--relabel-proportion", "0.9", "--out"];
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure")).args(args).arg(&path).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let row: Value = serde_json::from_str(std::fs::read_to_string(&path).unwrap().trim()).unwrap();
    assert_eq!(row["evaluation_blocks"].as_object().unwrap().keys().collect::<Vec<_>>(), ["validation"]);
    assert_eq!(row["agent_parameters"]["relabeled_proportion"], 0.9);
    let mut invalid = args;
    invalid[7] = "her_full";
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure")).args(invalid).arg(&path).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("only values variants"));
    std::fs::remove_file(path).unwrap();
}
