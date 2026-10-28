use acs2_agents::acs2her;
use acs2_agents::episodic_replay::EpisodicAgent;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_protocol_with_sink, AgentKind, Preset, RunMetadata,
    RunSettings,
};

#[test]
fn requested_proportions_reach_full_updates_and_measured_routes() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for (name, proportion) in [
        ("acs2her_shortened", 1.0 / 3.0),
        ("acs2her_shortened", 0.5),
        ("acs2her", 0.5),
    ] {
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            AgentKind::from_run_name(name, proportion).unwrap(),
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
            &mut |row| {
                assert_eq!(row["agent"], name);
                assert_eq!(
                    row["agent_parameters"]["relabeled_proportion"]
                        .as_f64()
                        .unwrap()
                        .to_bits(),
                    proportion.to_bits()
                );
                let diagnostics = &row["replay_diagnostics"];
                let copies = &diagnostics["routes"]["relabeled"];
                let draws = diagnostics["draws"].as_u64().unwrap();
                let relabeled = copies["draws"].as_u64().unwrap();
                let realized = diagnostics["realized_relabeled_share"].as_f64().unwrap();
                assert_eq!(relabeled, diagnostics["relabeled"].as_u64().unwrap());
                assert_eq!(
                    draws,
                    relabeled + diagnostics["routes"]["original"]["draws"].as_u64().unwrap()
                );
                assert_eq!(
                    realized.to_bits(),
                    (relabeled as f64 / draws as f64).to_bits()
                );
                assert_eq!(row["learning_update_counts"]["full"], draws);
                assert_eq!(row["learning_update_counts"]["values_only"], 0);
                assert_eq!(draws, 3 * row["actual_steps"].as_u64().unwrap());
                assert_eq!(
                    row["match_formations_train"],
                    10 * row["actual_steps"].as_u64().unwrap()
                );
                let requested =
                    (relabeled + diagnostics["fallbacks"].as_u64().unwrap()) as f64 / draws as f64;
                assert!((requested - proportion).abs() < 0.04);
                if name == "acs2her_shortened" {
                    assert_eq!(copies["already_reached"], 0);
                    assert_eq!(copies["after_counterfactual_end"], 0);
                    assert!(diagnostics["fallbacks"].as_u64().unwrap() > 0);
                    assert!(realized > 0.0 && realized < proportion);
                } else {
                    assert_eq!(diagnostics["fallbacks"], 0);
                    assert!((realized - 0.5).abs() < 0.04);
                    assert!(copies["already_reached"].as_u64().unwrap() > 0);
                    assert!(copies["after_counterfactual_end"].as_u64().unwrap() > 0);
                }
            },
        );
    }
}

#[test]
fn parameterized_evaluation_preserves_learning_store_counters_and_streams() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    for variant in [
        acs2her::Preset::Acs2HerShortened {
            proportion: 1.0 / 3.0,
        },
        acs2her::Preset::Acs2HerShortened { proportion: 0.5 },
        acs2her::Preset::Acs2Her { proportion: 0.5 },
    ] {
        assert_protocol_evaluation_read_only(
            &distributions.protocol(),
            42,
            &[10, 100],
            || {
                EpisodicGoalAgent(acs2her::Agent::<8, 2, 10, _, _>::new(
                    preset.config(8),
                    episodic_configuration(preset),
                    variant,
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                    ChaChaRandomSource::from_seed_and_stream(42, 7),
                ))
            },
            |agent| agent.0.learner().snapshot(),
        );
    }
}

#[test]
fn cli_accepts_free_proportions_and_refuses_other_agents_and_invalid_values() {
    for (index, name, proportion, accepted) in [
        (0, "acs2her", "0.5", true),
        (1, "acs2her_shortened", "0.3333333333333333", true),
        (2, "acs2her", "0", true),
        (3, "acs2her_shortened", "1", true),
        (4, "acs2her_shortened", "0.731", true),
        (5, "acs2her_nontrivial", "0.5", false),
        (6, "acs2er_episodic", "0.5", false),
        (7, "acs2vcp_k2", "0.5", false),
        (8, "acs2her", "-0.001", false),
        (9, "acs2her_shortened", "1.001", false),
        (10, "acs2her_shortened", "NaN", false),
        (11, "acs2her", "inf", false),
    ] {
        let out = std::env::temp_dir().join(format!(
            "her-proportion-{}-{index}.jsonl",
            std::process::id()
        ));
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
            .args([
                "--task",
                "maze4",
                "--cap",
                "5",
                "--protocol",
                "maze-pairs-v1",
                "--agents",
                name,
                "--relabel-proportion",
                proportion,
                "--seeds",
                "42",
                "--targets",
                "20",
                "--evaluation-blocks",
                "validation",
                "--out",
            ])
            .arg(&out)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            accepted,
            "{name} {proportion}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        if accepted {
            let row: serde_json::Value =
                serde_json::from_str(std::fs::read_to_string(&out).unwrap().trim()).unwrap();
            assert_eq!(
                row["agent_parameters"]["relabeled_proportion"]
                    .as_f64()
                    .unwrap()
                    .to_bits(),
                proportion.parse::<f64>().unwrap().to_bits()
            );
            std::fs::remove_file(out).unwrap();
        } else {
            assert!(!out.exists());
        }
    }
}
