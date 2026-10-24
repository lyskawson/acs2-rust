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
fn restricted_hindsight_rows_record_fixed_full_updates_and_zero_trivial_copies() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for (name, admissibility) in [
        ("acs2her_nontrivial", "FromNonGoalState"),
        ("acs2her_shortened", "CounterfactualEpisode"),
    ] {
        let kind = AgentKind::from_run_name(name, 0.9).unwrap();
        assert_eq!(kind.name(), name);
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
            &mut |row| {
                let parameters = &row["agent_parameters"];
                assert_eq!(parameters["admissibility"], admissibility);
                assert_eq!(parameters["copy_update"], "Full");
                assert_eq!(
                    parameters["relabeled_proportion"]
                        .as_f64()
                        .unwrap()
                        .to_bits(),
                    (2.0_f64 / 3.0).to_bits()
                );
                let steps = row["actual_steps"].as_u64().unwrap();
                assert_eq!(row["learning_update_counts"]["full"], 3 * steps);
                assert_eq!(row["learning_update_counts"]["values_only"], 0);
                assert_eq!(row["match_formations_train"], 10 * steps);
                let diagnostics = &row["replay_diagnostics"];
                let copies = &diagnostics["routes"]["relabeled"];
                assert!(copies["draws"].as_u64().unwrap() > 0);
                assert_eq!(copies["already_reached"], 0);
                if name == "acs2her_shortened" {
                    assert_eq!(copies["after_counterfactual_end"], 0);
                }
                assert!(diagnostics["fallbacks"].as_u64().unwrap() > 0);
                let walls = &diagnostics["wall_bumps"];
                assert_eq!(walls["relabeled_draws"], copies["draws"]);
                assert!(walls["unchanged_state"].as_u64().unwrap() > 0);
                assert_eq!(walls["unchanged_state_terminal"], 0);
                assert_eq!(walls["proposals_equal_draws"], true);
            },
        );
    }
}

#[test]
fn restricted_hindsight_evaluation_preserves_counters_store_and_both_random_streams() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    for variant in [
        acs2her::Preset::Acs2HerNontrivial,
        acs2her::Preset::Acs2HerShortened,
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
fn restricted_hindsight_cli_rejects_proportion_tuning() {
    for name in ["acs2her_nontrivial", "acs2her_shortened"] {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
            .args([
                "--task",
                "maze4",
                "--cap",
                "5",
                "--agents",
                name,
                "--relabel-proportion",
                "0.3333333333333333",
                "--out",
                "/tmp/unused-admissibility.jsonl",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("only values variants"));
    }
}
