use acs2_agents::acs2her;
use acs2_agents::episodic_replay::{
    EpisodicAgent, EpisodicReplay, ReplayOrder, ReplayPlan, ReplaySchedule,
};
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_protocol_with_replay_plan, AgentKind, GoalAgent,
    Preset, ReplayRunSettings, RunMetadata, RunSettings,
};

fn plans() -> Vec<ReplayPlan> {
    ["constant", "3to1at30000", "3to1at100000", "3to1at300000"]
        .into_iter()
        .map(|schedule| acs2_measure::replay_plan::parse(schedule, "independent"))
        .chain(
            ["backward", "random-pass"]
                .into_iter()
                .map(|order| acs2_measure::replay_plan::parse("constant", order)),
        )
        .collect()
}

#[test]
fn every_cli_setting_maps_to_exact_axes_and_unregistered_or_combined_settings_fail() {
    assert_eq!(
        plans(),
        vec![
            ReplayPlan::default(),
            ReplayPlan {
                schedule: ReplaySchedule::ThreeToOne {
                    switch_steps: 30_000
                },
                order: ReplayOrder::Independent
            },
            ReplayPlan {
                schedule: ReplaySchedule::ThreeToOne {
                    switch_steps: 100_000
                },
                order: ReplayOrder::Independent
            },
            ReplayPlan {
                schedule: ReplaySchedule::ThreeToOne {
                    switch_steps: 300_000
                },
                order: ReplayOrder::Independent
            },
            ReplayPlan {
                schedule: ReplaySchedule::Constant,
                order: ReplayOrder::Backward
            },
            ReplayPlan {
                schedule: ReplaySchedule::Constant,
                order: ReplayOrder::RandomPass
            }
        ]
    );
    for (schedule, order) in [
        ("3to1at40000", "independent"),
        ("constant", "forward"),
        ("3to1at30000", "backward"),
        ("3to1at100000", "random-pass"),
    ] {
        assert!(
            std::panic::catch_unwind(|| acs2_measure::replay_plan::parse(schedule, order)).is_err()
        );
    }
}

#[test]
fn library_replay_plans_preserve_copy_rules_for_supported_pass_admissibility() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    for name in [
        "acs2er_episodic",
        "acs2her",
        "acs2her_nontrivial",
        "acs2her_shortened",
        "acs2her_values",
        "acs2her_values_shortened",
    ] {
        for plan in plans() {
            if name == "acs2her_nontrivial" && plan.order != ReplayOrder::Independent {
                continue;
            }
            run_protocol_with_replay_plan::<_, _, 8, 2, 10>(
                &distributions.protocol(),
                AgentKind::from_run_name(name, 0.5).unwrap(),
                42,
                &[100],
                &RunMetadata {
                    record_starts: false,
                    ..RunMetadata::test("test")
                },
                ReplayRunSettings {
                    measurement: RunSettings {
                        evaluate_points: true,
                        capture_final_state: false,
                    },
                    replay: plan,
                },
                &mut |row| {
                    assert_eq!(row["agent"], name);
                    let p = &row["agent_parameters"];
                    if plan == ReplayPlan::default() {
                        assert!(
                            p.get("replay_order").is_none() && p.get("replay_schedule").is_none()
                        );
                    } else {
                        assert!(
                            p.get("replay_order").is_some() && p.get("replay_schedule").is_some()
                        );
                    }
                    if plan.order != ReplayOrder::Independent {
                        assert_eq!(p["draws"], "uniform_episode_passes");
                    }
                    assert_eq!(
                        p["copy_update"],
                        if name.contains("values") {
                            "ValuesOnly"
                        } else {
                            "Full"
                        }
                    );
                    let copies = &row["replay_diagnostics"]["routes"]["relabeled"];
                    if name.contains("shortened") {
                        assert_eq!(copies["already_reached"], 0);
                        assert_eq!(copies["after_counterfactual_end"], 0);
                        assert_eq!(p["admissibility"], "CounterfactualEpisode");
                    }
                    if plan.order != ReplayOrder::Independent {
                        assert_eq!(p["strategy"], "episode");
                        assert_eq!(
                            row["replay_diagnostics"]["passes"]["emitted_steps"],
                            row["replay_updates"]
                        );
                    } else {
                        assert_eq!(p["strategy"], "future");
                    }
                    assert_eq!(
                        row["replay_updates"],
                        3 * row["actual_steps"].as_u64().unwrap()
                    );
                    assert_eq!(
                        p["relabeled_proportion"],
                        if name == "acs2er_episodic" {
                            0.0
                        } else if name == "acs2her_nontrivial" {
                            2.0 / 3.0
                        } else {
                            0.5
                        }
                    );
                },
            );
        }
    }
}

#[test]
fn scheduled_er_and_shortened_her_both_switch_after_the_registered_boundary() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let plan = acs2_measure::replay_plan::parse("3to1at30000", "independent");
    for name in ["acs2er_episodic", "acs2her_shortened"] {
        run_protocol_with_replay_plan::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            AgentKind::from_run_name(name, 0.5).unwrap(),
            42,
            &[30_000, 30_100],
            &RunMetadata::test("test"),
            ReplayRunSettings {
                measurement: RunSettings {
                    evaluate_points: true,
                    capture_final_state: false,
                },
                replay: plan,
            },
            &mut |row| {
                let clock = &row["replay_diagnostics"]["draw_schedule"];
                let early = clock["early_steps"].as_u64().unwrap();
                let late = clock["late_steps"].as_u64().unwrap();
                assert_eq!(early + late, row["actual_steps"]);
                assert!(early <= 30_000);
                assert_eq!(row["replay_updates"], 3 * early + late);
                if row["nominal_step"] == 30_100 {
                    assert!(late >= 100);
                }
            },
        );
    }
}

struct GreedyOnly;
impl RandomSource for GreedyOnly {
    fn gen_bool(&mut self, _: f64) -> bool {
        panic!("evaluation drew an exploration coin")
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("evaluation drew an exploration variate")
    }
    fn gen_range(&mut self, _: usize) -> usize {
        0
    }
}

#[test]
fn evaluation_is_read_only_and_greedy_for_every_new_setting_at_epsilon_point_eight() {
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    for plan in plans() {
        assert_protocol_evaluation_read_only(
            &distributions.protocol(),
            42,
            &[10, 100],
            || {
                let mut agent = acs2her::Agent::<8, 2, 10, _, _>::new(
                    preset.config(8),
                    episodic_configuration(preset),
                    acs2her::Preset::Acs2HerShortened { proportion: 0.5 },
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                    ChaChaRandomSource::from_seed_and_stream(42, 7),
                );
                agent
                    .learner_mut()
                    .set_replay_plan(plan, acs2_measure::replay_plan::order_rng(plan, 42));
                EpisodicGoalAgent(agent)
            },
            |agent| agent.0.learner().snapshot(),
        );
        let config = preset.config(2);
        assert_eq!(config.epsilon, 0.8);
        let mut best = Classifier::<2>::general(Some(1), &config);
        best.effect.symbols[0] = Symbol::Token(1);
        best.r = 100.0;
        let mut agent = acs2her::Agent::<1, 1, 2, _, _>::new(
            config.clone(),
            episodic_configuration(preset),
            acs2her::Preset::Acs2HerShortened { proportion: 0.5 },
            ChaChaRandomSource::from_seed(42),
            ChaChaRandomSource::from_seed(42),
        );
        *agent.learner_mut() = EpisodicReplay::new(
            config,
            episodic_configuration(preset),
            agent.learner().sampler().configuration(),
            agent.learner().copy_update(),
            ChaChaRandomSource::from_seed(42),
            ChaChaRandomSource::from_seed(42),
        )
        .with_population(Population::from_classifiers(vec![best]))
        .with_replay_plan(plan, acs2_measure::replay_plan::order_rng(plan, 42));
        let before = agent.learner().snapshot();
        let measured = EpisodicGoalAgent(agent);
        for _ in 0..20 {
            assert_eq!(
                measured
                    .eval_action(
                        &Perception::new([Symbol::Token(0), Symbol::Token(3)]),
                        &mut GreedyOnly
                    )
                    .0,
                1
            );
        }
        assert_eq!(measured.0.learner().snapshot(), before);
    }
}

#[test]
fn cli_records_the_exact_axis_and_rejects_unsupported_agents_and_combinations() {
    for (index, name, schedule, order, accepted) in [
        (0, "acs2er_episodic", "3to1at30000", "independent", true),
        (1, "acs2her_shortened", "constant", "backward", true),
        (2, "acs2her_shortened", "constant", "random-pass", true),
        (3, "acs2", "constant", "backward", false),
        (4, "acs2er", "3to1at100000", "independent", false),
        (5, "acs2vcp_k2", "constant", "backward", false),
        (6, "acs2her", "3to1at300000", "backward", false),
        (7, "acs2her", "3to1at40000", "independent", false),
    ] {
        let out =
            std::env::temp_dir().join(format!("replay-axes-{}-{index}.jsonl", std::process::id()));
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
                "--seeds",
                "42",
                "--targets",
                "20",
                "--evaluation-blocks",
                "validation",
                "--replay-schedule",
                schedule,
                "--replay-order",
                order,
                "--out",
            ])
            .arg(&out)
            .output()
            .unwrap();
        assert_eq!(
            output.status.success(),
            accepted,
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        if accepted {
            let row: serde_json::Value =
                serde_json::from_str(std::fs::read_to_string(&out).unwrap().trim()).unwrap();
            assert_eq!(row["agent"], name);
            assert_eq!(
                row["agent_parameters"]["replay_order"],
                order.replace('-', "_")
            );
            assert_eq!(
                row["agent_parameters"]["replay_schedule"]["kind"],
                if schedule == "constant" {
                    "constant"
                } else {
                    "three_to_one"
                }
            );
            if schedule != "constant" {
                assert_eq!(
                    row["agent_parameters"]["replay_schedule"]["switch_steps"],
                    30_000
                );
            }
            std::fs::remove_file(out).unwrap();
        } else {
            assert!(!out.exists());
        }
    }
}

const UNMEASURED_AGENTS: &[&str] = &[
    "acs2",
    "acs2er",
    "acs2er_one_update",
    "acs2her",
    "acs2her_nontrivial",
    "acs2her_values",
    "acs2her_values_shortened",
    "acs2vcp_k2",
    "acs2vcp_k4",
    "acs2vcp_k2_uniform",
    "acs2vcp_k4_uniform",
    "acs2her_shortened,acs2her",
    "acs2er_episodic,acs2her_nontrivial",
];

fn assert_cli_refuses_replay_axis(agent: &str, flag: &str, value: &str) {
    let out = std::env::temp_dir().join(format!(
        "replay-agent-refusal-{}-{agent}-{flag}-{value}.jsonl",
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
            agent,
            "--seeds",
            "42",
            "--targets",
            "20",
            "--evaluation-blocks",
            "validation",
            flag,
            value,
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    let created_output = out.exists();
    if created_output {
        std::fs::remove_file(&out).unwrap();
    }
    let message = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "accepted {agent} {flag} {value}");
    assert!(
        message.contains("--replay-schedule and --replay-order are supported only for acs2her_shortened and acs2er_episodic"),
        "{agent} {flag} {value}: {message}"
    );
    assert!(!created_output, "refused axes must not create a row file");
}

#[test]
fn cli_refuses_unmeasured_agents_with_replay_schedules() {
    for agent in UNMEASURED_AGENTS {
        for schedule in ["constant", "3to1at30000", "3to1at100000", "3to1at300000"] {
            assert_cli_refuses_replay_axis(agent, "--replay-schedule", schedule);
        }
    }
}

#[test]
fn cli_refuses_unmeasured_agents_with_each_replay_order() {
    for agent in UNMEASURED_AGENTS {
        for order in ["independent", "backward", "random-pass"] {
            assert_cli_refuses_replay_axis(agent, "--replay-order", order);
        }
    }
}
