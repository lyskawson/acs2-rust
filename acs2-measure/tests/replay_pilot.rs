use std::collections::BTreeSet;

use acs2_envs::goal::maze::Coordinates;
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{
    run_protocol_with_replay_plan, AgentKind, ReplayRunSettings, RunMetadata, RunSettings,
};
use serde_json::Value;

fn floating_bits(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            Value::String(format!("f64:{:016x}", number.as_f64().unwrap().to_bits()))
        }
        Value::Array(values) => Value::Array(values.iter().map(floating_bits).collect()),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), floating_bits(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn reproduce_pilot_fixture(passes: bool) {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/replay_schedule_and_order_regression.json"
    ))
    .unwrap();
    let configurations = fixture["configurations"].as_array().unwrap();
    let expected: BTreeSet<_> = ["maze4", "maze5", "maze6", "maze7", "woods14", "mazema"]
        .into_iter()
        .flat_map(|task| {
            [
                ("acs2her_shortened", "3to1at30000"),
                ("acs2her_shortened", "3to1at100000"),
                ("acs2her_shortened", "3to1at300000"),
                ("acs2her_shortened", "backward"),
                ("acs2her_shortened", "random-pass"),
                ("acs2er_episodic", "3to1at30000"),
                ("acs2er_episodic", "3to1at100000"),
                ("acs2er_episodic", "3to1at300000"),
                ("acs2er_episodic", "backward"),
            ]
            .map(move |(agent, mode)| (task, agent, mode))
        })
        .collect();
    let actual: BTreeSet<_> = configurations
        .iter()
        .map(|config| {
            (
                config["task"].as_str().unwrap(),
                config["agent"].as_str().unwrap(),
                config["mode"].as_str().unwrap(),
            )
        })
        .collect();
    assert_eq!(configurations.len(), 54);
    assert_eq!(actual, expected);
    for config in configurations.iter().filter(|config| {
        config["mode"].as_str().unwrap().starts_with("3to1") != passes
    }) {
        let task = config["task"].as_str().unwrap();
        let acs2_envs::roles::ResearchTask::GoalMaze(geometry) =
            acs2_envs::roles::ResearchTask::named(task).unwrap()
        else {
            unreachable!()
        };
        let distributions = MazePairDistributions::<Coordinates, 2>::registered(
            task,
            geometry,
            config["cap"].as_u64().unwrap() as u32,
            "coordinates",
        );
        let mode = config["mode"].as_str().unwrap();
        let replay = acs2_measure::replay_plan::parse(
            if mode.starts_with("3to1") {
                mode
            } else {
                "constant"
            },
            if mode.starts_with("3to1") {
                "independent"
            } else {
                mode
            },
        );
        let kind = AgentKind::from_run_name(
            config["agent"].as_str().unwrap(),
            config["proportion"].as_f64().unwrap(),
        )
        .unwrap();
        let mut rows = Vec::new();
        run_protocol_with_replay_plan::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[1000, 3000],
            &RunMetadata {
                record_starts: false,
                ..RunMetadata::test("test")
            },
            ReplayRunSettings {
                measurement: RunSettings {
                    evaluate_points: true,
                    capture_final_state: false,
                },
                replay,
            },
            &mut |row| rows.push(row),
        );
        let pinned = config["rows"].as_array().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(pinned.len(), 2);
        for (mut row, reference) in rows.into_iter().zip(pinned) {
            for key in fixture["excluded"].as_array().unwrap() {
                row.as_object_mut().unwrap().remove(key.as_str().unwrap());
            }
            assert_eq!(
                floating_bits(&row),
                *reference,
                "{} {} at {}",
                task,
                config["arm"],
                row["nominal_step"]
            );
        }
    }
}

#[test]
fn schedule_settings_reproduce_the_original_six_maze_seed42_pilot_bit_for_bit() {
    reproduce_pilot_fixture(false);
}

#[test]
fn episode_pass_settings_reproduce_the_corrected_six_maze_seed42_pilot_bit_for_bit() {
    reproduce_pilot_fixture(true);
}
