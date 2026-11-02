use std::collections::BTreeSet;

use acs2_core::representation::symbol::Symbol;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::roles::ResearchTask;
use acs2_measure::maze_truth::{discounted_value, pair_truth};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{run_protocol_with_sink, AgentKind, RunMetadata, RunSettings};
use acs2_measure::task::Task;
use serde_json::Value;

fn bits(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            Value::String(format!("f64:{:016x}", number.as_f64().unwrap().to_bits()))
        }
        Value::Array(values) => values.iter().map(bits).collect(),
        Value::Object(values) => Value::Object(
            values
                .iter()
                .map(|(key, value)| (key.clone(), bits(value)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

fn distributions(name: &str, cap: u32) -> MazePairDistributions<Coordinates, 2> {
    let ResearchTask::GoalMaze(geometry) = ResearchTask::named(name).unwrap() else {
        unreachable!()
    };
    MazePairDistributions::registered(name, geometry, cap, "coordinates")
}

#[test]
fn thirty_pilot_slices_reproduce_every_shared_field_and_first_last_pair_bits() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/per_pair_diagnostics_regression.json"
    ))
    .unwrap();
    let configs = fixture["configurations"].as_array().unwrap();
    assert_eq!(configs.len(), 30);
    let mut cells = BTreeSet::new();
    for config in configs {
        let name = config["task"].as_str().unwrap();
        let arm = config["agent"].as_str().unwrap();
        assert!(cells.insert((name, arm)));
        let distributions = distributions(name, config["cap"].as_u64().unwrap() as u32);
        let kind = AgentKind::from_run_name(arm, 2.0 / 3.0).unwrap();
        let mut rows = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            kind,
            42,
            &[config["nominal_step"].as_u64().unwrap()],
            &RunMetadata {
                record_starts: true,
                ..RunMetadata::test("test")
            },
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        assert_eq!(rows.len(), 1);
        let mut row = rows.pop().unwrap();
        for (block, records) in config["records"].as_object().unwrap() {
            for sample in records.as_array().unwrap() {
                let index = sample["index"].as_u64().unwrap() as usize;
                assert_eq!(
                    bits(&row["evaluation_blocks"][block]["starts"][index]),
                    sample["record"],
                    "{name} {arm} {block} pair {index}"
                );
            }
        }
        for block in row["evaluation_blocks"]
            .as_object_mut()
            .unwrap()
            .values_mut()
        {
            block["starts"] = Value::Null;
        }
        for excluded in fixture["excluded"].as_array().unwrap() {
            row.as_object_mut()
                .unwrap()
                .remove(excluded.as_str().unwrap());
        }
        let mut expected = config["shared_fields"].clone();
        expected["goal_protocol"] = fixture["goal_protocols"][name].clone();
        assert_eq!(bits(&row), expected, "{name} {arm}");
    }
    assert_eq!(cells.len(), 30);
}

#[test]
fn every_pair_of_six_archived_rows_matches_capped_environment_ground_truth() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../fixtures/per_pair_diagnostics_regression.json"
    ))
    .unwrap();
    let configs = fixture["distance_rows"].as_array().unwrap();
    assert_eq!(configs.len(), 6);
    let mut outside_cap = 0;
    let mut checked = 0;
    for config in configs {
        let name = config["task"].as_str().unwrap();
        let cap = config["cap"].as_u64().unwrap() as u32;
        let distributions = distributions(name, cap);
        for block in distributions.protocol().evaluations {
            let pairs = <_ as Task<8, 2, 10>>::pairs(block.task);
            let distances = config["distances"][block.name].as_array().unwrap();
            assert_eq!(distances.len(), pairs.len());
            for ((start, goal, _), recorded) in pairs.into_iter().zip(distances) {
                let target = match goal.symbols {
                    [Symbol::Token(row), Symbol::Token(col)] => (row as usize, col as usize),
                    _ => unreachable!(),
                };
                let truth = pair_truth(block.task.maze.template.topology(), start, target, cap);
                let distance = recorded.as_u64().unwrap() as u32;
                assert_eq!(truth.shortest_distance, Some(distance));
                assert_eq!(
                    truth.value.to_bits(),
                    discounted_value(Some(distance), cap).to_bits()
                );
                if distance > cap {
                    outside_cap += 1;
                    assert_eq!(truth.value, 0.0);
                    assert_eq!(truth.optimal_actions.len(), 8);
                } else {
                    let expected = (1..distance).fold(1000.0_f64, |value, _| value * 0.95);
                    assert_eq!(truth.value.to_bits(), expected.to_bits());
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 3000);
    assert!(outside_cap > 0);
}
