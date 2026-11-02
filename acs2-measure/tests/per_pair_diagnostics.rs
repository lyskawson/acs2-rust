use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::{geometries::MazeGeometry, topology::MazeTopology};
use acs2_envs::roles::ResearchTask;
use acs2_measure::first_decision::profile;
use acs2_measure::maze_truth::{discounted_value, pair_truth};
use acs2_measure::pair_protocol::{registered_configuration, MazePairDistributions};
use acs2_measure::runner::{run_protocol_with_sink, AgentKind, Preset, RunMetadata, RunSettings};
use acs2_measure::task::Task;
use serde_json::Value;

fn bits(value: &Value) -> Value {
    match value {
        Value::Number(number) if number.is_f64() => {
            Value::String(format!("{:016x}", number.as_f64().unwrap().to_bits()))
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

fn deterministic(mut row: Value) -> Value {
    for name in [
        "wall_seconds_train",
        "wall_seconds_eval",
        "wall_seconds_total",
    ] {
        row.as_object_mut().unwrap().remove(name);
    }
    for block in row["evaluation_blocks"]
        .as_object_mut()
        .unwrap()
        .values_mut()
    {
        block["starts"] = Value::Null;
    }
    bits(&row)
}

#[test]
fn recording_all_five_arms_changes_only_pair_records_and_clock_times_in_binary64_bits() {
    let ResearchTask::GoalMaze(geometry) = ResearchTask::named("maze4").unwrap() else {
        unreachable!()
    };
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", geometry, 5, "coordinates");
    for arm in [
        "acs2",
        "acs2er_episodic",
        "acs2her",
        "acs2her_nontrivial",
        "acs2her_shortened",
    ] {
        let kind = AgentKind::from_run_name(arm, 2.0 / 3.0).unwrap();
        let run = |record_starts| {
            let mut rows = Vec::new();
            let final_state = run_protocol_with_sink::<_, _, 8, 2, 10>(
                &distributions.protocol(),
                kind,
                42,
                &[30, 60],
                &RunMetadata {
                    record_starts,
                    ..RunMetadata::test("test")
                },
                RunSettings {
                    evaluate_points: true,
                    capture_final_state: true,
                },
                &mut |row| rows.push(row),
            );
            (rows, final_state)
        };
        let with = run(true);
        let without = run(false);
        assert_eq!(with.1, without.1);
        for (with, without) in with.0.into_iter().zip(without.0) {
            for block in with["evaluation_blocks"].as_object().unwrap().values() {
                for record in block["starts"].as_array().unwrap() {
                    assert!(record["first_decision"]["random_fallback"].is_boolean());
                }
            }
            assert_eq!(deterministic(with), deterministic(without), "{arm}");
        }
    }
}

#[test]
fn winner_profiles_use_numerosity_ignore_null_effects_and_preserve_same_action_ties() {
    let config = Preset::thesis().config(8);
    let state = Perception::new([Symbol::Token(1); 10]);
    let mut rule = Classifier::general(Some(2), &config);
    rule.effect.symbols[0] = Symbol::Token(2);
    rule.q = 0.5;
    rule.r = 8.0;
    rule.num = 3;
    rule.exp = 17;
    rule.condition.symbols[8] = Symbol::Token(1);
    rule.condition.symbols[0] = Symbol::Token(1);
    let mut tied = rule.clone();
    tied.q = 1.0;
    tied.r = 12.0;
    tied.num = 1;
    tied.condition.symbols[8] = Symbol::Wildcard;
    let mut other_action = tied.clone();
    other_action.action = Some(4);
    let mut lower = tied.clone();
    lower.r = 11.0;
    let mut no_change = Classifier::general(Some(2), &config);
    no_change.r = 10000.0;
    let mut unmatched = tied.clone();
    unmatched.condition.symbols[0] = Symbol::Token(3);
    unmatched.r = 10000.0;
    let population =
        Population::from_classifiers(vec![rule, tied, other_action, lower, no_change, unmatched]);
    let observed = profile::<8, 2, 10>(&population, &state, 2);
    assert_eq!(observed["random_fallback"], false);
    let winners = observed["winners"].as_array().unwrap();
    assert_eq!(winners.len(), 2);
    assert_eq!(winners[0]["specified_goal_attributes"], 1);
    assert_eq!(winners[1]["specified_goal_attributes"], 0);
    assert_eq!(winners[0]["specified_state_attributes"], 1);
    assert_eq!(winners[0]["fitness"], 4.0);
    assert_eq!(winners[0]["selection_score"], 12.0);
    assert_eq!(winners[0]["experience"], 17);
    assert_eq!(winners[0]["q"], 0.5);
    assert_eq!(winners[0]["r"], 8.0);
}

#[test]
fn random_fallback_requires_no_matching_change_rule() {
    let config = Preset::thesis().config(8);
    let population =
        Population::from_classifiers(vec![Classifier::<10>::general(Some(1), &config)]);
    let result = profile::<8, 2, 10>(&population, &Perception::new([Symbol::Token(1); 10]), 1);
    assert_eq!(result["random_fallback"], true);
    assert!(result["winners"].as_array().unwrap().is_empty());
}

static CORRIDOR: MazeGeometry = MazeGeometry {
    id: "diagnostic-corridor",
    matrix: &[
        &[1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 9, 1],
        &[1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 5,
    source: acs2_envs::maze::geometries::MazeSource::Pyalcs,
};

static TWO_ROUTES: MazeGeometry = MazeGeometry {
    id: "diagnostic-two-routes",
    matrix: &[
        &[1, 1, 1, 1, 1],
        &[1, 0, 9, 0, 1],
        &[1, 0, 1, 0, 1],
        &[1, 0, 1, 0, 1],
        &[1, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1],
    ],
    max_episode_steps: 5,
    source: acs2_envs::maze::geometries::MazeSource::Pyalcs,
};

#[test]
fn corridor_q_values_include_wall_bumps_successor_discount_and_remaining_cap() {
    let topology = MazeTopology::new(&CORRIDOR).unwrap();
    let truth = pair_truth(&topology, (1, 2), (1, 4), 3);
    assert_eq!(truth.shortest_distance, Some(2));
    assert_eq!(truth.value, 950.0);
    assert_eq!(truth.successors[0], (1, 2));
    assert_eq!(truth.action_values[0], 902.5);
    assert_eq!(truth.action_values[2], 950.0);
    assert_eq!(truth.optimal_actions, vec![2]);
    assert_eq!(
        pair_truth(&topology, (1, 2), (1, 4), 2).action_values[0],
        0.0
    );
    let no_time = pair_truth(&topology, (1, 2), (1, 4), 1);
    assert_eq!(no_time.value, 0.0);
    assert_eq!(no_time.optimal_actions, (0..8).collect::<Vec<_>>());
    assert_eq!(
        pair_truth(&topology, (1, 3), (1, 4), 1).action_values[2],
        1000.0
    );
    assert_eq!(
        pair_truth(&topology, (1, 3), (1, 4), 0).action_values[2],
        0.0
    );
}

#[test]
fn equally_short_routes_retain_every_optimal_action() {
    let topology = MazeTopology::new(&TWO_ROUTES).unwrap();
    let truth = pair_truth(&topology, (4, 2), (1, 2), 4);
    assert_eq!(truth.shortest_distance, Some(3));
    assert_eq!(truth.optimal_actions, vec![1, 7]);
    for action in &truth.optimal_actions {
        assert_eq!(truth.action_values[*action], 902.5);
    }
}

#[test]
fn environment_truth_matches_every_registered_pair_distance_on_all_six_mazes() {
    for name in ["maze4", "maze5", "maze6", "maze7", "woods14", "mazema"] {
        let ResearchTask::GoalMaze(geometry) = ResearchTask::named(name).unwrap() else {
            unreachable!()
        };
        let cap = registered_configuration(name)["cap"].as_u64().unwrap() as u32;
        let distributions =
            MazePairDistributions::<Coordinates, 2>::registered(name, geometry, cap, "coordinates");
        for block in distributions.protocol().evaluations {
            for (start, goal, _) in <_ as Task<8, 2, 10>>::pairs(block.task) {
                let target = match goal.symbols {
                    [Symbol::Token(row), Symbol::Token(col)] => (row as usize, col as usize),
                    _ => unreachable!(),
                };
                let truth = pair_truth(block.task.maze.template.topology(), start, target, cap);
                let distance = <_ as Task<8, 2, 10>>::distance(block.task, start, &goal);
                assert_eq!(truth.shortest_distance, distance);
                assert_eq!(
                    truth.value.to_bits(),
                    discounted_value(distance, cap).to_bits()
                );
                if distance.unwrap() > cap {
                    assert_eq!(truth.value, 0.0);
                    assert_eq!(truth.optimal_actions.len(), 8);
                }
            }
        }
    }
}
