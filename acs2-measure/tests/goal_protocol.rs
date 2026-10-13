use std::collections::BTreeSet;

use acs2_core::checkpoint::Checkpointed;
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::perception::Perception;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::geometry_by_id;
use acs2_envs::maze::topology::MazeTopology;
use acs2_measure::goal_protocol::{
    goal_cells, registered_configuration, validate_partition, MazeGoalDistributions, MAZE_PROTOCOL,
};
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_goal_agent_protocol_with_sink,
    run_protocol_with_sink, run_with_sink, AgentKind, CoreAgent, GoalAgent, MeasuredEnvironment,
    Preset, RunMetadata, RunSettings,
};
use acs2_measure::task::{MazeTask, Task};
use serde_json::{json, Value};

fn distributions(name: &str) -> MazeGoalDistributions<Coordinates, 2> {
    let config = registered_configuration(name);
    let geometry = geometry_by_id(config["geometry"].as_str().unwrap()).unwrap();
    MazeGoalDistributions::registered(
        name,
        geometry,
        config["cap"].as_u64().unwrap() as u32,
        "coordinates",
    )
}

fn names() -> [&'static str; 6] {
    ["maze4", "maze5", "maze6", "maze7", "woods14", "mazema"]
}

#[test]
fn registered_goal_pools_are_disjoint_complete_and_span_difficulty() {
    for name in names() {
        let distributions = distributions(name);
        let config = &distributions.registration;
        let topology = distributions.training.template.topology();
        let all: BTreeSet<_> = topology.walkable_cells().iter().copied().collect();
        let mut union = BTreeSet::new();
        for block in ["training", "validation", "test"] {
            let pool: BTreeSet<_> = goal_cells(config, block).into_iter().collect();
            assert!(union.is_disjoint(&pool));
            union.extend(&pool);
            assert!(pool.len() >= 3);
            for stratum in 0..3 {
                assert!(config["goal_difficulty"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|goal| {
                        goal["stratum"] == stratum
                            && pool.contains(&(
                                goal["goal"][0].as_u64().unwrap() as usize,
                                goal["goal"][1].as_u64().unwrap() as usize,
                            ))
                    }));
            }
        }
        assert_eq!(all, union);
        assert_eq!(registered_configuration(name), *config);
    }
    assert!(serde_json::from_str::<Value>(MAZE_PROTOCOL).is_ok());
}

#[test]
fn overlapping_or_incomplete_goal_partitions_are_refused() {
    let config = registered_configuration("woods14");
    let geometry = geometry_by_id("Woods14-v0").unwrap();
    let mut overlap = config.clone();
    overlap["blocks"]["test"]["goals"][0] = config["blocks"]["training"]["goals"][0].clone();
    assert!(std::panic::catch_unwind(|| validate_partition(geometry, &overlap)).is_err());
    let mut missing = config;
    missing["blocks"]["training"]["goals"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(std::panic::catch_unwind(|| validate_partition(geometry, &missing)).is_err());
}

#[derive(Default)]
struct GoalExposureProbe {
    received: Vec<Goal<2>>,
    environment_pool: Vec<Goal<2>>,
    desired: Vec<Goal<2>>,
    starts: Vec<Goal<2>>,
    steps: u64,
}

impl GoalAgent<8, 2, 10> for GoalExposureProbe {
    fn name(&self) -> &'static str {
        "goal_exposure_probe"
    }
    fn register_training_goals(&mut self, goals: &[Goal<2>]) {
        self.received = goals.to_vec();
    }
    fn train_episode<E: MeasuredEnvironment<8, 2, 10>>(&mut self, env: &mut E, _time: u64) {
        self.environment_pool = env.training_goals().unwrap().to_vec();
        let joined = env.reset();
        let start = env.episode_start().unwrap();
        assert_eq!(GoalLayout::<8, 2, 10>::split(&joined).1, start.desired);
        assert!(self.received.contains(&start.desired));
        self.desired.push(start.desired);
        self.starts.push(start.achieved);
        loop {
            let outcome = env.step(0);
            self.steps += 1;
            assert_eq!(
                GoalLayout::<8, 2, 10>::split(&outcome.observation).1,
                start.desired
            );
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
    }
    fn declared_policy(&self) -> &'static str {
        "action_zero"
    }
    fn eval_action(&self, _state: &Perception<10>, _rng: &mut dyn RandomSource) -> (usize, f64) {
        (0, 0.0)
    }
    fn online_updates(&self) -> u64 {
        self.steps
    }
    fn replay_updates(&self) -> u64 {
        0
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn population_classifiers(&self) -> usize {
        0
    }
    fn population_numerosity(&self) -> u32 {
        0
    }
    fn population_logical_bytes(&self) -> usize {
        0
    }
    fn population_mark_entries(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn agent_parameters(&self) -> Value {
        json!({})
    }
}

#[test]
fn agent_registration_and_training_resets_expose_only_training_goals() {
    for name in names() {
        let distributions = distributions(name);
        let mut agent = GoalExposureProbe::default();
        run_goal_agent_protocol_with_sink::<_, _, _, 8, 2, 10>(
            &distributions.protocol(),
            &mut agent,
            42,
            &[2_000],
            &RunMetadata::test("test"),
            false,
            &mut |_| {},
        );
        let expected: Vec<_> = goal_cells(&distributions.registration, "training")
            .into_iter()
            .map(|cell| distributions.training.template.goal_at(cell))
            .collect();
        assert_eq!(agent.received, expected);
        assert_eq!(agent.environment_pool, expected);
        assert!(agent.desired.iter().all(|goal| expected.contains(goal)));
        let observed: BTreeSet<_> = agent.desired.iter().copied().collect();
        assert_eq!(observed.len(), expected.len());
        assert!(agent
            .starts
            .iter()
            .zip(&agent.desired)
            .all(|(start, goal)| start != goal));
        for withheld in distributions
            .validation
            .template
            .goal_pool()
            .iter()
            .chain(distributions.test.template.goal_pool())
        {
            assert!(!agent.received.contains(withheld));
            assert!(agent.starts.contains(withheld));
        }
    }
}

#[test]
fn every_evaluation_block_uses_its_registered_goals_weights_floor_and_ceiling() {
    for name in names() {
        let distributions = distributions(name);
        let mut rows = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            AgentKind::Acs2,
            42,
            &[50],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        let row = &rows[0];
        assert_eq!(row["schema"], 5);
        assert!(row.get("success").is_none());
        assert_eq!(row["evaluation_blocks"].as_object().unwrap().len(), 3);
        for (block, task) in [
            ("training", &distributions.training),
            ("validation", &distributions.validation),
            ("test", &distributions.test),
        ] {
            let pairs = <MazeTask<Coordinates, 2> as Task<8, 2, 10>>::pairs(task);
            let evaluated = &row["evaluation_blocks"][block];
            let registered = &distributions.registration["blocks"][block];
            assert_eq!(
                evaluated["evaluation_pairs"],
                registered["evaluation_pairs"]
            );
            assert_eq!(
                evaluated["evaluation_distribution"],
                "exhaustive_exact_weights"
            );
            assert_eq!(evaluated["evaluation_standard_error"], 0.0);
            assert!(
                (evaluated["random_floor"].as_f64().unwrap()
                    - registered["random_floor"].as_f64().unwrap())
                .abs()
                    < 1e-14
            );
            assert!(
                (evaluated["reachable_within_cap"].as_f64().unwrap()
                    - registered["reachable_within_cap"].as_f64().unwrap())
                .abs()
                    < 1e-14
            );
            let starts = evaluated["starts"].as_array().unwrap();
            assert_eq!(starts.len(), pairs.len());
            let mut success = 0.0;
            for (actual, &(start, goal, weight)) in starts.iter().zip(&pairs) {
                assert_eq!(actual["start"], format!("{start:?}"));
                assert_eq!(actual["goal"], format!("{goal:?}"));
                assert_eq!(actual["weight"].as_f64().unwrap(), 1.0 / pairs.len() as f64);
                if actual["success"].as_bool().unwrap() {
                    success += weight;
                }
            }
            assert_eq!(evaluated["success"].as_f64().unwrap(), success);
        }
    }
}

#[test]
fn read_only_blocks_preserve_the_complete_state_of_both_agents() {
    let preset = Preset::thesis();
    for name in names() {
        let distributions = distributions(name);
        assert_protocol_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
            &distributions.protocol(),
            42,
            &[50, 100],
            || CoreAgent {
                agent: acs2_agents::acs2::Agent::<10, _>::new(
                    preset.config(8),
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                )
                .with_truncation_mode(preset.truncation),
                preset,
                replay: false,
                steps: 0,
                updates: 0,
            },
            |agent| {
                format!(
                    "{:?};{};{}",
                    agent.agent.capture(),
                    agent.steps,
                    agent.updates
                )
            },
        );
        assert_protocol_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
            &distributions.protocol(),
            42,
            &[50, 100],
            || CoreAgent {
                agent: acs2_agents::acs2er::Agent::<10, _>::new(
                    preset.config(8),
                    acs2_agents::acs2er::replay::ReplayConfiguration {
                        buffer_size: 10_000,
                        min_samples: 1,
                        samples_number: 3,
                    },
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                )
                .with_truncation_mode(preset.truncation),
                preset,
                replay: true,
                steps: 0,
                updates: 0,
            },
            |agent| {
                format!(
                    "{:?};{};{}",
                    agent.agent.capture(),
                    agent.steps,
                    agent.updates
                )
            },
        );
    }
}

#[test]
fn full_pool_learning_fields_reproduce_three_archived_pilot_configurations() {
    let oracle: Value = serde_json::from_str(include_str!(
        "../../fixtures/goal_measurement_regression.json"
    ))
    .unwrap();
    for configuration in oracle["configurations"].as_array().unwrap() {
        let name = configuration["task"].as_str().unwrap();
        let geometry = geometry_by_id(&format!("Maze{}-v0", &name[4..])).unwrap();
        let task = MazeTask::<Coordinates, 2>::new(
            name.to_owned(),
            geometry,
            MazeTopology::new(geometry)
                .unwrap()
                .walkable_cells()
                .to_vec(),
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
        let mut rows = Vec::new();
        let mut metadata = RunMetadata::test("test");
        metadata.record_starts = false;
        run_with_sink::<_, _, 8, 2, 10>(
            &task,
            kind,
            42,
            &targets,
            &metadata,
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        for (actual, expected) in rows.iter().zip(configuration["rows"].as_array().unwrap()) {
            let serialized: Value = serde_json::from_str(&actual.to_string()).unwrap();
            for (key, value) in expected.as_object().unwrap() {
                assert_eq!(
                    &serialized[key],
                    value,
                    "{name} {} point {} field {key}",
                    kind.name(),
                    expected["nominal_step"]
                );
            }
        }
    }
}
