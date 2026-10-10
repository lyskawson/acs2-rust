use std::collections::BTreeSet;

use acs2_envs::goal::maze::MazeGoalEncoding;
use acs2_envs::maze::geometries::MazeGeometry;
use acs2_envs::maze::topology::{Cell, MazeTopology};
use serde_json::{json, Value};

use crate::task::MazeTask;

pub const MAZE_PROTOCOL_ID: &str = "maze-goals-v1";
pub const MAZE_PROTOCOL: &str = include_str!("../../experiments/maze-baselines/protocol.json");

pub struct EvaluationBlock<'a, T> {
    pub name: &'static str,
    pub task: &'a T,
}

pub struct TaskProtocol<'a, T> {
    pub training: &'a T,
    pub evaluations: Vec<EvaluationBlock<'a, T>>,
    pub registration: Option<Value>,
}

impl<'a, T> TaskProtocol<'a, T> {
    pub fn full_pool(training: &'a T) -> Self {
        Self {
            training,
            evaluations: vec![EvaluationBlock {
                name: "full_pool",
                task: training,
            }],
            registration: None,
        }
    }
}

pub struct MazeGoalDistributions<E, const G: usize> {
    pub training: MazeTask<E, G>,
    pub validation: MazeTask<E, G>,
    pub test: MazeTask<E, G>,
    pub registration: Value,
}

pub fn registered_configuration(name: &str) -> Value {
    let protocol: Value = serde_json::from_str(MAZE_PROTOCOL).expect("registered maze protocol");
    assert_eq!(protocol["id"], MAZE_PROTOCOL_ID);
    protocol["configurations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["task"] == name)
        .expect("maze belongs to the final set")
        .clone()
}

pub fn goal_cells(configuration: &Value, block: &str) -> Vec<Cell> {
    configuration["blocks"][block]["goals"]
        .as_array()
        .expect("registered goal block")
        .iter()
        .map(|value| {
            (
                value[0].as_u64().unwrap() as usize,
                value[1].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

pub fn validate_partition(geometry: &MazeGeometry, configuration: &Value) {
    assert_eq!(configuration["geometry"], geometry.id);
    assert_eq!(configuration["split_unit"], "goal_cells");
    let topology = MazeTopology::new(geometry).expect("valid geometry");
    let complete: BTreeSet<_> = topology.walkable_cells().iter().copied().collect();
    let mut seen = BTreeSet::new();
    for block in ["training", "validation", "test"] {
        let cells = goal_cells(configuration, block);
        assert!(cells.len() >= 3, "each block needs at least three goals");
        for cell in &cells {
            assert!(complete.contains(cell), "goal must be walkable");
            assert!(seen.insert(*cell), "goal blocks must be disjoint");
        }
        assert_eq!(
            configuration["blocks"][block]["evaluation_pairs"]
                .as_u64()
                .unwrap() as usize,
            cells.len() * (complete.len() - 1)
        );
    }
    assert_eq!(seen, complete, "goal blocks cover every walkable cell");
}

impl<E: MazeGoalEncoding<G>, const G: usize> MazeGoalDistributions<E, G> {
    pub fn registered(
        name: &str,
        geometry: &'static MazeGeometry,
        cap: u32,
        encoding: &'static str,
    ) -> Self {
        let registration = registered_configuration(name);
        assert_eq!(registration["cap"], cap, "use the registered episode cap");
        validate_partition(geometry, &registration);
        let make = |block| {
            MazeTask::new(
                name.to_owned(),
                geometry,
                goal_cells(&registration, block),
                cap,
                encoding,
            )
        };
        Self {
            training: make("training"),
            validation: make("validation"),
            test: make("test"),
            registration,
        }
    }

    pub fn protocol(&self) -> TaskProtocol<'_, MazeTask<E, G>> {
        TaskProtocol {
            training: &self.training,
            evaluations: vec![
                EvaluationBlock {
                    name: "training",
                    task: &self.training,
                },
                EvaluationBlock {
                    name: "validation",
                    task: &self.validation,
                },
                EvaluationBlock {
                    name: "test",
                    task: &self.test,
                },
            ],
            registration: Some(json!({"id": MAZE_PROTOCOL_ID, "configuration": self.registration})),
        }
    }
}
