use crate::goal_protocol::TaskProtocol;
use crate::pair_protocol::{goal_starts, MazePairDistributions, MazePairTask};
use acs2_envs::goal::maze::MazeGoalEncoding;
use acs2_envs::maze::geometries::MazeGeometry;
use acs2_envs::maze::topology::MazeTopology;
use serde_json::Value;
use std::collections::BTreeSet;

pub const ID: &str = "alcs-maze-pairs-v1";
pub const REGISTRATION: &str =
    include_str!("../../experiments/per-goal-reward-ablation/pair-protocol.json");

pub fn configuration(task: &str) -> Value {
    let value: Value = serde_json::from_str(REGISTRATION).expect("research pair registration");
    assert_eq!(value["id"], ID);
    value["configurations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["task"] == task)
        .expect("registered ALCS research task")
        .clone()
}

pub fn validate(geometry: &MazeGeometry, config: &Value) {
    assert_eq!(config["geometry"], geometry.id);
    assert_eq!(config["split_unit"], "start_goal_pairs");
    let topology = MazeTopology::multi_goal(geometry).unwrap();
    let cells: BTreeSet<_> = topology.walkable_cells().iter().copied().collect();
    let topology_ref = &topology;
    let expected: BTreeSet<_> = cells
        .iter()
        .flat_map(|&start| {
            cells
                .iter()
                .filter(move |&&goal| {
                    goal != start && topology_ref.shortest_distance(start, goal).is_some()
                })
                .map(move |&goal| (start, goal))
        })
        .collect();
    let diameter = expected
        .iter()
        .map(|&(start, goal)| topology.shortest_distance(start, goal).unwrap())
        .max()
        .unwrap();
    assert_eq!(config["cap"], diameter);
    let mut seen = BTreeSet::new();
    for block in ["training", "validation", "test"] {
        let mut goals = BTreeSet::new();
        let before = seen.len();
        for (goal, starts) in goal_starts(config, block) {
            assert!(cells.contains(&goal) && goals.insert(goal));
            assert!(!starts.is_empty());
            for start in starts {
                assert!(expected.contains(&(start, goal)) && seen.insert((start, goal)));
            }
        }
        assert_eq!(goals, cells);
        assert_eq!(
            config["blocks"][block]["evaluation_pairs"],
            seen.len() - before
        );
    }
    assert_eq!(seen, expected);
}

pub fn registered<E: MazeGoalEncoding<G>, const G: usize>(
    task: &str,
    geometry: &'static MazeGeometry,
    cap: u32,
    encoding: &'static str,
) -> MazePairDistributions<E, G> {
    let registration = configuration(task);
    assert_eq!(registration["cap"], cap);
    assert_eq!(encoding, "coordinates");
    validate(geometry, &registration);
    let make =
        |block| MazePairTask::registered(task, geometry, cap, encoding, &registration, block);
    MazePairDistributions {
        training: make("training"),
        validation: make("validation"),
        test: make("test"),
        registration,
    }
}

pub fn protocol<E: MazeGoalEncoding<G>, const G: usize>(
    distributions: &MazePairDistributions<E, G>,
) -> TaskProtocol<'_, MazePairTask<E, G>> {
    let mut result = distributions.protocol();
    result.registration.as_mut().unwrap()["id"] = ID.into();
    result
}
