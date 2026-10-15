use std::collections::{BTreeMap, BTreeSet};

use acs2_core::ports::goal::{Goal, GoalStart};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::support::knowledge::Transition;
use acs2_envs::goal::maze::{GoalMaze, MazeGoalEncoding};
use acs2_envs::maze::geometries::MazeGeometry;
use acs2_envs::maze::topology::{Cell, MazeTopology};
use serde_json::{json, Value};

use crate::goal_protocol::{EvaluationBlock, TaskProtocol};
use crate::task::{MazeTask, Pair, Task};

pub const MAZE_PAIR_PROTOCOL_ID: &str = "maze-pairs-v1";
pub const MAZE_PAIR_PROTOCOL: &str =
    include_str!("../../experiments/maze-baselines/pair-protocol.json");

pub fn registered_configuration(name: &str) -> Value {
    let protocol: Value =
        serde_json::from_str(MAZE_PAIR_PROTOCOL).expect("registered pair protocol");
    assert_eq!(protocol["id"], MAZE_PAIR_PROTOCOL_ID);
    protocol["configurations"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["task"] == name)
        .expect("maze belongs to the registered pair protocol")
        .clone()
}

fn cell(value: &Value) -> Cell {
    (
        usize::try_from(value[0].as_u64().unwrap()).unwrap(),
        usize::try_from(value[1].as_u64().unwrap()).unwrap(),
    )
}

pub fn goal_starts(configuration: &Value, block: &str) -> Vec<(Cell, Vec<Cell>)> {
    configuration["blocks"][block]["goal_starts"]
        .as_array()
        .expect("registered pair block")
        .iter()
        .map(|entry| {
            (
                cell(&entry["goal"]),
                entry["starts"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(cell)
                    .collect(),
            )
        })
        .collect()
}

pub fn validate_partition(geometry: &MazeGeometry, configuration: &Value) {
    assert_eq!(configuration["geometry"], geometry.id);
    assert_eq!(configuration["split_unit"], "start_goal_pairs");
    let topology = MazeTopology::new(geometry).expect("valid geometry");
    let cells = topology.walkable_cells();
    let complete: BTreeSet<_> = cells.iter().copied().collect();
    let mut seen = BTreeSet::new();
    for block in ["training", "validation", "test"] {
        let entries = goal_starts(configuration, block);
        let mut goals = BTreeSet::new();
        let count_before = seen.len();
        for (goal, starts) in entries {
            assert!(complete.contains(&goal) && goals.insert(goal));
            assert!(
                starts.len() >= 3,
                "every goal needs at least three starts per block"
            );
            assert_eq!(
                configuration["blocks"][block]["starts_per_goal"],
                starts.len()
            );
            let mut ranked: Vec<_> = cells
                .iter()
                .copied()
                .filter(|&start| start != goal)
                .map(|start| (topology.shortest_distance(start, goal).unwrap(), start))
                .collect();
            ranked.sort_unstable();
            let mut offset = 0;
            for stratum in 0..3 {
                let size = ranked.len() / 3 + usize::from(stratum < ranked.len() % 3);
                assert!(ranked[offset..offset + size]
                    .iter()
                    .any(|(_, start)| starts.contains(start)));
                offset += size;
            }
            let mut distances = BTreeSet::new();
            for start in starts {
                assert!(complete.contains(&start) && start != goal);
                assert!(seen.insert((start, goal)), "pair blocks must be disjoint");
                distances.insert(topology.shortest_distance(start, goal).unwrap());
            }
            assert!(distances.len() >= 2, "each goal needs near and far starts");
        }
        assert_eq!(goals, complete, "every block must contain every goal");
        let count = seen.len() - count_before;
        assert!(count >= 40);
        assert_eq!(configuration["blocks"][block]["evaluation_pairs"], count);
    }
    let all_pairs: BTreeSet<_> = cells
        .iter()
        .flat_map(|&start| {
            cells
                .iter()
                .filter(move |&&goal| start != goal)
                .map(move |&goal| (start, goal))
        })
        .collect();
    assert_eq!(
        seen, all_pairs,
        "pair blocks must cover every distinct walkable pair"
    );
}

pub struct MazePairTask<E, const G: usize> {
    pub maze: MazeTask<E, G>,
    starts_by_goal: Vec<Vec<Cell>>,
    random_floor: f64,
    ceiling: f64,
}

impl<E: MazeGoalEncoding<G>, const G: usize> MazePairTask<E, G> {
    fn registered(
        name: &str,
        geometry: &'static MazeGeometry,
        cap: u32,
        encoding: &'static str,
        configuration: &Value,
        block: &str,
    ) -> Self {
        let topology = MazeTopology::new(geometry).unwrap();
        let goals = topology.walkable_cells().to_vec();
        let starts: BTreeMap<_, _> = goal_starts(configuration, block).into_iter().collect();
        let starts_by_goal = goals.iter().map(|goal| starts[goal].clone()).collect();
        Self {
            maze: MazeTask::new(name.to_owned(), geometry, goals, cap, encoding),
            starts_by_goal,
            random_floor: configuration["blocks"][block]["random_floor"]
                .as_f64()
                .unwrap(),
            ceiling: configuration["blocks"][block]["reachable_within_cap"]
                .as_f64()
                .unwrap(),
        }
    }
}

impl<E: MazeGoalEncoding<G>, const G: usize, const M: usize> Task<8, G, M> for MazePairTask<E, G> {
    type Env = GoalMaze<E, G>;
    type State = Cell;

    fn name(&self) -> &str {
        &self.maze.name
    }
    fn actions(&self) -> usize {
        8
    }
    fn cap(&self) -> u32 {
        self.maze.cap
    }
    fn encoding(&self) -> &'static str {
        self.maze.encoding
    }
    fn pool_label(&self) -> String {
        "full".to_owned()
    }
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env {
        <MazeTask<E, G> as Task<8, G, M>>::environment(&self.maze, rng)
    }
    fn states(&self) -> Vec<Cell> {
        self.maze.template.topology().walkable_cells().to_vec()
    }
    fn next(&self, state: Cell, action: usize) -> Cell {
        self.maze.template.topology().next_cell(state, action)
    }
    fn achieved(&self, state: Cell) -> Goal<G> {
        self.maze.template.goal_at(state)
    }
    fn distance(&self, state: Cell, goal: &Goal<G>) -> Option<u32> {
        self.maze.template.distance_to_goal(state, goal)
    }
    fn pairs(&self) -> Vec<Pair<Cell, G>> {
        let count: usize = self.starts_by_goal.iter().map(Vec::len).sum();
        self.maze
            .pool
            .iter()
            .zip(&self.starts_by_goal)
            .flat_map(|(&goal, starts)| {
                starts.iter().map(move |&start| {
                    (start, self.maze.template.goal_at(goal), 1.0 / count as f64)
                })
            })
            .collect()
    }
    fn reset_at(&self, env: &mut Self::Env, state: Cell, goal: Goal<G>) -> GoalStart<8, G> {
        env.reset_at(state, goal)
    }
    fn reset_training(&self, env: &mut Self::Env, rng: &mut ChaChaRandomSource) -> GoalStart<8, G> {
        let goal_index = rng.gen_range(self.maze.pool.len());
        let starts = &self.starts_by_goal[goal_index];
        let start = starts[rng.gen_range(starts.len())];
        env.reset_at(
            start,
            self.maze.template.goal_at(self.maze.pool[goal_index]),
        )
    }
    fn training_goals(&self) -> Option<&[Goal<G>]> {
        Some(self.maze.template.goal_pool())
    }
    fn analytical_random_success(&self, _pairs: &[Pair<Cell, G>]) -> Option<f64> {
        Some(self.random_floor)
    }
    fn analytical_reachability(&self) -> Option<(f64, f64, f64)> {
        Some((self.ceiling, 1.0 - self.ceiling, 0.0))
    }
    fn knowledge_transitions(&self) -> Option<Vec<Transition<M>>> {
        Some(crate::knowledge::maze_transitions(&self.maze.template))
    }
    fn evaluation_knowledge_transitions(&self) -> Option<Vec<Transition<M>>> {
        Some(crate::knowledge::maze_pair_transitions(
            &self.maze.template,
            <Self as Task<8, G, M>>::pairs(self).into_iter().map(|(state, goal, _)| (state, goal)),
        ))
    }
}

pub struct MazePairDistributions<E, const G: usize> {
    pub training: MazePairTask<E, G>,
    pub validation: MazePairTask<E, G>,
    pub test: MazePairTask<E, G>,
    pub registration: Value,
}

impl<E: MazeGoalEncoding<G>, const G: usize> MazePairDistributions<E, G> {
    pub fn registered(
        name: &str,
        geometry: &'static MazeGeometry,
        cap: u32,
        encoding: &'static str,
    ) -> Self {
        let registration = registered_configuration(name);
        assert_eq!(registration["cap"], cap, "use the registered episode cap");
        assert_eq!(encoding, "coordinates");
        validate_partition(geometry, &registration);
        let make =
            |block| MazePairTask::registered(name, geometry, cap, encoding, &registration, block);
        Self {
            training: make("training"),
            validation: make("validation"),
            test: make("test"),
            registration,
        }
    }

    pub fn protocol(&self) -> TaskProtocol<'_, MazePairTask<E, G>> {
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
            registration: Some(
                json!({"id": MAZE_PAIR_PROTOCOL_ID, "configuration": self.registration}),
            ),
        }
    }
}
