use std::marker::PhantomData;

use acs2_core::goal::{ExactMatch, Goal, GoalEnvironment, GoalStart, GoalStep};
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;

use crate::maze::geometries::MazeGeometry;
use crate::maze::topology::{Cell, MazeGeometryError, MazeTopology};
use crate::maze::MAZE_PERCEPTION_LEN;

use super::SPARSE_GOAL_REWARD;

mod sealed {
    pub trait Encoding {}
}

pub trait MazeGoalEncoding<const G: usize>: sealed::Encoding {
    const REQUIRES_UNIQUE_PERCEPTION: bool;
    fn encode(topology: &MazeTopology, cell: Cell) -> Goal<G>;
}

pub struct Coordinates;
pub struct NeighbourPerception;

impl sealed::Encoding for Coordinates {}
impl sealed::Encoding for NeighbourPerception {}

impl MazeGoalEncoding<2> for Coordinates {
    const REQUIRES_UNIQUE_PERCEPTION: bool = false;

    fn encode(_topology: &MazeTopology, cell: Cell) -> Goal<2> {
        Goal::new([Symbol::Token(cell.0 as u8), Symbol::Token(cell.1 as u8)])
    }
}

impl MazeGoalEncoding<MAZE_PERCEPTION_LEN> for NeighbourPerception {
    const REQUIRES_UNIQUE_PERCEPTION: bool = true;

    fn encode(topology: &MazeTopology, cell: Cell) -> Goal<MAZE_PERCEPTION_LEN> {
        Goal::new(topology.perception_at(cell).symbols)
    }
}

pub type CoordinateGoalMaze = GoalMaze<Coordinates, 2>;
pub type PerceptionGoalMaze = GoalMaze<NeighbourPerception, MAZE_PERCEPTION_LEN>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalMazeError {
    Geometry(MazeGeometryError),
    ZeroStepCap,
    EmptyPool,
    InvalidGoalCell(Cell),
    DuplicateGoalCell(Cell),
    PerceptionTwin { goal: Cell, twin: Cell },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StartDistribution {
    PyalcsPaths,
    WalkableExceptGoal,
}

pub struct GoalMaze<E, const G: usize> {
    topology: MazeTopology,
    goal_cells: Vec<Cell>,
    goals: Vec<Goal<G>>,
    start_distribution: StartDistribution,
    position: Cell,
    desired: Option<Goal<G>>,
    step_cap: u32,
    elapsed_steps: u32,
    rng: Box<dyn RandomSource>,
    encoding: PhantomData<E>,
}

impl<E: MazeGoalEncoding<G>, const G: usize> GoalMaze<E, G> {
    pub fn multi_goal(
        geometry: &MazeGeometry,
        goal_cells: Vec<Cell>,
        step_cap: u32,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self, GoalMazeError> {
        Self::construct(
            geometry,
            goal_cells,
            step_cap,
            StartDistribution::WalkableExceptGoal,
            rng,
        )
    }

    pub fn all_walkable(
        geometry: &MazeGeometry,
        step_cap: u32,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self, GoalMazeError> {
        let topology = MazeTopology::new(geometry).map_err(GoalMazeError::Geometry)?;
        Self::multi_goal(geometry, topology.walkable_cells().to_vec(), step_cap, rng)
    }

    pub fn single_goal(
        geometry: &MazeGeometry,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self, GoalMazeError> {
        Self::single_goal_with_cap(geometry, geometry.max_episode_steps, rng)
    }

    pub fn single_goal_with_cap(
        geometry: &MazeGeometry,
        step_cap: u32,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self, GoalMazeError> {
        let topology = MazeTopology::new(geometry).map_err(GoalMazeError::Geometry)?;
        Self::construct(
            geometry,
            vec![topology.reward_cell()],
            step_cap,
            StartDistribution::PyalcsPaths,
            rng,
        )
    }

    fn construct(
        geometry: &MazeGeometry,
        goal_cells: Vec<Cell>,
        step_cap: u32,
        start_distribution: StartDistribution,
        rng: Box<dyn RandomSource>,
    ) -> Result<Self, GoalMazeError> {
        let topology = MazeTopology::new(geometry).map_err(GoalMazeError::Geometry)?;
        if step_cap == 0 {
            return Err(GoalMazeError::ZeroStepCap);
        }
        if goal_cells.is_empty() {
            return Err(GoalMazeError::EmptyPool);
        }
        for (index, &goal) in goal_cells.iter().enumerate() {
            if !topology.is_walkable(goal) {
                return Err(GoalMazeError::InvalidGoalCell(goal));
            }
            if goal_cells[..index].contains(&goal) {
                return Err(GoalMazeError::DuplicateGoalCell(goal));
            }
            if E::REQUIRES_UNIQUE_PERCEPTION {
                let perception = topology.perception_at(goal);
                if let Some(&twin) = topology
                    .walkable_cells()
                    .iter()
                    .find(|&&cell| cell != goal && topology.perception_at(cell) == perception)
                {
                    return Err(GoalMazeError::PerceptionTwin { goal, twin });
                }
            }
        }
        let goals = goal_cells
            .iter()
            .map(|&cell| E::encode(&topology, cell))
            .collect();
        let position = topology.path_cells()[0];
        Ok(Self {
            topology,
            goal_cells,
            goals,
            start_distribution,
            position,
            desired: None,
            step_cap,
            elapsed_steps: 0,
            rng,
            encoding: PhantomData,
        })
    }

    pub fn topology(&self) -> &MazeTopology {
        &self.topology
    }

    pub fn goal_pool(&self) -> &[Goal<G>] {
        &self.goals
    }

    pub fn goal_cells(&self) -> &[Cell] {
        &self.goal_cells
    }

    pub fn position(&self) -> Cell {
        self.position
    }

    pub fn step_cap(&self) -> u32 {
        self.step_cap
    }

    pub fn goal_at(&self, cell: Cell) -> Goal<G> {
        assert!(
            self.topology.is_walkable(cell),
            "a goal requires a walkable cell"
        );
        E::encode(&self.topology, cell)
    }

    pub fn distance_to_goal(&self, start: Cell, desired: &Goal<G>) -> Option<u32> {
        let mut matches = self
            .topology
            .walkable_cells()
            .iter()
            .copied()
            .filter(|&cell| E::encode(&self.topology, cell) == *desired);
        let target = matches.next()?;
        if matches.next().is_some() {
            return None;
        }
        self.topology.shortest_distance(start, target)
    }

    pub fn reset_at(&mut self, start: Cell, desired: Goal<G>) -> GoalStart<MAZE_PERCEPTION_LEN, G> {
        assert!(
            self.goals.contains(&desired),
            "desired goal must belong to the pool"
        );
        assert!(self.topology.is_walkable(start), "start must be walkable");
        assert_ne!(
            self.goal_at(start),
            desired,
            "start must differ from the goal"
        );
        if self.start_distribution == StartDistribution::PyalcsPaths {
            assert!(
                self.topology.path_cells().contains(&start),
                "single-goal start must be a path cell"
            );
        }
        self.position = start;
        self.elapsed_steps = 0;
        self.desired = Some(desired);
        GoalStart {
            observation: self.topology.perception_at(start),
            achieved: self.goal_at(start),
            desired,
        }
    }
}

impl<E: MazeGoalEncoding<G>, const G: usize> GoalEnvironment<MAZE_PERCEPTION_LEN, G>
    for GoalMaze<E, G>
{
    type Objective = ExactMatch;

    fn objective(&self) -> &Self::Objective {
        &SPARSE_GOAL_REWARD
    }

    fn reset(&mut self) -> GoalStart<MAZE_PERCEPTION_LEN, G> {
        let index = if self.start_distribution == StartDistribution::PyalcsPaths {
            0
        } else {
            self.rng.gen_range(self.goals.len())
        };
        self.reset_with_goal(self.goals[index])
    }

    fn reset_with_goal(&mut self, desired: Goal<G>) -> GoalStart<MAZE_PERCEPTION_LEN, G> {
        let goal_index = self
            .goals
            .iter()
            .position(|goal| *goal == desired)
            .expect("desired goal must belong to the pool");
        let start = match self.start_distribution {
            StartDistribution::PyalcsPaths => {
                self.topology.path_cells()[self.rng.gen_range(self.topology.path_cells().len())]
            }
            StartDistribution::WalkableExceptGoal => {
                let cells = self.topology.walkable_cells();
                let excluded = cells
                    .iter()
                    .position(|&cell| cell == self.goal_cells[goal_index])
                    .unwrap();
                let chosen = self.rng.gen_range(cells.len() - 1);
                cells[chosen + usize::from(chosen >= excluded)]
            }
        };
        self.reset_at(start, desired)
    }

    fn step(&mut self, action: usize) -> GoalStep<MAZE_PERCEPTION_LEN, G> {
        let desired = self.desired.expect("goal maze must be reset before a step");
        self.position = self.topology.next_cell(self.position, action);
        self.elapsed_steps += 1;
        let step = GoalStep {
            observation: self.topology.perception_at(self.position),
            achieved: self.goal_at(self.position),
            terminal_state: false,
            time_limit_reached: self.elapsed_steps >= self.step_cap,
        };
        let outcome = step.outcome(self.objective(), &desired);
        if outcome.terminated || outcome.truncated {
            self.desired = None;
        }
        step
    }
}
