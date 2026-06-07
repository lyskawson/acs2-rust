use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::rng::RandomSource;

pub const MAZE_PERCEPTION_LEN: usize = 8;

pub struct Maze {
    matrix: Vec<Vec<u8>>,
    agent_row: usize,
    agent_col: usize,
    max_episode_steps: u32,
    elapsed_steps: u32,
    rng: Box<dyn RandomSource>,
}

impl Maze {
    pub fn new(
        matrix: Vec<Vec<u8>>,
        max_episode_steps: u32,
        rng: Box<dyn RandomSource>,
    ) -> Self {
        todo!()
    }

    pub fn perception(&self) -> Perception<MAZE_PERCEPTION_LEN> {
        todo!()
    }
}

impl Environment<MAZE_PERCEPTION_LEN> for Maze {
    fn reset(&mut self) -> Perception<MAZE_PERCEPTION_LEN> {
        todo!()
    }

    fn step(&mut self, action: usize) -> StepOutcome<MAZE_PERCEPTION_LEN> {
        todo!()
    }
}
