use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;

use crate::maze_data::MazeGeometry;

pub const MAZE_PERCEPTION_LEN: usize = 8;

const PATH_CODE: u8 = 0;
const WALL_CODE: u8 = 1;
const REWARD_CODE: u8 = 9;
const DIGIT_ZERO: u8 = b'0';

const NEIGHBOUR_OFFSETS: [(isize, isize); MAZE_PERCEPTION_LEN] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

pub struct Maze {
    matrix: Vec<Vec<u8>>,
    path_cells: Vec<(usize, usize)>,
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
        let path_cells = collect_path_cells(&matrix);
        let (agent_row, agent_col) = path_cells[0];
        Self {
            matrix,
            path_cells,
            agent_row,
            agent_col,
            max_episode_steps,
            elapsed_steps: 0,
            rng,
        }
    }

    pub fn from_geometry(geometry: &MazeGeometry, rng: Box<dyn RandomSource>) -> Self {
        let matrix = geometry
            .matrix
            .iter()
            .map(|row| row.to_vec())
            .collect();
        Self::new(matrix, geometry.max_episode_steps, rng)
    }

    pub fn agent_position(&self) -> (usize, usize) {
        (self.agent_row, self.agent_col)
    }

    pub fn place_agent(&mut self, row: usize, col: usize) {
        assert_eq!(self.matrix[row][col], PATH_CODE);
        self.agent_row = row;
        self.agent_col = col;
        self.elapsed_steps = 0;
    }

    pub fn perception(&self) -> Perception<MAZE_PERCEPTION_LEN> {
        let row = self.agent_row as isize;
        let col = self.agent_col as isize;
        let symbols = core::array::from_fn(|index| {
            let (delta_row, delta_col) = NEIGHBOUR_OFFSETS[index];
            let cell = self.matrix[(row + delta_row) as usize][(col + delta_col) as usize];
            Symbol::Token(DIGIT_ZERO + cell)
        });
        Perception::new(symbols)
    }
}

fn collect_path_cells(matrix: &[Vec<u8>]) -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for (row_index, row) in matrix.iter().enumerate() {
        for (col_index, cell) in row.iter().enumerate() {
            if *cell == PATH_CODE {
                cells.push((row_index, col_index));
            }
        }
    }
    cells
}

impl Environment<MAZE_PERCEPTION_LEN> for Maze {
    fn reset(&mut self) -> Perception<MAZE_PERCEPTION_LEN> {
        let chosen = self.rng.gen_range(self.path_cells.len());
        let (row, col) = self.path_cells[chosen];
        self.agent_row = row;
        self.agent_col = col;
        self.elapsed_steps = 0;
        self.perception()
    }

    fn step(&mut self, action: usize) -> StepOutcome<MAZE_PERCEPTION_LEN> {
        let (delta_row, delta_col) = NEIGHBOUR_OFFSETS[action];
        let target_row = (self.agent_row as isize + delta_row) as usize;
        let target_col = (self.agent_col as isize + delta_col) as usize;
        let target_cell = self.matrix[target_row][target_col];

        let mut terminated = false;
        if target_cell != WALL_CODE {
            self.agent_row = target_row;
            self.agent_col = target_col;
            terminated = target_cell == REWARD_CODE;
        }

        self.elapsed_steps += 1;
        let truncated = !terminated && self.elapsed_steps >= self.max_episode_steps;
        let reward = if terminated { 1000.0 } else { 0.0 };

        StepOutcome {
            observation: self.perception(),
            reward,
            terminated,
            truncated,
            info: (),
        }
    }
}
