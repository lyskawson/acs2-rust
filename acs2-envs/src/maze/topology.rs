use std::collections::VecDeque;

use acs2_core::perception::Perception;
use acs2_core::symbol::Symbol;

use super::geometries::MazeGeometry;
use super::{MAZE_PERCEPTION_LEN, NEIGHBOUR_OFFSETS};

pub type Cell = (usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MazeGeometryError {
    NonRectangular,
    UnclosedBorder,
    UnsupportedCell,
    CoordinateOverflow,
    MissingPaths,
    RewardCount,
}

pub struct MazeTopology {
    matrix: &'static [&'static [u8]],
    walkable: Vec<Cell>,
    paths: Vec<Cell>,
    reward: Cell,
}

impl MazeTopology {
    pub fn new(geometry: &MazeGeometry) -> Result<Self, MazeGeometryError> {
        let matrix = geometry.matrix;
        let height = matrix.len();
        let width = matrix.first().map_or(0, |row| row.len());
        if height < 3 || width < 3 || matrix.iter().any(|row| row.len() != width) {
            return Err(MazeGeometryError::NonRectangular);
        }
        if height > 256 || width > 256 {
            return Err(MazeGeometryError::CoordinateOverflow);
        }
        if (0..width).any(|col| matrix[0][col] != 1 || matrix[height - 1][col] != 1)
            || (0..height).any(|row| matrix[row][0] != 1 || matrix[row][width - 1] != 1)
        {
            return Err(MazeGeometryError::UnclosedBorder);
        }
        let mut paths = Vec::new();
        let mut walkable = Vec::new();
        let mut rewards = Vec::new();
        for (row, values) in matrix.iter().enumerate() {
            for (col, &value) in values.iter().enumerate() {
                match value {
                    0 => {
                        paths.push((row, col));
                        walkable.push((row, col));
                    }
                    9 => {
                        rewards.push((row, col));
                        walkable.push((row, col));
                    }
                    1 => {}
                    _ => return Err(MazeGeometryError::UnsupportedCell),
                }
            }
        }
        if paths.is_empty() {
            return Err(MazeGeometryError::MissingPaths);
        }
        if rewards.len() != 1 {
            return Err(MazeGeometryError::RewardCount);
        }
        Ok(Self {
            matrix,
            walkable,
            paths,
            reward: rewards[0],
        })
    }

    pub fn walkable_cells(&self) -> &[Cell] {
        &self.walkable
    }

    pub fn path_cells(&self) -> &[Cell] {
        &self.paths
    }

    pub fn reward_cell(&self) -> Cell {
        self.reward
    }

    pub fn is_walkable(&self, cell: Cell) -> bool {
        self.matrix
            .get(cell.0)
            .and_then(|row| row.get(cell.1))
            .is_some_and(|&value| value == 0 || value == 9)
    }

    pub fn perception_at(&self, cell: Cell) -> Perception<MAZE_PERCEPTION_LEN> {
        assert!(
            self.is_walkable(cell),
            "perception requires a walkable cell"
        );
        Perception::new(core::array::from_fn(|index| {
            let neighbour = offset(cell, index);
            Symbol::Token(b'0' + self.matrix[neighbour.0][neighbour.1])
        }))
    }

    pub fn next_cell(&self, cell: Cell, action: usize) -> Cell {
        assert!(self.is_walkable(cell), "a move requires a walkable cell");
        let target = offset(cell, action);
        if self.is_walkable(target) {
            target
        } else {
            cell
        }
    }

    pub fn shortest_distance(&self, start: Cell, goal: Cell) -> Option<u32> {
        if !self.is_walkable(start) || !self.is_walkable(goal) {
            return None;
        }
        let width = self.matrix[0].len();
        let mut distances = vec![None; width * self.matrix.len()];
        let mut queue = VecDeque::from([start]);
        distances[start.0 * width + start.1] = Some(0);
        while let Some(cell) = queue.pop_front() {
            let distance = distances[cell.0 * width + cell.1].unwrap();
            if cell == goal {
                return Some(distance);
            }
            for action in 0..MAZE_PERCEPTION_LEN {
                let next = self.next_cell(cell, action);
                let index = next.0 * width + next.1;
                if distances[index].is_none() {
                    distances[index] = Some(distance + 1);
                    queue.push_back(next);
                }
            }
        }
        None
    }
}

fn offset(cell: Cell, action: usize) -> Cell {
    let (row, col) = NEIGHBOUR_OFFSETS[action];
    (
        (cell.0 as isize + row) as usize,
        (cell.1 as isize + col) as usize,
    )
}
