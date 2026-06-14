use std::collections::{HashSet, VecDeque};

use acs2_core::environment::Environment;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::maze::Maze;
use acs2_envs::maze_data::{MazeSource, UNOLD_GEOMETRIES};

const PATH: u8 = 0;
const WALL: u8 = 1;
const GOAL: u8 = 9;

const OFFSETS: [(isize, isize); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

fn rng() -> Box<ChaChaRandomSource> {
    Box::new(ChaChaRandomSource::from_seed(7))
}

fn goal_cell(matrix: &[&[u8]]) -> (usize, usize) {
    for (row_index, row) in matrix.iter().enumerate() {
        for (col_index, cell) in row.iter().enumerate() {
            if *cell == GOAL {
                return (row_index, col_index);
            }
        }
    }
    panic!("no goal cell");
}

fn reachable_count(matrix: &[&[u8]], start: (usize, usize)) -> usize {
    let height = matrix.len();
    let width = matrix[0].len();
    let mut seen = HashSet::from([start]);
    let mut queue = VecDeque::from([start]);
    while let Some((row, col)) = queue.pop_front() {
        for (delta_row, delta_col) in OFFSETS {
            let next_row = row as isize + delta_row;
            let next_col = col as isize + delta_col;
            if next_row < 0 || next_col < 0 || next_row >= height as isize || next_col >= width as isize
            {
                continue;
            }
            let cell = (next_row as usize, next_col as usize);
            if matrix[cell.0][cell.1] != WALL && seen.insert(cell) {
                queue.push_back(cell);
            }
        }
    }
    seen.len()
}

#[test]
fn unold_geometries_are_well_formed() {
    assert_eq!(UNOLD_GEOMETRIES.len(), 22);
    for geometry in UNOLD_GEOMETRIES {
        assert_eq!(geometry.source, MazeSource::OunoldAlcs, "{}", geometry.id);
        assert_eq!(geometry.max_episode_steps, 200, "{}", geometry.id);
        let width = geometry.matrix[0].len();
        assert!(
            geometry.matrix.iter().all(|row| row.len() == width),
            "{} is not rectangular",
            geometry.id
        );
        assert!(
            geometry
                .matrix
                .iter()
                .flat_map(|row| row.iter())
                .all(|&cell| cell == PATH || cell == WALL || cell == GOAL),
            "{} has an unexpected cell value",
            geometry.id
        );
        let goal_count = geometry
            .matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&cell| cell == GOAL)
            .count();
        assert_eq!(goal_count, 1, "{} must have exactly one goal", geometry.id);
        let height = geometry.matrix.len();
        let border_is_wall = (0..width).all(|col| {
            geometry.matrix[0][col] == WALL && geometry.matrix[height - 1][col] == WALL
        }) && (0..height)
            .all(|row| geometry.matrix[row][0] == WALL && geometry.matrix[row][width - 1] == WALL);
        assert!(border_is_wall, "{} lacks a closed wall border", geometry.id);
    }
}

#[test]
fn unold_goal_is_reachable() {
    for geometry in UNOLD_GEOMETRIES {
        let goal = goal_cell(geometry.matrix);
        assert!(
            reachable_count(geometry.matrix, goal) > 1,
            "{} goal is isolated",
            geometry.id
        );
    }
}

#[test]
fn unold_transitions_match_geometry() {
    for geometry in UNOLD_GEOMETRIES {
        let height = geometry.matrix.len();
        let width = geometry.matrix[0].len();
        for row in 0..height {
            for col in 0..width {
                if geometry.matrix[row][col] != PATH {
                    continue;
                }
                for (action, (delta_row, delta_col)) in OFFSETS.iter().enumerate() {
                    let target_row = (row as isize + delta_row) as usize;
                    let target_col = (col as isize + delta_col) as usize;
                    let target = geometry.matrix[target_row][target_col];

                    let mut env = Maze::from_geometry(geometry, rng());
                    env.place_agent(row, col);
                    let outcome = env.step(action);
                    let position = env.agent_position();

                    if target == WALL {
                        assert_eq!(position, (row, col), "{} wall must block", geometry.id);
                        assert!(!outcome.terminated, "{} wall must not terminate", geometry.id);
                    } else {
                        assert_eq!(
                            position,
                            (target_row, target_col),
                            "{} must move into open cell",
                            geometry.id
                        );
                        assert_eq!(
                            outcome.terminated,
                            target == GOAL,
                            "{} termination must match goal cell",
                            geometry.id
                        );
                        if target == GOAL {
                            assert_eq!(outcome.reward, 1000.0, "{} goal reward", geometry.id);
                        }
                    }
                }
            }
        }
    }
}
