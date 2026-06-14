use crate::mazes::{MazeGeometry, MazeSource};

pub const MAZE5: MazeGeometry = MazeGeometry {
    id: "Maze5-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 9, 1],
        &[1, 0, 0, 1, 0, 1, 1, 0, 1],
        &[1, 0, 1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 1, 1, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 0, 1, 1],
        &[1, 0, 1, 0, 0, 1, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
    source: MazeSource::Pyalcs,
};
