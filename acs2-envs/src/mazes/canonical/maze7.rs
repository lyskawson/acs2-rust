use crate::mazes::{MazeGeometry, MazeSource};

pub const MAZE7: MazeGeometry = MazeGeometry {
    id: "Maze7-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 1, 9, 1],
        &[1, 0, 0, 1, 0, 1, 1, 0, 1],
        &[1, 0, 1, 0, 0, 1, 0, 0, 1],
        &[1, 0, 0, 0, 1, 1, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 0, 1, 1],
        &[1, 0, 1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
    source: MazeSource::Pyalcs,
};
