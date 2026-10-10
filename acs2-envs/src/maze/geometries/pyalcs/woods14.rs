use crate::maze::geometries::{MazeGeometry, MazeSource};

pub const WOODS14: MazeGeometry = MazeGeometry {
    id: "Woods14-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        &[1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
        &[1, 9, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
    source: MazeSource::Pyalcs,
};
