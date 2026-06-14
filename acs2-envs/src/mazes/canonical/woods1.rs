use crate::mazes::{MazeGeometry, MazeSource};

pub const WOODS1: MazeGeometry = MazeGeometry {
    id: "Woods1-v0",
    matrix: &[
        &[1, 1, 1, 1, 1],
        &[1, 0, 0, 9, 1],
        &[1, 0, 0, 0, 1],
        &[1, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
    source: MazeSource::Pyalcs,
};
