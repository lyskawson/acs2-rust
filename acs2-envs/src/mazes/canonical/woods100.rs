use crate::mazes::{MazeGeometry, MazeSource};

pub const WOODS100: MazeGeometry = MazeGeometry {
    id: "Woods100-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 9, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 500,
    source: MazeSource::Pyalcs,
};
