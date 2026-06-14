use crate::mazes::{MazeGeometry, MazeSource};

pub const WOODS1: MazeGeometry = MazeGeometry {
    id: "Woods1-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 1, 1, 9, 0, 0, 1],
        &[1, 1, 1, 1, 0, 0, 1],
        &[1, 1, 1, 1, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
