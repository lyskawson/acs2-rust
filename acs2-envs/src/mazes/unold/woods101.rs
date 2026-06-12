use crate::mazes::{MazeGeometry, MazeSource};

pub const WOODS101: MazeGeometry = MazeGeometry {
    id: "Woods101-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
