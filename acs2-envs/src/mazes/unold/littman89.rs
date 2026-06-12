use crate::mazes::{MazeGeometry, MazeSource};

pub const LITTMAN89: MazeGeometry = MazeGeometry {
    id: "Littman89-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 1, 0, 0, 0, 0, 0, 1, 1],
        &[1, 0, 0, 1, 0, 1, 0, 0, 1],
        &[1, 1, 0, 1, 0, 1, 0, 1, 1],
        &[1, 0, 0, 1, 0, 1, 0, 9, 1],
        &[1, 1, 0, 0, 0, 0, 0, 1, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
