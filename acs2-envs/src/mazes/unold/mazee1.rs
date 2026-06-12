use crate::mazes::{MazeGeometry, MazeSource};

pub const MAZEE1: MazeGeometry = MazeGeometry {
    id: "MazeE1-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 0, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 9, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 0, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
