use crate::mazes::{MazeGeometry, MazeSource};

pub const MAZEE2: MazeGeometry = MazeGeometry {
    id: "MazeE2-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 9, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
