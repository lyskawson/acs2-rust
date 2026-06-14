use crate::mazes::{MazeGeometry, MazeSource};

pub const MAZEF4: MazeGeometry = MazeGeometry {
    id: "MazeF4-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 9, 1],
        &[1, 0, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 1, 1],
        &[1, 0, 1, 1, 1, 1, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
