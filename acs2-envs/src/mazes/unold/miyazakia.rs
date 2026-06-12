use crate::mazes::{MazeGeometry, MazeSource};

pub const MIYAZAKIA: MazeGeometry = MazeGeometry {
    id: "MiyazakiA-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 1, 0, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 0, 0, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 1, 0, 0, 0, 1],
        &[1, 0, 0, 9, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
