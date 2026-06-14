use crate::mazes::{MazeGeometry, MazeSource};

pub const WOODS102: MazeGeometry = MazeGeometry {
    id: "Woods102-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
