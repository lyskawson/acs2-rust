use crate::maze::geometries::{MazeGeometry, MazeSource};

pub const MAZEB: MazeGeometry = MazeGeometry {
    id: "MazeB-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 1, 1, 1],
        &[1, 0, 0, 1, 0, 1, 9, 1],
        &[1, 1, 0, 1, 0, 0, 0, 1],
        &[1, 1, 0, 0, 1, 0, 0, 1],
        &[1, 0, 1, 0, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
    source: MazeSource::Pyalcs,
};
