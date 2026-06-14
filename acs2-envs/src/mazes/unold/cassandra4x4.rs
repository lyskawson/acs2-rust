use crate::mazes::{MazeGeometry, MazeSource};

pub const CASSANDRA4X4: MazeGeometry = MazeGeometry {
    id: "Cassandra4x4-ounold",
    matrix: &[
        &[1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 9, 1],
        &[1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::OunoldAlcs,
};
