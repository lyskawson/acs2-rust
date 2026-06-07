pub struct MazeGeometry {
    pub id: &'static str,
    pub matrix: &'static [&'static [u8]],
    pub max_episode_steps: u32,
}

pub const MAZE_GEOMETRIES: &[MazeGeometry] = &[];
