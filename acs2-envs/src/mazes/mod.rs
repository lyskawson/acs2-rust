pub mod canonical;
pub mod unold;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MazeSource {
    Pyalcs,
    OunoldAlcs,
}

pub struct MazeGeometry {
    pub id: &'static str,
    pub matrix: &'static [&'static [u8]],
    pub max_episode_steps: u32,
    pub source: MazeSource,
}

pub const MAZE_GEOMETRIES: &[MazeGeometry] = canonical::CANONICAL_GEOMETRIES;
pub const UNOLD_GEOMETRIES: &[MazeGeometry] = unold::UNOLD_GEOMETRIES;

pub fn geometry_by_id(id: &str) -> Option<&'static MazeGeometry> {
    MAZE_GEOMETRIES
        .iter()
        .chain(UNOLD_GEOMETRIES.iter())
        .find(|geometry| geometry.id == id)
}
