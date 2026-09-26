pub mod alcs;
pub mod pyalcs;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MazeSource {
    Pyalcs,
    Alcs,
}

pub struct MazeGeometry {
    pub id: &'static str,
    pub matrix: &'static [&'static [u8]],
    pub max_episode_steps: u32,
    pub source: MazeSource,
}

pub fn geometry_by_id(id: &str) -> Option<&'static MazeGeometry> {
    pyalcs::GEOMETRIES
        .iter()
        .chain(alcs::GEOMETRIES.iter())
        .find(|geometry| geometry.id == id)
}
