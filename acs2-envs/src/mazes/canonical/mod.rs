pub mod maze4;
pub mod maze5;
pub mod maze7;
pub mod woods1;
pub mod woods100;

pub use maze4::MAZE4;
pub use maze5::MAZE5;
pub use maze7::MAZE7;
pub use woods1::WOODS1;
pub use woods100::WOODS100;

use super::MazeGeometry;

pub const CANONICAL_GEOMETRIES: &[MazeGeometry] = &[MAZE4, MAZE5, MAZE7, WOODS1, WOODS100];
