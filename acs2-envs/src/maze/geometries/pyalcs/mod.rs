pub mod maze4;
pub mod maze5;
pub mod maze6;
pub mod maze7;
pub mod mazeb;
pub mod mazef3;
pub mod woods1;
pub mod woods100;

pub use maze4::MAZE4;
pub use maze5::MAZE5;
pub use maze6::MAZE6;
pub use maze7::MAZE7;
pub use mazeb::MAZEB;
pub use mazef3::MAZEF3;
pub use woods1::WOODS1;
pub use woods100::WOODS100;

use super::MazeGeometry;

pub const GEOMETRIES: &[MazeGeometry] =
    &[MAZE4, MAZE5, MAZE6, MAZE7, MAZEB, MAZEF3, WOODS1, WOODS100];
