use super::MazeGeometry;

pub mod cassandra4x4;
pub mod littman57;
pub mod littman89;
pub mod maze10;
pub mod maze4;
pub mod maze7;
pub mod mazea;
pub mod mazeb;
pub mod mazed;
pub mod mazee1;
pub mod mazee2;
pub mod mazee3;
pub mod mazef2;
pub mod mazef3;
pub mod mazef4;
pub mod miyazakia;
pub mod miyazakib;
pub mod woods1;
pub mod woods100;
pub mod woods101;
pub mod woods1015;
pub mod woods102;

pub use cassandra4x4::CASSANDRA4X4;
pub use littman57::LITTMAN57;
pub use littman89::LITTMAN89;
pub use maze10::MAZE10;
pub use maze4::MAZE4;
pub use maze7::MAZE7;
pub use mazea::MAZEA;
pub use mazeb::MAZEB;
pub use mazed::MAZED;
pub use mazee1::MAZEE1;
pub use mazee2::MAZEE2;
pub use mazee3::MAZEE3;
pub use mazef2::MAZEF2;
pub use mazef3::MAZEF3;
pub use mazef4::MAZEF4;
pub use miyazakia::MIYAZAKIA;
pub use miyazakib::MIYAZAKIB;
pub use woods1::WOODS1;
pub use woods100::WOODS100;
pub use woods101::WOODS101;
pub use woods1015::WOODS1015;
pub use woods102::WOODS102;

pub const UNOLD_GEOMETRIES: &[MazeGeometry] = &[
    CASSANDRA4X4,
    LITTMAN57,
    LITTMAN89,
    MAZE10,
    MAZE4,
    MAZE7,
    MAZEA,
    MAZEB,
    MAZED,
    MAZEE1,
    MAZEE2,
    MAZEE3,
    MAZEF2,
    MAZEF3,
    MAZEF4,
    MIYAZAKIA,
    MIYAZAKIB,
    WOODS1,
    WOODS100,
    WOODS101,
    WOODS1015,
    WOODS102,
];
