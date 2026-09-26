use crate::maze::geometries::{alcs, pyalcs, MazeGeometry};

pub const VALIDATION_MAZES: &[MazeGeometry] = &[
    pyalcs::MAZE4,
    pyalcs::MAZE5,
    pyalcs::MAZE7,
    pyalcs::WOODS1,
    pyalcs::WOODS100,
];

pub const PERFORMANCE_MAZES: &[MazeGeometry] = alcs::GEOMETRIES;

pub const RESEARCH_MAZES: &[MazeGeometry] = &[
    pyalcs::MAZE4,
    pyalcs::MAZE5,
    pyalcs::MAZE7,
    alcs::MAZEF3,
    alcs::MAZEB,
];

pub const RESEARCH_TASKS: &[&str] = &["GoalMaze", "HandEye", "Taxi", "BitFlipping"];
pub const VALIDATION_ORACLES: &[&str] = &[
    "pyalcs maze parity and episode differential",
    "single-goal maze bridge",
    "HandEye simulator parity",
    "gym Taxi transition table parity",
    "goal-port corridor",
];
pub const PERFORMANCE_BENCHMARKS: &[&str] = &[
    "multiplexer",
    "P9 five-maze benchmark",
    "ALCS 22-geometry comparison",
];
