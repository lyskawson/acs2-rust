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
    pyalcs::MAZE6,
    pyalcs::MAZE7,
    pyalcs::MAZEF3,
    pyalcs::MAZEB,
];

#[derive(Clone, Copy)]
pub enum ResearchTask {
    GoalMaze(&'static MazeGeometry),
    HandEye(usize),
    Taxi,
    BitFlipping(usize),
}

impl ResearchTask {
    pub fn named(name: &str) -> Option<Self> {
        match name {
            "maze4" => Some(Self::GoalMaze(&pyalcs::MAZE4)),
            "maze5" => Some(Self::GoalMaze(&pyalcs::MAZE5)),
            "maze6" => Some(Self::GoalMaze(&pyalcs::MAZE6)),
            "maze7" => Some(Self::GoalMaze(&pyalcs::MAZE7)),
            "mazef3" => Some(Self::GoalMaze(&pyalcs::MAZEF3)),
            "mazeb" => Some(Self::GoalMaze(&pyalcs::MAZEB)),
            "handeye3" => Some(Self::HandEye(3)),
            "handeye4" => Some(Self::HandEye(4)),
            "handeye5" => Some(Self::HandEye(5)),
            "taxi" => Some(Self::Taxi),
            _ => name
                .strip_prefix("bitflip")?
                .parse()
                .ok()
                .filter(|n| (1..=16).contains(n))
                .map(Self::BitFlipping),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValidationOracle {
    MazeParity,
    SingleGoalMazeBridge,
    HandEyeParity,
    TaxiTransitionParity,
    GoalPortCorridor,
}

pub const VALIDATION_ORACLES: &[ValidationOracle] = &[
    ValidationOracle::MazeParity,
    ValidationOracle::SingleGoalMazeBridge,
    ValidationOracle::HandEyeParity,
    ValidationOracle::TaxiTransitionParity,
    ValidationOracle::GoalPortCorridor,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PerformanceBenchmark {
    Multiplexer,
    P9Mazes,
    AlcsGeometries,
}

pub const PERFORMANCE_BENCHMARKS: &[PerformanceBenchmark] = &[
    PerformanceBenchmark::Multiplexer,
    PerformanceBenchmark::P9Mazes,
    PerformanceBenchmark::AlcsGeometries,
];
