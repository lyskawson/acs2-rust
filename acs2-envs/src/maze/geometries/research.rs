use super::{alcs, MazeGeometry, MazeSource};

pub const MAZEF1: MazeGeometry = MazeGeometry {
    id: "MazeF1-alcs-research",
    matrix: &[
        &[1, 1, 1, 1],
        &[1, 0, 9, 1],
        &[1, 0, 1, 1],
        &[1, 0, 0, 1],
        &[1, 0, 1, 1],
        &[1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::Alcs,
};

pub const MIYAZAKIA: MazeGeometry = MazeGeometry {
    id: "MiyazakiA-alcs-research",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 1, 0, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 0, 0, 0, 1, 9, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 1, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::Alcs,
};

pub const WOODS1015: MazeGeometry = MazeGeometry {
    id: "Woods101_5-alcs-research",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 0, 1, 0, 1, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 0, 1, 0, 1, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::Alcs,
};

pub const WOODS102: MazeGeometry = MazeGeometry {
    id: "Woods102-alcs-research",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 1],
        &[1, 0, 1, 9, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 200,
    source: MazeSource::Alcs,
};

pub const ALCS_TASKS: &[(&str, MazeGeometry)] = &[
    ("cassandra4x4_alcs", alcs::CASSANDRA4X4),
    ("littman57_alcs", alcs::LITTMAN57),
    ("littman89_alcs", alcs::LITTMAN89),
    ("maze10_alcs", alcs::MAZE10),
    ("maze7_alcs", alcs::MAZE7),
    ("mazea_alcs", alcs::MAZEA),
    ("mazeb_alcs", alcs::MAZEB),
    ("mazed_alcs", alcs::MAZED),
    ("mazee1_alcs", alcs::MAZEE1),
    ("mazee2_alcs", alcs::MAZEE2),
    ("mazee3_alcs", alcs::MAZEE3),
    ("mazef1_alcs", MAZEF1),
    ("mazef2_alcs", alcs::MAZEF2),
    ("mazef3_alcs", alcs::MAZEF3),
    ("mazef4_alcs", alcs::MAZEF4),
    ("miyazakia_alcs", MIYAZAKIA),
    ("miyazakib_alcs", alcs::MIYAZAKIB),
    ("woods1_alcs", alcs::WOODS1),
    ("woods100_alcs", alcs::WOODS100),
    ("woods101_alcs", alcs::WOODS101),
    ("woods101_5_alcs", WOODS1015),
    ("woods102_alcs", WOODS102),
];
