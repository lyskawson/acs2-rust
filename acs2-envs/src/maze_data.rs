pub struct MazeGeometry {
    pub id: &'static str,
    pub matrix: &'static [&'static [u8]],
    pub max_episode_steps: u32,
}

pub const MAZE4: MazeGeometry = MazeGeometry {
    id: "Maze4-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 1, 0, 0, 9, 1],
        &[1, 1, 0, 0, 1, 0, 0, 1],
        &[1, 1, 0, 1, 0, 0, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 1],
        &[1, 1, 0, 1, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
};

pub const MAZE5: MazeGeometry = MazeGeometry {
    id: "Maze5-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 0, 9, 1],
        &[1, 0, 0, 1, 0, 1, 1, 0, 1],
        &[1, 0, 1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 1, 1, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 0, 1, 1],
        &[1, 0, 1, 0, 0, 1, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
};

pub const MAZE7: MazeGeometry = MazeGeometry {
    id: "Maze7-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 0, 1, 9, 1],
        &[1, 0, 0, 1, 0, 1, 1, 0, 1],
        &[1, 0, 1, 0, 0, 1, 0, 0, 1],
        &[1, 0, 0, 0, 1, 1, 0, 0, 1],
        &[1, 0, 1, 0, 1, 0, 0, 1, 1],
        &[1, 0, 1, 0, 0, 0, 0, 0, 1],
        &[1, 0, 0, 0, 0, 0, 1, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
};

pub const WOODS1: MazeGeometry = MazeGeometry {
    id: "Woods1-v0",
    matrix: &[
        &[1, 1, 1, 1, 1],
        &[1, 0, 0, 9, 1],
        &[1, 0, 0, 0, 1],
        &[1, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1],
    ],
    max_episode_steps: 50,
};

pub const WOODS100: MazeGeometry = MazeGeometry {
    id: "Woods100-v0",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 9, 0, 0, 0, 1],
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 500,
};

pub const MAZE_GEOMETRIES: &[MazeGeometry] = &[MAZE4, MAZE5, MAZE7, WOODS1, WOODS100];

pub fn geometry_by_id(id: &str) -> Option<&'static MazeGeometry> {
    MAZE_GEOMETRIES.iter().find(|geometry| geometry.id == id)
}
