use acs2_core::environment::Environment;
use acs2_core::perception::Perception;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_envs::maze::{Maze, MAZE_PERCEPTION_LEN};
use acs2_envs::maze_data::{geometry_by_id, MazeGeometry};
use serde_json::Value;

fn load_probes() -> Value {
    let path = format!("{}/../fixtures/maze_probes.json", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn rng() -> Box<dyn RandomSource> {
    Box::new(ChaChaRandomSource::from_seed(0))
}

fn expected_perception(value: &Value) -> Perception<MAZE_PERCEPTION_LEN> {
    let array = value.as_array().unwrap();
    assert_eq!(array.len(), MAZE_PERCEPTION_LEN);
    let symbols =
        core::array::from_fn(|index| Symbol::Token(array[index].as_str().unwrap().as_bytes()[0]));
    Perception::new(symbols)
}

fn assert_grid_matches(geometry: &MazeGeometry, grid: &Value, id: &str) {
    let rows = grid.as_array().unwrap();
    assert_eq!(rows.len(), geometry.matrix.len(), "{id} row count");
    for (row_index, row) in rows.iter().enumerate() {
        let cells = row.as_array().unwrap();
        let expected_row = geometry.matrix[row_index];
        assert_eq!(cells.len(), expected_row.len(), "{id} row {row_index} width");
        for (col_index, cell) in cells.iter().enumerate() {
            assert_eq!(
                cell.as_u64().unwrap() as u8,
                expected_row[col_index],
                "{id} cell ({row_index},{col_index})"
            );
        }
    }
}

#[test]
fn rust_maze_matches_pyalcs_probes() {
    let data = load_probes();
    let mazes = data["mazes"].as_array().unwrap();

    for maze in mazes {
        let id = maze["id"].as_str().unwrap();
        let geometry = geometry_by_id(id).unwrap_or_else(|| panic!("unknown maze {id}"));

        assert_grid_matches(geometry, &maze["grid"], id);
        assert_eq!(
            maze["max_episode_steps"].as_u64().unwrap() as u32,
            geometry.max_episode_steps,
            "{id} cap"
        );

        for probe in maze["probes"].as_array().unwrap() {
            let row = probe["row"].as_u64().unwrap() as usize;
            let col = probe["col"].as_u64().unwrap() as usize;
            let action = probe["action"].as_u64().unwrap() as usize;

            let mut env = Maze::from_geometry(geometry, rng());
            env.place_agent(row, col);

            assert_eq!(
                env.perception(),
                expected_perception(&probe["perception_before"]),
                "{id} perception_before at ({row},{col}) action {action}"
            );

            let outcome = env.step(action);

            assert_eq!(
                outcome.observation,
                expected_perception(&probe["perception_after"]),
                "{id} perception_after at ({row},{col}) action {action}"
            );
            assert_eq!(
                outcome.reward,
                probe["reward"].as_i64().unwrap() as f64,
                "{id} reward at ({row},{col}) action {action}"
            );
            assert_eq!(
                outcome.terminated,
                probe["done"].as_bool().unwrap(),
                "{id} terminated at ({row},{col}) action {action}"
            );
        }
    }
}

fn boundary_maze(max_episode_steps: u32) -> Maze {
    let matrix = vec![
        vec![1, 1, 1, 1],
        vec![1, 0, 9, 1],
        vec![1, 1, 1, 1],
    ];
    Maze::new(matrix, max_episode_steps, rng())
}

#[test]
fn truncation_fires_exactly_at_cap_on_blocked_moves() {
    let geometry = geometry_by_id("Maze4-v0").unwrap();
    let mut env = Maze::from_geometry(geometry, rng());
    env.place_agent(1, 1);

    let action_into_wall = 0;
    for step_index in 1..=geometry.max_episode_steps {
        let outcome = env.step(action_into_wall);
        assert!(!outcome.terminated, "blocked move must not terminate");
        assert_eq!(outcome.reward, 0.0);
        if step_index < geometry.max_episode_steps {
            assert!(!outcome.truncated, "must not truncate before cap");
        } else {
            assert!(outcome.truncated, "must truncate at cap");
        }
    }
}

#[test]
fn termination_wins_over_truncation_at_the_boundary() {
    let mut env = boundary_maze(1);
    env.place_agent(1, 1);

    let action_east = 2;
    let outcome = env.step(action_east);

    assert!(outcome.terminated, "entering reward terminates");
    assert!(!outcome.truncated, "terminated step is not truncated");
    assert_eq!(outcome.reward, 1000.0);
}

#[test]
fn truncation_fires_when_goal_not_reached_at_cap() {
    let mut env = boundary_maze(1);
    env.place_agent(1, 1);

    let action_into_wall = 0;
    let outcome = env.step(action_into_wall);

    assert!(!outcome.terminated);
    assert!(outcome.truncated, "non-terminal cap step truncates");
    assert_eq!(outcome.reward, 0.0);
}

#[test]
fn reset_places_agent_on_a_path_cell() {
    let geometry = geometry_by_id("Woods1-v0").unwrap();
    let mut env = Maze::from_geometry(geometry, rng());
    env.reset();
    let (row, col) = env.agent_position();
    assert_eq!(geometry.matrix[row][col], 0, "reset must land on a path cell");
}
