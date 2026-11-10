use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::CoordinateGoalMaze;
use acs2_envs::maze::geometries::{alcs, research};
use acs2_envs::maze::topology::{MazeGeometryError, MazeTopology};
use acs2_envs::maze::Maze;
use acs2_envs::roles::{ResearchTask, PERFORMANCE_MAZES};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn all_24_research_matrices_equal_pinned_alcs_sources_and_known_benchmark_differences_are_exact() {
    let fixture: Value =
        serde_json::from_str(include_str!("../../fixtures/alcs_maze_geometries.json")).unwrap();
    assert_eq!(
        fixture["commit"],
        "ed0e76282a121fa2bad7dac82b744b2cc763eb72"
    );
    assert_eq!(fixture["files"].as_array().unwrap().len(), 24);
    assert_eq!(PERFORMANCE_MAZES.len(), 22);
    let mut deviations = 0;
    for entry in fixture["files"].as_array().unwrap() {
        assert_eq!(entry["sha256"].as_str().unwrap().len(), 64);
        let Some(ResearchTask::GoalMaze(geometry)) =
            ResearchTask::named(entry["task"].as_str().unwrap())
        else {
            panic!()
        };
        assert_eq!(
            serde_json::to_value(geometry.matrix).unwrap(),
            entry["matrix"]
        );
        let differences = entry["known_benchmark_differences"].as_array().unwrap();
        if !differences.is_empty() {
            deviations += 1;
            let benchmark = PERFORMANCE_MAZES
                .iter()
                .find(|g| g.id == entry["benchmark_geometry"])
                .unwrap();
            assert_ne!(benchmark.id, geometry.id);
            let actual:Vec<_>=geometry.matrix.iter().enumerate().flat_map(|(r,row)|row.iter().enumerate()
                .filter(move |&(c,&v)|v!=benchmark.matrix[r][c])
                .map(move |(c,&v)|serde_json::json!({"cell":[r,c],"alcs":v,"benchmark":benchmark.matrix[r][c]}))).collect();
            assert_eq!(&actual, differences);
        }
    }
    assert_eq!(deviations, 3);
}

#[test]
fn research_names_are_unique_complete_and_aliasing_is_computed_by_environment_perception() {
    let names: BTreeMap<String, String> = serde_json::from_str(include_str!(
        "../../experiments/per-goal-reward-ablation/maze-names.json"
    ))
    .unwrap();
    assert_eq!(names.len(), 29);
    assert_eq!(names.values().collect::<BTreeSet<_>>().len(), names.len());
    assert_eq!(names["maze7"], "Maze7-pyalcs");
    assert_eq!(names["woods1"], "Woods1-pyalcs");
    let entries: Value = serde_json::from_str(include_str!(
        "../../experiments/per-goal-reward-ablation/geometry.json"
    ))
    .unwrap();
    let mut unaliased = BTreeSet::new();
    assert_eq!(entries.as_array().unwrap().len(), 28);
    assert_eq!(research::ALCS_TASKS.len(), 22);
    for entry in entries.as_array().unwrap() {
        let task = entry["task"].as_str().unwrap();
        assert!(names.contains_key(task));
        let Some(ResearchTask::GoalMaze(g)) = ResearchTask::named(task) else {
            panic!()
        };
        assert_eq!(entry["id"], g.id);
        let topology = MazeTopology::multi_goal(g).unwrap();
        let mut counts = BTreeMap::new();
        for &cell in topology.walkable_cells() {
            *counts
                .entry(format!("{:?}", topology.perception_at(cell)))
                .or_insert(0) += 1;
        }
        let aliased: usize = counts.values().filter(|&&count| count > 1).sum();
        assert_eq!(entry["aliased_cells"], aliased);
        assert!(
            (entry["aliased_share"].as_f64().unwrap()
                - aliased as f64 / topology.walkable_cells().len() as f64)
                .abs()
                < 1e-15
        );
        if aliased == 0 {
            unaliased.insert(task);
        }
    }
    assert_eq!(
        unaliased,
        BTreeSet::from([
            "maze4",
            "maze5",
            "maze6",
            "maze7",
            "woods14",
            "mazema",
            "mazea_alcs",
            "mazef1_alcs",
            "mazef2_alcs"
        ])
    );
}

#[test]
fn two_landmarks_remain_visible_walkable_and_valid_goals_and_starts_only_in_multi_goal_paths() {
    for geometry in [&research::WOODS1015, &research::WOODS102] {
        assert!(matches!(
            MazeTopology::new(geometry),
            Err(MazeGeometryError::RewardCount)
        ));
        assert!(CoordinateGoalMaze::single_goal(
            geometry,
            Box::new(ChaChaRandomSource::from_seed(42))
        )
        .is_err());
        assert!(std::panic::catch_unwind(|| Maze::from_geometry(
            geometry,
            Box::new(ChaChaRandomSource::from_seed(42))
        ))
        .is_err());
        let mut maze = CoordinateGoalMaze::all_walkable(
            geometry,
            6,
            Box::new(ChaChaRandomSource::from_seed(42)),
        )
        .unwrap();
        for landmark in [(1, 3), (9, 3)] {
            assert!(maze.topology().walkable_cells().contains(&landmark));
            let goal = maze.goal_at(landmark);
            assert!(maze.goal_pool().contains(&goal));
            let start = maze.reset_at(landmark, maze.goal_at((2, 3)));
            assert_eq!(start.achieved, goal);
            let below = (landmark.0 - 1, landmark.1);
            if maze.topology().is_walkable(below) {
                assert!(format!("{:?}", maze.topology().perception_at(below)).contains("Token(57)"));
            }
        }
    }
    for geometry in [&alcs::WOODS1015, &alcs::WOODS102] {
        assert!(MazeTopology::new(geometry).is_ok());
    }
}
