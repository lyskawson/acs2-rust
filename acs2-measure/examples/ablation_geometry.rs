use acs2_envs::maze::geometries::research::ALCS_TASKS;
use acs2_envs::maze::topology::MazeTopology;
use acs2_envs::roles::ResearchTask;
use serde_json::json;
use std::collections::BTreeMap;

fn main() {
    let mut tasks: Vec<_> = ["maze4", "maze5", "maze6", "maze7", "woods14", "mazema"]
        .into_iter()
        .collect();
    tasks.extend(ALCS_TASKS.iter().map(|(task, _)| *task));
    let entries: Vec<_> = tasks
        .into_iter()
        .map(|task| {
            let Some(ResearchTask::GoalMaze(geometry)) = ResearchTask::named(task) else {
                unreachable!()
            };
            let topology = MazeTopology::multi_goal(geometry).unwrap();
            let cells = topology.walkable_cells();
            let perceptions: Vec<_> = cells
                .iter()
                .map(|&cell| format!("{:?}", topology.perception_at(cell)))
                .collect();
            let mut counts = BTreeMap::new();
            for perception in &perceptions {
                *counts.entry(perception).or_insert(0) += 1;
            }
            let aliased = perceptions
                .iter()
                .filter(|perception| counts[perception] > 1)
                .count();
            let distances: Vec<Vec<_>> = cells
                .iter()
                .map(|&start| {
                    cells
                        .iter()
                        .map(|&goal| topology.shortest_distance(start, goal))
                        .collect()
                })
                .collect();
            let successors: Vec<Vec<_>> = cells
                .iter()
                .map(|&cell| {
                    (0..8)
                        .map(|action| {
                            cells
                                .iter()
                                .position(|&next| next == topology.next_cell(cell, action))
                                .unwrap()
                        })
                        .collect()
                })
                .collect();
            json!({"task": task, "id": geometry.id, "matrix": geometry.matrix,
            "walkable_cells": cells, "perceptions": perceptions, "aliased_cells": aliased,
            "aliased_share": aliased as f64 / cells.len() as f64, "distances": distances,
            "successors": successors})
        })
        .collect();
    println!("{}", serde_json::to_string_pretty(&entries).unwrap());
}
