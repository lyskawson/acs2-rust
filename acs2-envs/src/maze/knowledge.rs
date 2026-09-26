use acs2_core::goal::{Goal, GoalLayout};
use acs2_core::knowledge::Transition;
use acs2_core::symbol::Symbol;

use super::geometries::MazeGeometry;
use super::topology::MazeTopology;
use super::MAZE_PERCEPTION_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MazeKnowledgeSet {
    PyalcsPaths,
    MultiGoalWalkable,
}

pub fn pyalcs_transitions(geometry: &MazeGeometry) -> Vec<Transition<MAZE_PERCEPTION_LEN>> {
    transitions(geometry, MazeKnowledgeSet::PyalcsPaths)
}

pub fn multi_goal_transitions(geometry: &MazeGeometry) -> Vec<Transition<MAZE_PERCEPTION_LEN>> {
    transitions(geometry, MazeKnowledgeSet::MultiGoalWalkable)
}

pub fn transitions(
    geometry: &MazeGeometry,
    set: MazeKnowledgeSet,
) -> Vec<Transition<MAZE_PERCEPTION_LEN>> {
    let topology = MazeTopology::new(geometry).expect("knowledge requires a valid maze geometry");
    let starts = match set {
        MazeKnowledgeSet::PyalcsPaths => topology.path_cells(),
        MazeKnowledgeSet::MultiGoalWalkable => topology.walkable_cells(),
    };
    let mut result = Vec::new();
    for &start in starts {
        for action in 0..MAZE_PERCEPTION_LEN {
            let next = topology.next_cell(start, action);
            if next != start {
                result.push(Transition::new(
                    topology.perception_at(start),
                    action,
                    topology.perception_at(next),
                ));
            }
        }
    }
    result
}

pub fn goal_transitions<const G: usize, const M: usize>(
    geometry: &MazeGeometry,
    set: MazeKnowledgeSet,
) -> Vec<Transition<M>> {
    let wildcard_goal = Goal::new([Symbol::Wildcard; G]);
    transitions(geometry, set)
        .into_iter()
        .map(|transition| {
            Transition::new(
                GoalLayout::<MAZE_PERCEPTION_LEN, G, M>::join(&transition.p0, &wildcard_goal),
                transition.action,
                GoalLayout::<MAZE_PERCEPTION_LEN, G, M>::join(&transition.p1, &wildcard_goal),
            )
        })
        .collect()
}
