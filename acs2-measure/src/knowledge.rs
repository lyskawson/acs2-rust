use acs2_core::goal::GoalLayout;
use acs2_core::support::knowledge::Transition;
use acs2_envs::goal::maze::{GoalMaze, MazeGoalEncoding};

pub fn maze_transitions<E: MazeGoalEncoding<G>, const G: usize, const M: usize>(
    maze: &GoalMaze<E, G>,
) -> Vec<Transition<M>> {
    let topology = maze.topology();
    let mut transitions = Vec::new();
    for &goal_cell in topology.walkable_cells() {
        let goal = maze.goal_at(goal_cell);
        for &start in topology.walkable_cells() {
            if start == goal_cell {
                continue;
            }
            for action in 0..8 {
                let next = topology.next_cell(start, action);
                if next != start {
                    transitions.push(Transition::new(
                        GoalLayout::<8, G, M>::join(&topology.perception_at(start), &goal),
                        action,
                        GoalLayout::<8, G, M>::join(&topology.perception_at(next), &goal),
                    ));
                }
            }
        }
    }
    transitions
}
