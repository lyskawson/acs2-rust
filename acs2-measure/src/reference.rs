use acs2_core::goal::Goal;

use crate::task::{Pair, Task};

#[derive(Clone, Copy, Debug)]
pub struct Reference {
    pub random_success: f64,
    pub reachable_within_cap: f64,
    pub reachable_after_cap: f64,
    pub unreachable_or_ambiguous: f64,
}

pub fn reference<T, const S: usize, const G: usize, const M: usize>(
    task: &T,
    pairs: &[Pair<T::State, G>],
) -> Reference
where
    T: Task<S, G, M>,
{
    let mut result = Reference {
        random_success: 0.0,
        reachable_within_cap: 0.0,
        reachable_after_cap: 0.0,
        unreachable_or_ambiguous: 0.0,
    };
    result.random_success = task
        .analytical_random_success(pairs)
        .unwrap_or_else(|| generic_random_success(task, pairs));
    if let Some((within, after, missing)) = task.analytical_reachability() {
        result.reachable_within_cap = within;
        result.reachable_after_cap = after;
        result.unreachable_or_ambiguous = missing;
        return result;
    }
    for &(state, goal, weight) in pairs {
        match task.distance(state, &goal) {
            Some(distance) if distance <= task.cap() => result.reachable_within_cap += weight,
            Some(_) => result.reachable_after_cap += weight,
            None => result.unreachable_or_ambiguous += weight,
        }
    }
    result
}

fn generic_random_success<T, const S: usize, const G: usize, const M: usize>(
    task: &T,
    pairs: &[Pair<T::State, G>],
) -> f64
where
    T: Task<S, G, M>,
{
    let states = task.states();
    let kernel: Vec<Vec<usize>> = states
        .iter()
        .map(|&state| {
            (0..task.actions())
                .map(|action| {
                    states
                        .iter()
                        .position(|&candidate| candidate == task.next(state, action))
                        .expect("closed task states")
                })
                .collect()
        })
        .collect();
    let mut goals: Vec<Goal<G>> = Vec::new();
    for &(_, goal, _) in pairs {
        if !goals.contains(&goal) {
            goals.push(goal);
        }
    }
    let probabilities: Vec<Vec<f64>> = goals
        .iter()
        .map(|goal| {
            let reached: Vec<bool> = states
                .iter()
                .map(|&state| task.achieved(state) == *goal)
                .collect();
            let mut current: Vec<f64> = reached
                .iter()
                .map(|&hit| if hit { 1.0 } else { 0.0 })
                .collect();
            for _ in 0..task.cap() {
                current = (0..states.len())
                    .map(|index| {
                        if reached[index] {
                            1.0
                        } else {
                            kernel[index].iter().map(|&next| current[next]).sum::<f64>()
                                / task.actions() as f64
                        }
                    })
                    .collect();
            }
            current
        })
        .collect();
    pairs
        .iter()
        .map(|&(state, goal, weight)| {
            let state_index = states
                .iter()
                .position(|&candidate| candidate == state)
                .expect("evaluated start");
            let goal_index = goals
                .iter()
                .position(|&candidate| candidate == goal)
                .expect("evaluated goal");
            weight * probabilities[goal_index][state_index]
        })
        .sum()
}
