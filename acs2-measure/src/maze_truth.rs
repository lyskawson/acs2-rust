use acs2_envs::maze::topology::{Cell, MazeTopology};
use serde_json::{json, Value};

pub const GAMMA: f64 = 0.95;
pub const REWARD: f64 = 1000.0;

#[derive(Debug)]
pub struct PairTruth {
    pub successors: Vec<Cell>,
    pub shortest_distance: Option<u32>,
    pub value: f64,
    pub action_values: Vec<f64>,
    pub optimal_actions: Vec<usize>,
}

pub fn discounted_value(distance: Option<u32>, cap: u32) -> f64 {
    match distance {
        Some(distance) if distance > 0 && distance <= cap => {
            (1..distance).fold(REWARD, |value, _| GAMMA * value)
        }
        _ => 0.0,
    }
}

pub fn pair_truth(topology: &MazeTopology, start: Cell, goal: Cell, cap: u32) -> PairTruth {
    assert_ne!(start, goal);
    let successors: Vec<_> = (0..8)
        .map(|action| topology.next_cell(start, action))
        .collect();
    let action_values: Vec<_> = successors
        .iter()
        .map(|&next| {
            if cap == 0 {
                0.0
            } else if next == goal {
                REWARD
            } else {
                GAMMA * discounted_value(topology.shortest_distance(next, goal), cap - 1)
            }
        })
        .collect();
    let best = action_values.iter().copied().fold(0.0, f64::max);
    let optimal_actions = action_values
        .iter()
        .enumerate()
        .filter_map(|(action, &value)| (value == best).then_some(action))
        .collect();
    let shortest_distance = topology.shortest_distance(start, goal);
    let value = discounted_value(shortest_distance, cap);
    assert_eq!(
        best.to_bits(),
        value.to_bits(),
        "Bellman optimum differs from shortest-path value"
    );
    PairTruth {
        successors,
        shortest_distance,
        value,
        action_values,
        optimal_actions,
    }
}

pub fn model(topology: &MazeTopology, task: &str, cap: u32) -> Value {
    let pairs: Vec<_> = topology
        .walkable_cells()
        .iter()
        .flat_map(|&start| {
            topology
                .walkable_cells()
                .iter()
                .filter(move |&&goal| goal != start)
                .map(move |&goal| {
                    let truth = pair_truth(topology, start, goal, cap);
                    json!({"start": start, "goal": goal, "successors": truth.successors,
                "shortest_distance": truth.shortest_distance, "value": truth.value,
                "action_values": truth.action_values, "optimal_actions": truth.optimal_actions})
                })
        })
        .collect();
    json!({"task": task, "cap": cap, "gamma": GAMMA, "reward": REWARD, "actions": 8,
        "source": "acs2_envs::maze::topology::MazeTopology", "pairs": pairs})
}
