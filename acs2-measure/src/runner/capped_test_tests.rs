use super::*;
use crate::task::MazeTask;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::{MazeGeometry, MazeSource};
use std::cell::RefCell;

static CORRIDOR: MazeGeometry = MazeGeometry {
    id: "test-corridor",
    matrix: &[
        &[1, 1, 1, 1, 1, 1, 1],
        &[1, 0, 0, 0, 0, 9, 1],
        &[1, 1, 1, 1, 1, 1, 1],
    ],
    max_episode_steps: 2,
    source: MazeSource::Pyalcs,
};

#[derive(Default)]
struct East {
    decisions: RefCell<Vec<(Perception<10>, usize, u64)>>,
}

impl GoalAgent<8, 2, 10> for East {
    fn name(&self) -> &'static str {
        "east"
    }

    fn declared_policy(&self) -> &'static str {
        "east"
    }

    fn train_episode<E: MeasuredEnvironment<8, 2, 10>>(&mut self, _env: &mut E, _time: u64) {
        unreachable!();
    }

    fn eval_action(&self, state: &Perception<10>, rng: &mut dyn RandomSource) -> (usize, f64) {
        let draw = rng.gen_unit();
        self.decisions
            .borrow_mut()
            .push((*state, 2, draw.to_bits()));
        (2, draw)
    }

    fn online_updates(&self) -> u64 {
        0
    }

    fn replay_updates(&self) -> u64 {
        0
    }

    fn replay_samples(&self) -> usize {
        0
    }

    fn population_classifiers(&self) -> usize {
        0
    }

    fn population_numerosity(&self) -> u32 {
        0
    }

    fn population_logical_bytes(&self) -> usize {
        0
    }

    fn population_mark_entries(&self) -> usize {
        0
    }

    fn replay_logical_bytes(&self) -> usize {
        0
    }

    fn agent_parameters(&self) -> Value {
        Value::Null
    }
}

#[test]
fn paired_caps_share_prefixes_and_freeze_the_registered_stream() {
    let maze = |cap| {
        MazeTask::<Coordinates, 2>::new(
            CORRIDOR.id.to_owned(),
            &CORRIDOR,
            (1..=5).map(|col| (1, col)).collect(),
            cap,
            "coordinates",
        )
    };
    let short = maze(2);
    let long = maze(100);
    let pairs = [
        ((1, 1), short.template.goal_at((1, 2)), 0.2),
        ((1, 1), short.template.goal_at((1, 4)), 0.3),
        ((1, 5), short.template.goal_at((1, 1)), 0.4),
        ((1, 2), short.template.goal_at((1, 4)), 0.1),
    ];
    let control = East::default();
    let expected = evaluate(&short, &control, &pairs, 42, 0.95, true);
    let agent = East::default();
    let paired = evaluate_with_test_cap(&short, &agent, &pairs, 42, 0.95, true, Some(&long));
    assert_eq!(paired.starts, expected.starts);

    let control = control.decisions.into_inner();
    let decisions = agent.decisions.into_inner();
    let (mut reference_cursor, mut cursor) = (0, 0);
    for (registered, extended) in [(1, 1), (2, 3), (2, 100), (2, 2)] {
        let prefix = &decisions[cursor..cursor + registered];
        assert_eq!(
            prefix,
            &control[reference_cursor..reference_cursor + registered]
        );
        cursor += registered;
        assert_eq!(prefix, &decisions[cursor..cursor + registered]);
        cursor += extended;
        reference_cursor += registered;
    }
    assert_eq!(cursor, decisions.len());
    assert_eq!(reference_cursor, control.len());

    let result = paired.capped_test.unwrap();
    let close = |actual: f64, expected: f64| assert!((actual - expected).abs() < 1e-12);
    close(paired.success, 0.3);
    close(result.extended_success, 0.6);
    close(result.registered_steps, 1.8);
    close(result.extended_steps, 41.3);
    close(result.shortest_path, 2.9);
    assert_eq!(
        result.starts.unwrap(),
        vec![
            json!({"success": true, "steps": 1}),
            json!({"success": true, "steps": 3}),
            json!({"success": false, "steps": 100}),
            json!({"success": true, "steps": 2}),
        ]
    );
}
