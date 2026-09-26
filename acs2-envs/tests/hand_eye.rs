use acs2_core::environment::Environment;
use acs2_core::goal::{GoalConditioned, GoalEnvironment};
use acs2_core::perception::Perception;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_envs::goal::hand_eye::{position_goal, HandEye, HandEyeState};
use serde_json::Value;

fn rng(seed: u64) -> Box<dyn RandomSource> {
    Box::new(ChaChaRandomSource::from_seed_and_stream(seed, 1))
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/hand_eye.json")).unwrap()
}

fn state(value: &Value) -> HandEyeState {
    let coordinate = |i| value[i].as_u64().unwrap() as usize;
    HandEyeState {
        gripper: (coordinate(0), coordinate(1)),
        block: (coordinate(2), coordinate(3)),
        held: coordinate(4) == 1,
    }
}

fn perception<const S: usize>(value: &Value) -> Perception<S> {
    let bytes = value.as_str().unwrap().as_bytes();
    assert_eq!(bytes.len(), S);
    Perception::new(core::array::from_fn(|i| Symbol::Token(bytes[i])))
}

fn parity<const SIDE: usize, const S: usize>() {
    let data = fixture();
    let entry = data["grids"]
        .as_array()
        .unwrap()
        .iter()
        .find(|x| x["side"].as_u64() == Some(SIDE as u64))
        .unwrap();
    let mut env = HandEye::<SIDE, S>::new(1, rng(0));
    assert_eq!(env.states().count(), SIDE * SIDE * (SIDE * SIDE + 1));
    assert_eq!(
        entry["probes"].as_array().unwrap().len(),
        env.states().count() * 6
    );
    let goals = env.goal_pool().to_vec();
    for probe in entry["probes"].as_array().unwrap() {
        let start = state(&probe[0]);
        let action = probe[1].as_u64().unwrap() as usize;
        let expected = state(&probe[2]);
        assert_eq!(start.after_action::<SIDE>(action), expected);
        assert_eq!(start.perception::<SIDE, S>(), perception::<S>(&probe[3]));
        assert_eq!(expected.perception::<SIDE, S>(), perception::<S>(&probe[4]));
        for desired in &goals {
            if position_goal(start.block) == *desired {
                continue;
            }
            env.reset_at(start, *desired);
            let step = env.step(action);
            assert_eq!(env.state(), expected);
            assert_eq!(step.observation, perception::<S>(&probe[4]));
            assert_eq!(step.achieved, position_goal(expected.block));
            let result = step.outcome(env.objective(), desired);
            let reached = position_goal(expected.block) == *desired;
            assert_eq!(result.reward, if reached { 1000.0 } else { 0.0 });
            assert_eq!(result.terminated, reached);
            assert_eq!(result.truncated, !reached);
        }
    }
    let mut actual: Vec<_> = env
        .knowledge_transitions()
        .into_iter()
        .map(|t| (t.p0.symbols, t.action, t.p1.symbols))
        .collect();
    let mut expected: Vec<_> = entry["knowledge"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            (
                perception::<S>(&t[0]).symbols,
                t[1].as_u64().unwrap() as usize,
                perception::<S>(&t[2]).symbols,
            )
        })
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
    for (index, start) in entry["states"].as_array().unwrap().iter().enumerate() {
        for (goal_index, desired) in goals.iter().enumerate() {
            assert_eq!(
                env.distance_to_goal(state(start), desired),
                Some(entry["distances"][index][goal_index].as_u64().unwrap() as u32)
            );
        }
    }
}

#[test]
fn all_reachable_hand_eye_states_match_simulator_objective_knowledge_and_distances() {
    parity::<3, 10>();
    parity::<4, 17>();
    parity::<5, 26>();
}

#[test]
fn hand_eye_reset_distribution_draws_block_then_half_held_then_optional_gripper() {
    let mut env = HandEye::<3, 10>::new(10, rng(42));
    let mut reference = ChaChaRandomSource::from_seed_and_stream(42, 1);
    for _ in 0..100 {
        let block = (reference.gen_range(3), reference.gen_range(3));
        let held = reference.gen_bool(0.5);
        let gripper = if held {
            block
        } else {
            (reference.gen_range(3), reference.gen_range(3))
        };
        let excluded = block.1 * 3 + block.0;
        let index = reference.gen_range(8);
        let expected_goal = env.goal_pool()[index + usize::from(index >= excluded)];
        let start = env.reset();
        assert_eq!(
            env.state(),
            HandEyeState {
                gripper,
                block,
                held
            }
        );
        assert_eq!(start.desired, expected_goal);
        assert_ne!(start.achieved, start.desired);
    }
}

#[test]
fn held_block_reaches_the_goal_without_release_on_the_limit_step() {
    let mut env = HandEye::<3, 10>::new(2, rng(0));
    let desired = position_goal((2, 0));
    env.reset_at(
        HandEyeState {
            gripper: (0, 0),
            block: (0, 0),
            held: true,
        },
        desired,
    );
    assert!(!env.step(1).time_limit_reached);
    let step = env.step(1);
    assert!(env.state().held);
    assert_eq!(step.observation.symbols[2], Symbol::Token(b'b'));
    assert_eq!(step.observation.symbols[9], Symbol::Token(b'2'));
    let result = step.outcome(env.objective(), &desired);
    assert_eq!(result.reward, 1000.0);
    assert!(result.terminated);
    assert!(!result.truncated);
}

#[test]
fn hand_eye_seeded_goal_evaluation_and_adapter_flags_reproduce() {
    let mut first = GoalConditioned::<_, 10, 2, 12>::new(HandEye::<3, 10>::new(5, rng(42)));
    let mut second = GoalConditioned::<_, 10, 2, 12>::new(HandEye::<3, 10>::new(5, rng(42)));
    let goals = first.environment().goal_pool().to_vec();
    let mut actions = ChaChaRandomSource::from_seed_and_stream(42, 2);
    for desired in goals.into_iter().cycle().take(100) {
        let start = first.reset_with_goal(desired);
        assert_eq!(start, second.reset_with_goal(desired));
        assert_eq!(&start.symbols[10..], &desired.symbols);
        assert_ne!(position_goal(first.environment().state().block), desired);
        loop {
            let action = actions.gen_range(6);
            let a = first.step(action);
            let b = second.step(action);
            assert_eq!(a.observation, b.observation);
            assert_eq!(
                (a.reward, a.terminated, a.truncated),
                (b.reward, b.terminated, b.truncated)
            );
            let reached = position_goal(first.environment().state().block) == desired;
            assert_eq!(a.terminated, reached);
            assert_eq!(a.reward, if reached { 1000.0 } else { 0.0 });
            assert_eq!(&a.observation.symbols[10..], &desired.symbols);
            if a.terminated || a.truncated {
                break;
            }
        }
    }
}
