use std::collections::BTreeSet;

use acs2_core::environment::Environment;
use acs2_core::goal::{GoalConditioned, GoalEnvironment};
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::taxi::{passenger_goal, Taxi, TaxiState, STANDS};
use serde_json::Value;

fn rng(seed: u64) -> Box<dyn RandomSource> {
    Box::new(ChaChaRandomSource::from_seed_and_stream(seed, 1))
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../fixtures/taxi.json")).unwrap()
}

fn state(value: &Value) -> TaxiState {
    TaxiState {
        row: value[0].as_u64().unwrap() as usize,
        col: value[1].as_u64().unwrap() as usize,
        passenger: value[2].as_u64().unwrap() as usize,
    }
}

#[test]
fn every_taxi_transition_matches_gym_p_and_the_sparse_objective() {
    let data = fixture();
    let mut env = Taxi::new(1, rng(0));
    assert_eq!(data["probes"].as_array().unwrap().len(), 3000);
    let mut expected_knowledge = BTreeSet::new();
    for probe in data["probes"].as_array().unwrap() {
        let start = state(&probe[0]);
        let action = probe[1].as_u64().unwrap() as usize;
        let expected = state(&probe[2]);
        let desired = passenger_goal(probe[0][3].as_u64().unwrap() as usize);
        assert_eq!(start.after_action(action), expected);
        if start != expected {
            expected_knowledge.insert((
                start.perception().symbols,
                action,
                expected.perception().symbols,
            ));
        }
        if start.achieved() == desired {
            continue;
        }
        env.reset_at(start, desired);
        let step = env.step(action);
        assert_eq!(step.observation, expected.perception());
        assert_eq!(step.achieved, expected.achieved());
        let result = step.outcome(env.objective(), &desired);
        assert_eq!(result.terminated, probe[4].as_bool().unwrap());
        assert_eq!(result.truncated, !result.terminated);
        assert_eq!(result.reward, if result.terminated { 1000.0 } else { 0.0 });
    }
    let actual = env.knowledge_transitions();
    let keys: BTreeSet<_> = actual
        .iter()
        .map(|t| (t.p0.symbols, t.action, t.p1.symbols))
        .collect();
    assert_eq!(keys.len(), actual.len());
    assert_eq!(keys, expected_knowledge);
    assert_eq!(env.states().count(), 125);
    for (index, start) in data["states"].as_array().unwrap().iter().enumerate() {
        for goal in 0..4 {
            assert_eq!(
                env.distance_to_goal(state(start), &passenger_goal(goal)),
                Some(data["distances"][index][goal].as_u64().unwrap() as u32)
            );
        }
    }
}

#[test]
fn taxi_initial_distribution_is_uniform_over_gym_admissible_states() {
    let data = fixture();
    let initial: BTreeSet<_> = data["initial_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s[0].as_u64().unwrap(),
                s[1].as_u64().unwrap(),
                s[2].as_u64().unwrap(),
                s[3].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(initial.len(), 300);
    let mut env = Taxi::new(50, rng(42));
    let mut reference = ChaChaRandomSource::from_seed_and_stream(42, 1);
    for _ in 0..100 {
        let goal = reference.gen_range(4);
        let chosen = reference.gen_range(3);
        let row = reference.gen_range(5);
        let col = reference.gen_range(5);
        let passenger = chosen + usize::from(chosen >= goal);
        let start = env.reset();
        assert_eq!(
            env.state(),
            TaxiState {
                row,
                col,
                passenger
            }
        );
        assert_eq!(start.desired, passenger_goal(goal));
        assert!(initial.contains(&(row as u64, col as u64, passenger as u64, goal as u64)));
        assert_ne!(start.achieved, start.desired);
    }
}

#[test]
fn dropoff_at_another_stand_keeps_the_episode_running_and_limit_success_wins() {
    let mut ongoing = Taxi::new(3, rng(0));
    let target = passenger_goal(1);
    ongoing.reset_at(
        TaxiState {
            row: 0,
            col: 0,
            passenger: 4,
        },
        target,
    );
    let deposited = ongoing.step(5).outcome(ongoing.objective(), &target);
    assert_eq!(ongoing.state().passenger, 0);
    assert_eq!(deposited.reward, 0.0);
    assert!(!deposited.terminated);
    assert!(!deposited.truncated);
    let mut env = Taxi::new(1, rng(0));
    for (stand, &(row, col)) in STANDS.iter().enumerate() {
        for goal in 0..4 {
            let desired = passenger_goal(goal);
            env.reset_at(
                TaxiState {
                    row,
                    col,
                    passenger: 4,
                },
                desired,
            );
            let step = env.step(5);
            assert_eq!(env.state().passenger, stand);
            let result = step.outcome(env.objective(), &desired);
            assert_eq!(result.terminated, stand == goal);
            assert_eq!(result.truncated, stand != goal);
        }
    }
    let desired = passenger_goal(1);
    env.reset_at(
        TaxiState {
            row: 0,
            col: 4,
            passenger: 4,
        },
        desired,
    );
    let step = env.step(0);
    assert!(!step.outcome(env.objective(), &desired).terminated);
}

#[test]
fn taxi_seeded_goal_evaluation_and_adapter_offsets_reproduce() {
    let mut first = GoalConditioned::<_, 3, 1, 4>::new(Taxi::new(7, rng(42)));
    let mut second = GoalConditioned::<_, 3, 1, 4>::new(Taxi::new(7, rng(42)));
    let mut actions = ChaChaRandomSource::from_seed_and_stream(42, 2);
    for goal in (0..4).cycle().take(100) {
        let desired = passenger_goal(goal);
        let start = first.reset_with_goal(desired);
        assert_eq!(start, second.reset_with_goal(desired));
        assert_eq!(start.symbols[3], desired.symbols[0]);
        let mut steps = 0;
        loop {
            let action = actions.gen_range(6);
            let a = first.step(action);
            let b = second.step(action);
            steps += 1;
            assert_eq!(a.observation, b.observation);
            assert_eq!(
                (a.reward, a.terminated, a.truncated),
                (b.reward, b.terminated, b.truncated)
            );
            assert_eq!(a.observation.symbols[3], desired.symbols[0]);
            assert_eq!(
                a.terminated,
                first.environment().state().achieved() == desired
            );
            assert_eq!(a.truncated, !a.terminated && steps == 7);
            if a.terminated || a.truncated {
                break;
            }
        }
    }
}
