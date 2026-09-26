use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::config::Configuration;
use acs2_core::environment::Environment;
use acs2_core::goal::{Goal, GoalConditioned, GoalEnvironment, GoalLayout};
use acs2_core::perception::Perception;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;
use acs2_envs::goal::bit_flipping::BitFlipping;

fn rng(seed: u64) -> Box<dyn RandomSource> {
    Box::new(ChaChaRandomSource::from_seed_and_stream(seed, 1))
}

#[test]
fn every_flip_and_goal_has_exact_reward_termination_and_hamming_distance() {
    let mut env = BitFlipping::<4>::with_step_cap(1, rng(42));
    let goals: Vec<_> = env.goal_pool().collect();
    assert_eq!(goals.len(), 16);
    for start in &goals {
        for desired in &goals {
            let state = Perception::new(start.symbols);
            let expected_distance = start
                .symbols
                .iter()
                .zip(desired.symbols)
                .filter(|(a, b)| **a != *b)
                .count() as u32;
            assert_eq!(
                env.distance_to_goal(&state, desired),
                Some(expected_distance)
            );
            if start == desired {
                continue;
            }
            for action in 0..4 {
                env.reset_at(state, *desired);
                let step = env.step(action);
                for position in 0..4 {
                    assert_eq!(
                        step.observation.symbols[position] != start.symbols[position],
                        position == action
                    );
                }
                assert_eq!(step.achieved.symbols, step.observation.symbols);
                let outcome = step.outcome(env.objective(), desired);
                assert_eq!(
                    outcome.reward,
                    if step.achieved == *desired {
                        1000.0
                    } else {
                        0.0
                    }
                );
                assert_eq!(outcome.terminated, step.achieved == *desired);
                assert_eq!(outcome.truncated, step.achieved != *desired);
            }
        }
    }
    assert_eq!(env.knowledge_transitions().count(), 64);
}

#[test]
fn seeded_binary_starts_goals_and_steps_reproduce_through_the_adapter() {
    let mut first = GoalConditioned::<_, 4, 4, 8>::new(BitFlipping::<4>::new(rng(42)));
    let mut second = GoalConditioned::<_, 4, 4, 8>::new(BitFlipping::<4>::new(rng(42)));
    let mut actions = ChaChaRandomSource::from_seed_and_stream(42, 2);
    for _ in 0..100 {
        let a = first.reset();
        assert_eq!(a, second.reset());
        let (state, goal) = GoalLayout::<4, 4, 8>::split(&a);
        assert_ne!(state.symbols, goal.symbols);
        assert_eq!(goal, first.desired().unwrap());
        loop {
            let action = actions.gen_range(4);
            let a = first.step(action);
            let b = second.step(action);
            assert_eq!(a.observation, b.observation);
            assert_eq!(
                (a.reward, a.terminated, a.truncated),
                (b.reward, b.terminated, b.truncated)
            );
            assert_eq!(&a.observation.symbols[4..], &goal.symbols);
            if a.terminated || a.truncated {
                break;
            }
        }
    }
    for goal in first.environment().goal_pool() {
        let start = first.reset_with_goal(goal);
        assert_eq!(&start.symbols[4..], &goal.symbols);
    }
}

fn anticipation_without_context_failures_never_specializes_goals<const N: usize, const M: usize>() {
    for do_ga in [false, true] {
        for seed in 42..45 {
            let config = Configuration {
                number_of_possible_actions: N,
                do_ga,
                ..Configuration::default_protocol()
            };
            let mut agent =
                Agent::<M, _>::new(config, ChaChaRandomSource::from_seed_and_stream(seed, 0));
            let mut env = GoalConditioned::<_, N, N, M>::new(BitFlipping::<N>::new(rng(seed)));
            let selector = EpsilonGreedy {
                number_of_possible_actions: N,
                epsilon: 0.8,
            };
            let mut time = 0;
            for _ in 0..300 {
                let result =
                    agent.run_explore_trial(&mut env, &selector, &MaxFitnessBootstrap, time);
                time += u64::from(result.steps);
                for classifier in agent.population().iter() {
                    for position in GoalLayout::<N, N, M>::goal_positions() {
                        assert!(
                            classifier.condition.get(position).is_wildcard(),
                            "goal specialization at n={N}, seed={seed}, GA={do_ga}"
                        );
                        assert!(classifier.effect.get(position).is_wildcard());
                    }
                    assert!(!classifier.mark.is_marked());
                }
            }
            assert_eq!(agent.population().len(), 2 * N);
        }
    }
}

#[test]
fn no_failed_anticipation_means_acs2_cannot_specialize_on_bit_flipping_goals() {
    anticipation_without_context_failures_never_specializes_goals::<4, 8>();
    anticipation_without_context_failures_never_specializes_goals::<6, 12>();
    anticipation_without_context_failures_never_specializes_goals::<8, 16>();
}

#[test]
#[should_panic(expected = "desired goal must be binary")]
fn a_wildcard_goal_is_not_a_real_bit_flipping_goal() {
    BitFlipping::<4>::new(rng(0)).reset_with_goal(Goal::new([Symbol::Wildcard; 4]));
}

#[test]
#[should_panic(expected = "step cap must be positive")]
fn bit_flipping_refuses_a_zero_cap() {
    BitFlipping::<4>::with_step_cap(0, rng(0));
}

#[test]
fn a_configurable_cap_truncates_after_exactly_that_many_flips() {
    let mut env = BitFlipping::<4>::with_step_cap(3, rng(0));
    let desired = Goal::new([Symbol::Token(b'1'); 4]);
    env.reset_at(Perception::new([Symbol::Token(b'0'); 4]), desired);
    for index in 1..=3 {
        let step = env.step(0);
        let result = step.outcome(env.objective(), &desired);
        assert!(!result.terminated);
        assert_eq!(result.truncated, index == 3);
    }
}
