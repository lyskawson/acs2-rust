use std::cell::Cell;
use std::collections::BTreeMap;
use std::mem::size_of;

use acs2_core::goal::{
    ExactMatch, Goal, GoalEnvironment, GoalLayout, GoalObjective, GoalStart, GoalStep,
};
use acs2_core::perception::Perception;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::TruncationMode;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::{position_goal, HandEye4, HandEyeState};
use acs2_envs::goal::maze::CoordinateGoalMaze;
use acs2_envs::maze::geometries::pyalcs::{maze4, mazef3};
use acs2_trajectory::*;

const OBJECTIVE: ExactMatch = ExactMatch {
    reward_on_reach: 1000.0,
};
const STRATEGIES: [GoalStrategy; 5] = [
    GoalStrategy::Original,
    GoalStrategy::Final,
    GoalStrategy::Future,
    GoalStrategy::Episode,
    GoalStrategy::UniformReal,
];
const RULES: [Admissibility; 3] = [
    Admissibility::EveryTransition,
    Admissibility::FromNonGoalState,
    Admissibility::CounterfactualEpisode,
];

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}
fn observation(value: u8) -> Perception<1> {
    Perception::new([Symbol::Token(value)])
}
fn selection(strategy: GoalStrategy, rule: Admissibility, filter: bool) -> Selection {
    Selection {
        strategy,
        admissibility: rule,
        candidate_filter: filter,
    }
}
fn config(
    strategy: GoalStrategy,
    rule: Admissibility,
    filter: bool,
    share: f64,
) -> SamplerConfiguration {
    SamplerConfiguration {
        selection: selection(strategy, rule, filter),
        relabeled_proportion: share,
        truncation: TruncationMode::Bootstrap,
    }
}
fn episode(
    store: &mut TrajectoryStore<1, 1>,
    states: &[u8],
    desired: u8,
    limit: bool,
) -> EpisodeId {
    let mut raw = store.begin_episode(GoalStart {
        observation: observation(90),
        achieved: goal(states[0]),
        desired: goal(desired),
    });
    for (index, &state) in states[1..].iter().enumerate() {
        raw.push(
            index,
            GoalStep {
                observation: observation(91 + index as u8),
                achieved: goal(state),
                terminal_state: false,
                time_limit_reached: limit && index + 2 == states.len(),
            },
        );
    }
    store.insert(raw).unwrap()
}

fn independent<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    t: usize,
    selected: Selection,
    candidates: &[Goal<G>],
) -> (usize, BTreeMap<Goal<G>, (usize, bool, bool)>) {
    let source = match selected.strategy {
        GoalStrategy::Original => vec![e.start().desired],
        GoalStrategy::Final => vec![e.steps().last().unwrap().step.achieved],
        GoalStrategy::Future => e.steps()[t..]
            .iter()
            .map(|step| step.step.achieved)
            .collect(),
        GoalStrategy::Episode => e.steps().iter().map(|step| step.step.achieved).collect(),
        GoalStrategy::UniformReal => candidates.to_vec(),
    };
    let current = if t == 0 {
        e.start().achieved
    } else {
        e.steps()[t - 1].step.achieved
    };
    let prefix: Vec<Goal<G>> = [e.start().achieved]
        .into_iter()
        .chain(e.steps()[..t].iter().map(|step| step.step.achieved))
        .collect();
    let mut counts = BTreeMap::new();
    for candidate in &source {
        let reached = current == *candidate;
        let after = prefix.contains(candidate);
        let admits = match selected.admissibility {
            Admissibility::EveryTransition => true,
            Admissibility::FromNonGoalState => !reached,
            Admissibility::CounterfactualEpisode => !after,
        };
        if admits && (!selected.candidate_filter || candidates.contains(candidate)) {
            counts.entry(*candidate).or_insert((0, reached, after)).0 += 1;
        }
    }
    (source.len(), counts)
}

fn assert_enumeration<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    candidates: &[Goal<G>],
) {
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    for t in 0..e.len() {
        for strategy in STRATEGIES {
            for rule in RULES {
                for filter in [false, true] {
                    let selected = selection(strategy, rule, filter);
                    let (source, expected) = independent(e, t, selected, candidates);
                    let actual = goal_distribution(e, t, selected, candidates, &evaluator);
                    let total: usize = expected.values().map(|entry| entry.0).sum();
                    assert_eq!(actual.source_count, source);
                    assert_eq!(actual.admitted_count, total, "{t} {selected:?}");
                    assert_eq!(actual.choices.len(), expected.len());
                    for choice in actual.choices {
                        let &(weight, reached, after) =
                            expected.get(&choice.goal).expect("admissible goal");
                        assert_eq!(choice.weight, weight);
                        assert_eq!(
                            choice.probability.to_bits(),
                            (weight as f64 / total as f64).to_bits()
                        );
                        assert_eq!(
                            choice.facts,
                            GoalFacts {
                                already_reached: reached,
                                after_counterfactual_end: after
                            }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_four_bit_transition_and_goal_matches_independent_enumeration() {
    let mut env = BitFlipping::<4>::with_step_cap(4, Box::new(ChaChaRandomSource::from_seed(42)));
    let goals: Vec<_> = env.goal_pool().collect();
    let mut store = TrajectoryStore::new(2000);
    for desired in &goals {
        let start = Perception::new(core::array::from_fn(|i| {
            if desired.symbols[i] == Symbol::Token(b'0') {
                Symbol::Token(b'1')
            } else {
                Symbol::Token(b'0')
            }
        }));
        let mut raw = store.begin_episode(env.reset_at(start, *desired));
        raw.push(0, env.step(0));
        store.insert(raw).unwrap();
    }
    for start_goal in &goals {
        for action in 0..4 {
            let desired = Goal::new(core::array::from_fn(|i| {
                if start_goal.symbols[i] == Symbol::Token(b'0') {
                    Symbol::Token(b'1')
                } else {
                    Symbol::Token(b'0')
                }
            }));
            let mut raw =
                store.begin_episode(env.reset_at(Perception::new(start_goal.symbols), desired));
            for a in [action, (action + 1) % 4, action, (action + 2) % 4] {
                let step = env.step(a);
                raw.push(a, step);
                if step.outcome(env.objective(), &desired).terminated || step.time_limit_reached {
                    break;
                }
            }
            let id = store.insert(raw).unwrap();
            assert_enumeration(store.episode(id).unwrap(), &goals);
        }
    }
}

#[test]
fn a_stationary_handeye_block_exposes_trivial_goals_and_fallbacks() {
    let mut env = HandEye4::new(4, Box::new(ChaChaRandomSource::from_seed(42)));
    let start = env.reset_at(
        HandEyeState {
            gripper: (0, 0),
            block: (3, 3),
            held: false,
        },
        position_goal((1, 1)),
    );
    let mut store = TrajectoryStore::new(4);
    let mut raw = store.begin_episode(start);
    for _ in 0..4 {
        raw.push(5, env.step(5));
    }
    let id = store.insert(raw).unwrap();
    let e = store.episode(id).unwrap();
    assert_enumeration(e, store.candidates());
    for strategy in [
        GoalStrategy::Final,
        GoalStrategy::Future,
        GoalStrategy::Episode,
    ] {
        let every = episode_composition::<17, 2, 19>(
            e,
            store.candidates(),
            config(strategy, RULES[0], false, 1.0),
            &ObjectiveEvaluator(env.objective()),
        )
        .unwrap()
        .expected;
        assert_eq!(every.already_reached, 1.0);
        assert_eq!(every.after_counterfactual_end, 1.0);
        assert_eq!(every.done, 1.0);
        assert_eq!(every.outside_candidates, 1.0);
        assert_eq!(every.mean_reward, 1000.0);
        for rule in [RULES[1], RULES[2]] {
            let filtered = episode_composition::<17, 2, 19>(
                e,
                store.candidates(),
                config(strategy, rule, false, 1.0),
                &ObjectiveEvaluator(env.objective()),
            )
            .unwrap()
            .expected;
            assert_eq!(filtered.admissible_share, 0.0);
            assert_eq!(filtered.no_admissible_goal, 1.0);
            assert_eq!(filtered.fallback, 1.0);
            assert_eq!(filtered.original, 1.0);
            assert_eq!(filtered.mean_reward, 0.0);
        }
    }
}

#[test]
fn a_maze_return_distinguishes_non_goal_from_counterfactual_and_keeps_the_start() {
    let mut env = CoordinateGoalMaze::all_walkable(
        &maze4::MAZE4,
        3,
        Box::new(ChaChaRandomSource::from_seed(42)),
    )
    .unwrap();
    let start_cell = (2, 5);
    let forward = (0..8)
        .find(|&a| env.topology().next_cell(start_cell, a) != start_cell)
        .unwrap();
    let other = env.topology().next_cell(start_cell, forward);
    let backward = (0..8)
        .find(|&a| env.topology().next_cell(other, a) == start_cell)
        .unwrap();
    let desired = env.goal_at((6, 4));
    let mut store = TrajectoryStore::new(3);
    let mut raw = store.begin_episode(env.reset_at(start_cell, desired));
    for a in [forward, backward, forward] {
        raw.push(a, env.step(a));
    }
    let id = store.insert(raw).unwrap();
    let e = store.episode(id).unwrap();
    let candidates = [desired, env.goal_at(other), env.goal_at(start_cell)];
    assert_enumeration(e, &candidates);
    let evaluator = ObjectiveEvaluator(env.objective());
    let non_goal = goal_distribution(
        e,
        2,
        selection(GoalStrategy::Future, RULES[1], false),
        &candidates,
        &evaluator,
    );
    let counter = goal_distribution(
        e,
        2,
        selection(GoalStrategy::Future, RULES[2], false),
        &candidates,
        &evaluator,
    );
    assert_eq!(non_goal.choices.len(), 1);
    assert!(non_goal.choices[0].facts.after_counterfactual_end);
    assert!(counter.choices.is_empty());
    let start_goal = relabel_episode::<8, 2, 10>(
        e,
        &env.goal_at(start_cell),
        RULES[2],
        &evaluator,
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert!(start_goal.samples.is_empty());
    assert_eq!(start_goal.end, EpisodeEnd::AlreadyReachedAtStart);
    let shortened = relabel_episode::<8, 2, 10>(
        e,
        &env.goal_at(other),
        RULES[2],
        &evaluator,
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert_eq!(shortened.samples.len(), 1);
    assert_eq!(shortened.samples[0].transition, 0);
    assert_eq!(shortened.end, EpisodeEnd::Terminated);
}

struct AtLeast;
fn value(goal: &Goal<1>) -> u8 {
    match goal.symbols[0] {
        Symbol::Token(n) => n,
        Symbol::Wildcard => panic!("token"),
    }
}
impl GoalObjective<1> for AtLeast {
    fn reward(&self, achieved: &Goal<1>, desired: &Goal<1>) -> f64 {
        if self.is_reached(achieved, desired) {
            f64::from(value(achieved))
        } else {
            0.0
        }
    }
    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        value(achieved) >= value(desired)
    }
}

#[test]
fn rewards_termination_and_admissibility_follow_a_non_equality_objective() {
    let mut store = TrajectoryStore::new(3);
    let id = episode(&mut store, &[0, 5, 2, 4], 9, true);
    let e = store.episode(id).unwrap();
    let evaluator = ObjectiveEvaluator(&AtLeast);
    let reached =
        build_sample::<1, 1, 2>(e, 0, &goal(3), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert_eq!(reached.sample.reward.to_bits(), 5.0f64.to_bits());
    assert!(reached.sample.done && reached.outcome.terminated);
    let missed =
        build_sample::<1, 1, 2>(e, 1, &goal(3), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert_eq!(missed.sample.reward, 0.0);
    assert!(!missed.sample.done);
    let distribution = goal_distribution(
        e,
        1,
        selection(GoalStrategy::Future, RULES[1], false),
        store.candidates(),
        &evaluator,
    );
    assert!(distribution.choices.is_empty());
}

#[test]
fn maze_f3_perceptual_twin_neither_pays_nor_terminates_after_relabeling() {
    let mut env = CoordinateGoalMaze::all_walkable(
        &mazef3::MAZEF3,
        5,
        Box::new(ChaChaRandomSource::from_seed(42)),
    )
    .unwrap();
    let desired = env.goal_at((1, 4));
    let mut store = TrajectoryStore::new(5);
    let mut raw = store.begin_episode(env.reset_at((3, 2), desired));
    let action = (0..8)
        .find(|&a| env.topology().next_cell((3, 2), a) == (3, 3))
        .unwrap();
    raw.push(action, env.step(action));
    let id = store.insert(raw).unwrap();
    let e = store.episode(id).unwrap();
    let sample = build_sample::<8, 2, 10>(
        e,
        0,
        &desired,
        &ObjectiveEvaluator(env.objective()),
        TruncationMode::Bootstrap,
    )
    .unwrap();
    let twin_observation = env.reset_at((1, 4), env.goal_at((3, 2))).observation;
    assert_eq!(e.steps()[0].step.observation, twin_observation);
    assert_eq!(sample.sample.reward, 0.0);
    assert!(!sample.outcome.terminated && !sample.sample.done);
    let true_goal = e.steps()[0].step.achieved;
    assert!(
        build_sample::<8, 2, 10>(
            e,
            0,
            &true_goal,
            &ObjectiveEvaluator(env.objective()),
            TruncationMode::Bootstrap
        )
        .unwrap()
        .sample
        .done
    );
}

#[test]
fn terminal_cap_interior_and_cut_flags_are_recomputed_for_each_goal() {
    let mut store = TrajectoryStore::new(4);
    let cap = episode(&mut store, &[0, 2, 4], 9, true);
    let cut = episode(&mut store, &[0, 4], 4, false);
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    let e = store.episode(cap).unwrap();
    let reached =
        build_sample::<1, 1, 2>(e, 1, &goal(4), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert!(reached.outcome.terminated && reached.sample.done);
    assert!(!reached.outcome.truncated);
    let truncated =
        build_sample::<1, 1, 2>(e, 1, &goal(9), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert!(truncated.outcome.truncated);
    assert!(!truncated.outcome.terminated && !truncated.sample.done);
    assert!(
        build_sample::<1, 1, 2>(e, 1, &goal(9), &evaluator, TruncationMode::Pyalcs)
            .unwrap()
            .sample
            .done
    );
    let interior =
        build_sample::<1, 1, 2>(e, 0, &goal(9), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert!(!interior.outcome.terminated && !interior.outcome.truncated && !interior.sample.done);
    let cut_episode = relabel_episode::<1, 1, 2>(
        store.episode(cut).unwrap(),
        &goal(9),
        RULES[2],
        &evaluator,
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert_eq!(cut_episode.end, EpisodeEnd::Cut);
    assert_eq!(cut_episode.samples.len(), 1);
    assert!(
        !cut_episode.samples[0].scored.outcome.terminated
            && !cut_episode.samples[0].scored.outcome.truncated
    );
}

#[test]
fn environment_terminal_state_terminates_for_every_goal_without_reach_queries() {
    let mut store = TrajectoryStore::new(1);
    let mut raw = store.begin_episode(GoalStart {
        observation: observation(5),
        achieved: goal(0),
        desired: goal(9),
    });
    raw.push(
        0,
        GoalStep {
            observation: observation(5),
            achieved: goal(2),
            terminal_state: true,
            time_limit_reached: true,
        },
    );
    let id = store.insert(raw).unwrap();
    let scored = build_sample::<1, 1, 2>(
        store.episode(id).unwrap(),
        0,
        &goal(9),
        &ObjectiveEvaluator(&OBJECTIVE),
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert!(scored.sample.done && scored.outcome.terminated && !scored.outcome.truncated);
    assert_eq!(scored.cost.reach_evaluations, 0);
}

#[test]
fn both_perceptions_carry_one_goal_and_achieved_goals_are_not_observations() {
    let mut store = TrajectoryStore::new(3);
    let id = episode(&mut store, &[2, 4, 1, 3], 9, true);
    let e = store.episode(id).unwrap();
    for t in 0..3 {
        let sample = build_sample::<1, 1, 2>(
            e,
            t,
            &goal(7),
            &ObjectiveEvaluator(&OBJECTIVE),
            TruncationMode::Bootstrap,
        )
        .unwrap()
        .sample;
        assert_eq!(
            GoalLayout::<1, 1, 2>::split(&sample.state),
            (*e.observation(t), goal(7))
        );
        assert_eq!(
            GoalLayout::<1, 1, 2>::split(&sample.next_state),
            (*e.observation(t + 1), goal(7))
        );
    }
    let selected = goal_distribution(
        e,
        0,
        selection(GoalStrategy::Episode, RULES[0], false),
        store.candidates(),
        &ObjectiveEvaluator(&OBJECTIVE),
    );
    assert!(selected
        .choices
        .iter()
        .all(|choice| [goal(4), goal(1), goal(3)].contains(&choice.goal)));
    assert!(!selected.choices.iter().any(|choice| choice.goal == goal(2)));
}

#[test]
fn negative_nan_and_infinite_rewards_are_rejected_explicitly() {
    let mut store = TrajectoryStore::new(1);
    let id = episode(&mut store, &[0, 1], 1, false);
    for (reward, error) in [
        (-1.0, SampleError::NegativeReward),
        (f64::NAN, SampleError::NonFiniteReward),
        (f64::INFINITY, SampleError::NonFiniteReward),
    ] {
        let objective = ExactMatch {
            reward_on_reach: reward,
        };
        assert_eq!(
            build_sample::<1, 1, 2>(
                store.episode(id).unwrap(),
                0,
                &goal(1),
                &ObjectiveEvaluator(&objective),
                TruncationMode::Bootstrap
            ),
            Err(error)
        );
    }
}

#[test]
fn fifo_capacity_counts_steps_preserves_prefixes_and_keeps_candidate_history() {
    let mut store = TrajectoryStore::new(5);
    let first = episode(&mut store, &[0, 1, 2], 9, true);
    let second = episode(&mut store, &[0, 1, 2, 3], 8, true);
    assert_eq!(store.len(), 5);
    let third = episode(&mut store, &[0, 1], 7, true);
    assert!(store.episode(first).is_none());
    assert_eq!(store.len(), 4);
    assert_eq!(store.episode(second).unwrap().start().achieved, goal(0));
    assert_eq!(
        store.episodes().map(StoredEpisode::id).collect::<Vec<_>>(),
        vec![second, third]
    );
    assert_eq!(store.candidates(), &[goal(7), goal(8), goal(9)]);
    assert_eq!(store.transition(2), (store.episode(second).unwrap(), 2));
    assert_eq!(store.transition(3), (store.episode(third).unwrap(), 0));
    let bytes = size_of::<TrajectoryStore<1, 1>>()
        + 2 * size_of::<StoredEpisode<1, 1>>()
        + 4 * size_of::<StoredStep<1, 1>>()
        + 3 * size_of::<Goal<1>>();
    assert_eq!(store.logical_bytes(), bytes);
    let before = store.clone();
    let _ = goal_distribution(
        store.episode(second).unwrap(),
        2,
        selection(GoalStrategy::Future, RULES[0], false),
        store.candidates(),
        &ObjectiveEvaluator(&OBJECTIVE),
    );
    assert_eq!(store, before);
    for desired in 10..30 {
        let id = episode(&mut store, &[0, 1, 2, 3], desired, true);
        assert!(id > third);
        assert!(store.len() <= 5);
    }
}

#[test]
fn oversized_and_empty_episodes_fail_without_evicting_history() {
    let mut store = TrajectoryStore::new(1);
    let first = episode(&mut store, &[0, 1], 9, true);
    let start = GoalStart {
        observation: observation(0),
        achieved: goal(0),
        desired: goal(8),
    };
    assert_eq!(
        store.insert(Episode::new(start)),
        Err(StoreError::EmptyEpisode)
    );
    let mut long = store.begin_episode(start);
    for _ in 0..2 {
        long.push(
            0,
            GoalStep {
                observation: observation(0),
                achieved: goal(0),
                terminal_state: false,
                time_limit_reached: false,
            },
        );
    }
    assert_eq!(
        store.insert(long),
        Err(StoreError::EpisodeTooLong {
            steps: 2,
            capacity: 1
        })
    );
    assert!(store.episode(first).is_some());
    assert_eq!(store.candidates(), &[goal(8), goal(9)]);
}

#[test]
fn duplicate_achieved_states_keep_the_future_distribution_weighted_by_indices() {
    let mut store = TrajectoryStore::new(3);
    let id = episode(&mut store, &[0, 1, 1, 2], 9, true);
    let e = store.episode(id).unwrap();
    let distribution = goal_distribution(
        e,
        0,
        selection(GoalStrategy::Future, RULES[0], false),
        store.candidates(),
        &ObjectiveEvaluator(&OBJECTIVE),
    );
    assert_eq!(distribution.choices[0].weight, 2);
    assert_eq!(distribution.choices[0].probability, 2.0 / 3.0);
    let restricted = goal_distribution(
        e,
        0,
        selection(GoalStrategy::Future, RULES[0], true),
        &[goal(1)],
        &ObjectiveEvaluator(&OBJECTIVE),
    );
    assert_eq!(restricted.admissible_share(), 2.0 / 3.0);
    assert_eq!(restricted.choices[0].probability, 1.0);
    let mut rng = ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM);
    let count = (0..30_000)
        .filter(|_| distribution.draw(&mut rng).unwrap().goal == goal(1))
        .count();
    assert!((count as i64 - 20_000).abs() < 350);
}

#[test]
fn transition_draws_are_uniform_and_relabel_share_is_an_independent_coin() {
    let mut store = TrajectoryStore::new(4);
    let first = episode(&mut store, &[0, 1], 9, true);
    let second = episode(&mut store, &[0, 1, 2, 3], 8, true);
    let mut sampler = Sampler::new(
        config(GoalStrategy::Future, RULES[0], false, 0.8),
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    let samples = sampler
        .draw::<1, 1, 2>(&store, 40_000, &ObjectiveEvaluator(&OBJECTIVE))
        .unwrap();
    let mut frequencies = BTreeMap::new();
    for sample in samples {
        *frequencies
            .entry((sample.episode, sample.transition))
            .or_insert(0i64) += 1;
    }
    for pair in [(first, 0), (second, 0), (second, 1), (second, 2)] {
        assert!((frequencies[&pair] - 10_000).abs() < 400);
    }
    assert_eq!(sampler.counters().draws, 40_000);
    assert!((sampler.counters().relabeled as i64 - 32_000).abs() < 500);
    assert_eq!(
        sampler.counters().original + sampler.counters().relabeled,
        40_000
    );
}

#[test]
fn every_strategy_rule_and_filter_draw_is_admissible_or_a_counted_original_fallback() {
    let mut store = TrajectoryStore::new(5);
    episode(&mut store, &[1, 2, 1, 3, 4, 3], 9, true);
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    for strategy in STRATEGIES {
        for rule in RULES {
            for filter in [false, true] {
                let mut sampler = Sampler::new(
                    config(strategy, rule, filter, 1.0),
                    ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
                );
                let draws = sampler.draw::<1, 1, 2>(&store, 1000, &evaluator).unwrap();
                assert_eq!(draws.len(), 1000);
                for draw in &draws {
                    let e = store.episode(draw.episode).unwrap();
                    let (_, admissible) = independent(
                        e,
                        draw.transition,
                        selection(strategy, rule, filter),
                        store.candidates(),
                    );
                    if draw.provenance.origin == SampleOrigin::Relabeled {
                        assert!(admissible.contains_key(&draw.goal));
                    }
                    if draw.provenance.fallback {
                        assert!(admissible.is_empty());
                        assert_eq!(draw.goal, e.start().desired);
                    }
                    assert_eq!(
                        draw.provenance.already_reached,
                        e.achieved(draw.transition) == &draw.goal
                    );
                    assert_eq!(
                        draw.provenance.after_counterfactual_end,
                        (0..=draw.transition).any(|s| e.achieved(s) == &draw.goal)
                    );
                }
                assert_eq!(
                    sampler.counters().fallbacks,
                    draws.iter().filter(|draw| draw.provenance.fallback).count() as u64
                );
            }
        }
    }
}

#[test]
fn relabeled_provenance_depends_on_the_route_even_when_the_goals_are_equal() {
    let mut store = TrajectoryStore::new(1);
    episode(&mut store, &[0, 1], 1, false);
    let mut sampler = Sampler::new(
        config(GoalStrategy::Final, RULES[2], true, 1.0),
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    let samples = sampler
        .draw::<1, 1, 2>(&store, 10, &ObjectiveEvaluator(&OBJECTIVE))
        .unwrap();
    assert!(samples.iter().all(|s| s.goal == goal(1)
        && s.provenance.origin == SampleOrigin::Relabeled
        && s.provenance.done));
    assert_eq!(sampler.counters().relabeled, 10);
    assert_eq!(
        sampler.counters().strategies[GoalStrategy::Final.index()],
        10
    );
}

#[test]
fn zero_relabel_share_and_empty_store_have_explicit_behavior() {
    let mut store = TrajectoryStore::new(1);
    let mut sampler = Sampler::new(
        config(GoalStrategy::Future, RULES[2], true, 0.0),
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    assert_eq!(
        sampler.draw::<1, 1, 2>(&store, 1, &ObjectiveEvaluator(&OBJECTIVE)),
        Err(SampleError::EmptyStore)
    );
    assert!(sampler
        .draw::<1, 1, 2>(&store, 0, &ObjectiveEvaluator(&OBJECTIVE))
        .unwrap()
        .is_empty());
    episode(&mut store, &[0, 1], 9, true);
    let samples = sampler
        .draw::<1, 1, 2>(&store, 30, &ObjectiveEvaluator(&OBJECTIVE))
        .unwrap();
    assert!(samples
        .iter()
        .all(|s| s.provenance.origin == SampleOrigin::Original && !s.provenance.fallback));
    assert_eq!(sampler.counters().original, 30);
}

struct CountedObjective {
    rewards: Cell<u64>,
    reached: Cell<u64>,
}
impl GoalObjective<1> for CountedObjective {
    fn reward(&self, achieved: &Goal<1>, desired: &Goal<1>) -> f64 {
        self.rewards.set(self.rewards.get() + 1);
        if achieved == desired {
            1000.0
        } else {
            0.0
        }
    }
    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        self.reached.set(self.reached.get() + 1);
        achieved == desired
    }
}

#[test]
fn objective_cost_counts_actual_calls_and_determinism_includes_the_sampler_stream() {
    let mut store = TrajectoryStore::new(4);
    episode(&mut store, &[1, 2, 1, 3, 4], 9, true);
    let c = config(GoalStrategy::Future, RULES[1], false, 0.8);
    let mut first = Sampler::new(
        c,
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    let mut second = Sampler::new(
        c,
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    let objective = CountedObjective {
        rewards: Cell::new(0),
        reached: Cell::new(0),
    };
    let a = first
        .draw::<1, 1, 2>(&store, 200, &ObjectiveEvaluator(&objective))
        .unwrap();
    assert_eq!(
        first.counters().cost,
        ObjectiveCost {
            reward_evaluations: objective.rewards.get(),
            reach_evaluations: objective.reached.get()
        }
    );
    let b = second
        .draw::<1, 1, 2>(&store, 200, &ObjectiveEvaluator(&OBJECTIVE))
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(first.counters(), second.counters());
    assert_eq!(
        first.random_source().capture_state(),
        second.random_source().capture_state()
    );
    assert_eq!(first.random_source().capture_state().unwrap().stream, 7);
}

#[test]
fn exact_episode_composition_matches_a_separate_enumeration_including_cost_and_fallback() {
    let mut store = TrajectoryStore::new(4);
    let id = episode(&mut store, &[1, 2, 1, 3, 4], 9, true);
    let e = store.episode(id).unwrap();
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    for strategy in STRATEGIES {
        for rule in RULES {
            for filter in [false, true] {
                for share in [0.0, 0.3, 1.0] {
                    let configuration = config(strategy, rule, filter, share);
                    let actual = episode_composition::<1, 1, 2>(
                        e,
                        store.candidates(),
                        configuration,
                        &evaluator,
                    )
                    .unwrap()
                    .expected;
                    let mut done = 0.0;
                    let mut reward = 0.0;
                    let mut queries = 0.0;
                    let mut fallback = 0.0;
                    let attempt = if strategy == GoalStrategy::Original {
                        0.0
                    } else {
                        share
                    };
                    for t in 0..e.len() {
                        let distribution = goal_distribution(
                            e,
                            t,
                            configuration.selection,
                            store.candidates(),
                            &evaluator,
                        );
                        queries += attempt * distribution.cost.reach_evaluations as f64;
                        let mut original_weight = 1.0 - attempt;
                        if distribution.choices.is_empty() {
                            original_weight += attempt;
                            fallback += attempt;
                        }
                        let mut weighted: Vec<(Goal<1>, f64)> = distribution
                            .choices
                            .iter()
                            .map(|choice| (choice.goal, attempt * choice.probability))
                            .collect();
                        weighted.push((e.start().desired, original_weight));
                        let mut original_cost = ObjectiveCost::default();
                        acs2_trajectory::selection::goal_facts(
                            e,
                            t,
                            &e.start().desired,
                            &evaluator,
                            &mut original_cost,
                        );
                        queries += original_weight * original_cost.reach_evaluations as f64;
                        for (goal, probability) in weighted {
                            let scored = build_sample::<1, 1, 2>(
                                e,
                                t,
                                &goal,
                                &evaluator,
                                TruncationMode::Bootstrap,
                            )
                            .unwrap();
                            done += probability * f64::from(scored.sample.done);
                            reward += probability * scored.sample.reward;
                            queries += probability * scored.cost.total() as f64;
                        }
                    }
                    assert!((actual.done - done / 4.0).abs() < 1e-12);
                    assert!((actual.mean_reward - reward / 4.0).abs() < 1e-12);
                    assert!((actual.fallback - fallback / 4.0).abs() < 1e-12);
                    assert!(
                        (actual.mean_objective_evaluations() - queries / 4.0).abs() < 1e-12,
                        "{configuration:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn composition_analysis_cost_reports_its_actual_precomputation_calls() {
    let mut store = TrajectoryStore::new(4);
    let id = episode(&mut store, &[1, 2, 1, 3, 4], 9, true);
    let objective = CountedObjective {
        rewards: Cell::new(0),
        reached: Cell::new(0),
    };
    let result = episode_composition::<1, 1, 2>(
        store.episode(id).unwrap(),
        store.candidates(),
        config(GoalStrategy::Future, RULES[1], true, 0.8),
        &ObjectiveEvaluator(&objective),
    )
    .unwrap();
    assert_eq!(
        result.analysis_cost,
        ObjectiveCost {
            reward_evaluations: objective.rewards.get(),
            reach_evaluations: objective.reached.get()
        }
    );
}

#[test]
fn invalid_proportions_are_rejected_before_drawing() {
    for share in [-0.1, 1.1, f64::NAN] {
        assert!(std::panic::catch_unwind(|| Sampler::new(
            config(GoalStrategy::Future, RULES[0], false, share),
            ChaChaRandomSource::from_seed(42)
        ))
        .is_err());
    }
}

#[test]
fn a_rejected_sample_counts_its_objective_cost_without_claiming_a_successful_draw() {
    let mut store = TrajectoryStore::new(1);
    episode(&mut store, &[0, 1], 1, false);
    let mut sampler = Sampler::new(
        config(GoalStrategy::Final, RULES[0], false, 1.0),
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    assert_eq!(
        sampler.draw::<1, 1, 2>(
            &store,
            1,
            &ObjectiveEvaluator(&ExactMatch {
                reward_on_reach: -1.0
            })
        ),
        Err(SampleError::NegativeReward)
    );
    assert_eq!(sampler.counters().draws, 0);
    assert_eq!(sampler.counters().failed_draws, 1);
    assert_eq!(sampler.counters().cost.reward_evaluations, 1);
    assert_eq!(sampler.counters().cost.reach_evaluations, 2);
}
