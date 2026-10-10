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

fn states<const S: usize, const G: usize>(e: &StoredEpisode<S, G>) -> Vec<Goal<G>> {
    [e.start().achieved]
        .into_iter()
        .chain(e.steps().iter().map(|step| step.step.achieved))
        .collect()
}

fn source_multiset<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    t: usize,
    strategy: GoalStrategy,
    candidates: &[Goal<G>],
) -> Vec<Goal<G>> {
    let states = states(e);
    match strategy {
        GoalStrategy::Original => vec![e.start().desired],
        GoalStrategy::Final => vec![states[e.steps().len()]],
        GoalStrategy::Future => states[t + 1..].to_vec(),
        GoalStrategy::Episode => states[1..].to_vec(),
        GoalStrategy::UniformReal => candidates.to_vec(),
    }
}

fn considered_goals<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    t: usize,
    selected: Selection,
    candidates: &[Goal<G>],
) -> Vec<Goal<G>> {
    let mut goals = source_multiset(e, t, selected.strategy, candidates);
    goals.sort();
    goals.dedup();
    goals.retain(|goal| !selected.candidate_filter || candidates.contains(goal));
    goals
}

fn queries_of<const G: usize>(states: &[Goal<G>], t: usize, goal: &Goal<G>) -> (u64, u64) {
    let earlier = if states[t] == *goal {
        0
    } else {
        states[..t]
            .iter()
            .position(|state| state == goal)
            .map_or(t, |index| index + 1)
    };
    (1, earlier as u64)
}

fn rule_reads<const G: usize>(
    rule: Admissibility,
    states: &[Goal<G>],
    t: usize,
    goal: &Goal<G>,
) -> (u64, u64) {
    let (current, earlier) = queries_of(states, t, goal);
    match rule {
        Admissibility::EveryTransition => (0, current + earlier),
        Admissibility::FromNonGoalState => (current, earlier),
        Admissibility::CounterfactualEpisode => (current + earlier, 0),
    }
}

fn independent<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    t: usize,
    selected: Selection,
    candidates: &[Goal<G>],
) -> (usize, BTreeMap<Goal<G>, (usize, bool, bool)>) {
    let source = source_multiset(e, t, selected.strategy, candidates);
    let states = states(e);
    let current = states[t];
    let prefix = &states[..t];
    let mut counts = BTreeMap::new();
    for candidate in &source {
        let reached = current == *candidate;
        let after = reached || prefix.contains(candidate);
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

fn assert_relabeled_episodes<const S: usize, const G: usize, const M: usize>(
    e: &StoredEpisode<S, G>,
    goals: &[Goal<G>],
) {
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    let states = states(e);
    for goal in goals {
        for rule in RULES {
            let relabeled =
                relabel_episode::<S, G, M>(e, goal, rule, &evaluator, TruncationMode::Bootstrap)
                    .unwrap();
            let admitted: Vec<usize> = (0..e.len())
                .filter(|&t| match rule {
                    Admissibility::EveryTransition => true,
                    Admissibility::FromNonGoalState => states[t] != *goal,
                    Admissibility::CounterfactualEpisode => !states[..=t].contains(goal),
                })
                .collect();
            assert_eq!(
                relabeled
                    .samples
                    .iter()
                    .map(|sample| sample.transition)
                    .collect::<Vec<_>>(),
                admitted
            );
            let expected_end = match relabeled.samples.last() {
                None if rule == Admissibility::CounterfactualEpisode => {
                    assert_eq!(states[0], *goal);
                    EpisodeEnd::AlreadyReachedAtStart
                }
                None => EpisodeEnd::NoAdmissibleTransition,
                Some(last) if last.scored.outcome.terminated => EpisodeEnd::Terminated,
                Some(last) if last.scored.outcome.truncated => EpisodeEnd::Truncated,
                Some(_) => EpisodeEnd::Cut,
            };
            assert_eq!(relabeled.end, expected_end, "{goal:?} {rule:?}");
            let mut selection = 0;
            let mut provenance = 0;
            for t in 0..e.len() {
                let (read, other) = rule_reads(rule, &states, t, goal);
                selection += read;
                provenance += other;
                if rule == Admissibility::CounterfactualEpisode
                    && admitted.contains(&t)
                    && states[t + 1] == *goal
                {
                    break;
                }
            }
            let scored = relabeled.samples.len() as u64;
            assert_eq!(
                relabeled.cost,
                CostByPurpose {
                    scoring: ObjectiveCost {
                        reward_evaluations: scored,
                        reach_evaluations: scored
                    },
                    selection: ObjectiveCost::reach(selection),
                    provenance: ObjectiveCost::reach(provenance),
                }
            );
        }
    }
}

fn assert_enumeration<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    candidates: &[Goal<G>],
) {
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    let states = states(e);
    for t in 0..e.len() {
        for strategy in STRATEGIES {
            for rule in RULES {
                for filter in [false, true] {
                    let selected = selection(strategy, rule, filter);
                    let (source, expected) = independent(e, t, selected, candidates);
                    let actual = goal_distribution(e, t, selected, candidates, &evaluator);
                    let total: usize = expected.values().map(|entry| entry.0).sum();
                    let mut reads = (0, 0);
                    for goal in considered_goals(e, t, selected, candidates) {
                        let (read, other) = rule_reads(rule, &states, t, &goal);
                        reads.0 += read;
                        reads.1 += other;
                    }
                    assert_eq!(
                        actual.cost,
                        CostByPurpose {
                            scoring: ObjectiveCost::default(),
                            selection: ObjectiveCost::reach(reads.0),
                            provenance: ObjectiveCost::reach(reads.1),
                        }
                    );
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
            assert_relabeled_episodes::<4, 4, 8>(store.episode(id).unwrap(), &goals);
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
        assert_eq!(every.original, RouteComposition::default());
        assert_eq!(
            every.relabeled,
            RouteComposition {
                share: 1.0,
                already_reached: 1.0,
                after_counterfactual_end: 1.0,
                done: 1.0,
                outside_candidates: 1.0,
                reward: 1000.0,
            }
        );
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
            assert_eq!(filtered.original.share, 1.0);
            assert_eq!(filtered.mean_reward, 0.0);
            assert_eq!(filtered.relabeled, RouteComposition::default());
            assert_eq!(filtered.relabeled.given_route(), None);
            assert_eq!(filtered.original.given_route(), Some(RouteMeans::default()));
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
    assert_relabeled_episodes::<8, 2, 10>(e, &candidates);
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
    let reached_on_cap = episode(&mut store, &[0, 4], 4, true);
    let e = store.episode(reached_on_cap).unwrap();
    let original =
        build_sample::<1, 1, 2>(e, 0, &goal(4), &evaluator, TruncationMode::Bootstrap).unwrap();
    assert!(original.outcome.terminated && !original.outcome.truncated);
    let another =
        relabel_episode::<1, 1, 2>(e, &goal(9), RULES[2], &evaluator, TruncationMode::Bootstrap)
            .unwrap();
    assert_eq!(another.end, EpisodeEnd::Truncated);
    assert!(another.samples[0].scored.outcome.truncated);
    assert!(!another.samples[0].scored.outcome.terminated);
    assert!(!another.samples[0].scored.sample.done);
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
    for sample in &samples {
        *frequencies
            .entry((sample.episode, sample.transition))
            .or_insert(0i64) += 1;
    }
    for pair in [(first, 0), (second, 0), (second, 1), (second, 2)] {
        assert!((frequencies[&pair] - 10_000).abs() < 400);
    }
    assert_eq!(sampler.counters().pooled().draws, 40_000);
    assert!((sampler.counters().relabeled.draws as i64 - 32_000).abs() < 500);
    assert_eq!(
        sampler.counters().original.draws + sampler.counters().relabeled.draws,
        40_000
    );
    assert_eq!(sampler.counters(), &recount(&samples, 0));
}

fn recount<const G: usize, const M: usize>(
    draws: &[DrawnSample<G, M>],
    failed_draws: u64,
) -> ReplayCounters {
    let mut counters = ReplayCounters {
        failed_draws,
        ..ReplayCounters::default()
    };
    for draw in draws {
        let p = draw.provenance;
        let route = if p.origin == SampleOrigin::Original {
            &mut counters.original
        } else {
            &mut counters.relabeled
        };
        route.draws += 1;
        route.already_reached += u64::from(p.already_reached);
        route.after_counterfactual_end += u64::from(p.after_counterfactual_end);
        route.done += u64::from(p.done);
        route.outside_candidates += u64::from(p.outside_candidates);
        route.reward += draw.scored.sample.reward;
        counters.fallbacks += u64::from(p.fallback);
        counters.strategies[p.strategy.index()] += 1;
        for (total, part) in [
            (&mut counters.cost.scoring, draw.cost.scoring),
            (&mut counters.cost.selection, draw.cost.selection),
            (&mut counters.cost.provenance, draw.cost.provenance),
        ] {
            total.reward_evaluations += part.reward_evaluations;
            total.reach_evaluations += part.reach_evaluations;
        }
    }
    counters
}

#[test]
fn every_strategy_rule_and_filter_draw_is_admissible_or_a_counted_original_fallback() {
    let mut store = TrajectoryStore::new(8);
    episode(&mut store, &[1, 2, 1, 3, 4, 3], 9, true);
    episode(&mut store, &[3, 4], 4, false);
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    for strategy in STRATEGIES {
        for rule in RULES {
            for filter in [false, true] {
                for share in [0.0, 0.6, 1.0] {
                    let selected = selection(strategy, rule, filter);
                    let mut sampler = Sampler::new(
                        config(strategy, rule, filter, share),
                        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
                    );
                    let draws = sampler.draw::<1, 1, 2>(&store, 1000, &evaluator).unwrap();
                    assert_eq!(draws.len(), 1000);
                    for draw in &draws {
                        let e = store.episode(draw.episode).unwrap();
                        let states = states(e);
                        let t = draw.transition;
                        let (_, admissible) = independent(e, t, selected, store.candidates());
                        let p = draw.provenance;
                        let attempted = p.origin == SampleOrigin::Relabeled || p.fallback;
                        if strategy == GoalStrategy::Original || share == 0.0 {
                            assert!(!attempted);
                        } else if share == 1.0 {
                            assert!(attempted);
                        }
                        assert_eq!(p.requested_strategy, strategy);
                        if p.origin == SampleOrigin::Relabeled {
                            assert!(admissible.contains_key(&draw.goal));
                            assert_eq!(p.strategy, strategy);
                        } else {
                            assert_eq!(draw.goal, e.start().desired);
                            assert_eq!(p.strategy, GoalStrategy::Original);
                        }
                        if p.fallback {
                            assert!(admissible.is_empty());
                        }
                        assert_eq!(p.already_reached, states[t] == draw.goal);
                        assert_eq!(
                            p.after_counterfactual_end,
                            states[..=t].contains(&draw.goal)
                        );
                        assert_eq!(p.done, states[t + 1] == draw.goal);
                        assert_eq!(
                            p.outside_candidates,
                            !store.candidates().contains(&draw.goal)
                        );
                        assert_eq!(draw.scored.sample.reward, if p.done { 1000.0 } else { 0.0 });
                        let mut reads = (0, 0);
                        if attempted {
                            for goal in considered_goals(e, t, selected, store.candidates()) {
                                let (read, other) = rule_reads(rule, &states, t, &goal);
                                reads.0 += read;
                                reads.1 += other;
                            }
                        }
                        if p.origin == SampleOrigin::Original {
                            let (current, earlier) = queries_of(&states, t, &draw.goal);
                            reads.1 += current + earlier;
                        }
                        assert_eq!(
                            draw.cost,
                            CostByPurpose {
                                scoring: ObjectiveCost {
                                    reward_evaluations: 1,
                                    reach_evaluations: 1
                                },
                                selection: ObjectiveCost::reach(reads.0),
                                provenance: ObjectiveCost::reach(reads.1),
                            },
                            "{selected:?} {share} {draw:?}"
                        );
                    }
                    assert_eq!(sampler.counters(), &recount(&draws, 0));
                    let pooled = sampler.counters().pooled();
                    assert_eq!(pooled.draws, 1000);
                    assert_eq!(
                        pooled.already_reached,
                        draws
                            .iter()
                            .filter(|d| d.provenance.already_reached)
                            .count() as u64
                    );
                    assert_eq!(
                        pooled.after_counterfactual_end,
                        draws
                            .iter()
                            .filter(|d| d.provenance.after_counterfactual_end)
                            .count() as u64
                    );
                    assert_eq!(
                        pooled.done,
                        draws.iter().filter(|d| d.provenance.done).count() as u64
                    );
                    assert_eq!(
                        pooled.outside_candidates,
                        draws
                            .iter()
                            .filter(|d| d.provenance.outside_candidates)
                            .count() as u64
                    );
                    for (route, origin) in [
                        (sampler.counters().original, SampleOrigin::Original),
                        (sampler.counters().relabeled, SampleOrigin::Relabeled),
                    ] {
                        let members: Vec<_> = draws
                            .iter()
                            .filter(|d| d.provenance.origin == origin)
                            .collect();
                        let count = members.len() as f64;
                        assert_eq!(
                            route.given_route(),
                            (!members.is_empty()).then(|| RouteMeans {
                                already_reached: members
                                    .iter()
                                    .filter(|d| d.provenance.already_reached)
                                    .count()
                                    as f64
                                    / count,
                                after_counterfactual_end: members
                                    .iter()
                                    .filter(|d| d.provenance.after_counterfactual_end)
                                    .count()
                                    as f64
                                    / count,
                                done: members.iter().filter(|d| d.provenance.done).count() as f64
                                    / count,
                                outside_candidates: members
                                    .iter()
                                    .filter(|d| d.provenance.outside_candidates)
                                    .count()
                                    as f64
                                    / count,
                                reward: members.iter().map(|d| d.scored.sample.reward).sum::<f64>()
                                    / count,
                            })
                        );
                    }
                }
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
    assert_eq!(sampler.counters().relabeled.draws, 10);
    assert_eq!(
        sampler.counters().strategies[GoalStrategy::Final.index()],
        10
    );
    assert_eq!(sampler.counters(), &recount(&samples, 0));
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
    assert_eq!(sampler.counters().original.draws, 30);
    assert_eq!(sampler.counters().relabeled, RouteCounters::default());
    assert_eq!(sampler.counters(), &recount(&samples, 0));
    assert_eq!(sampler.counters().cost.selection, ObjectiveCost::default());
    assert_eq!(
        sampler.counters().cost.provenance,
        ObjectiveCost::reach(samples.iter().map(|s| 1 + s.transition as u64).sum::<u64>())
    );
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
        first.counters().cost.total(),
        ObjectiveCost {
            reward_evaluations: objective.rewards.get(),
            reach_evaluations: objective.reached.get()
        }
    );
    assert_eq!(first.counters().cost.scoring.reward_evaluations, 200);
    assert_eq!(first.counters().cost.scoring.reach_evaluations, 200);
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

fn modeled_composition<const S: usize, const G: usize>(
    e: &StoredEpisode<S, G>,
    candidates: &[Goal<G>],
    configuration: SamplerConfiguration,
) -> ExpectedComposition {
    let selected = configuration.selection;
    let states = states(e);
    let desired = e.start().desired;
    let attempt = if selected.strategy == GoalStrategy::Original {
        0.0
    } else {
        configuration.relabeled_proportion
    };
    let weight = 1.0 / e.steps().len() as f64;
    let mut model = ExpectedComposition::default();
    for t in 0..e.steps().len() {
        let source = source_multiset(e, t, selected.strategy, candidates);
        let mut reads = (0, 0);
        let mut admitted = Vec::new();
        for goal in considered_goals(e, t, selected, candidates) {
            let (read, other) = rule_reads(selected.admissibility, &states, t, &goal);
            reads.0 += read;
            reads.1 += other;
            let admits = match selected.admissibility {
                Admissibility::EveryTransition => true,
                Admissibility::FromNonGoalState => states[t] != goal,
                Admissibility::CounterfactualEpisode => !states[..=t].contains(&goal),
            };
            if admits {
                admitted.push((goal, source.iter().filter(|&&g| g == goal).count()));
            }
        }
        let admitted_count: usize = admitted.iter().map(|&(_, count)| count).sum();
        let fallback = if admitted_count == 0 { attempt } else { 0.0 };
        let original_share = 1.0 - attempt + fallback;
        let (current, earlier) = queries_of(&states, t, &desired);
        model.admissible_share += weight * admitted_count as f64 / source.len() as f64;
        model.no_admissible_goal += weight * f64::from(admitted_count == 0);
        model.fallback += weight * fallback;
        model.original.share += weight * original_share;
        model.relabeled.share += weight * (attempt - fallback);
        model.mean_selection_evaluations += weight * attempt * reads.0 as f64;
        model.mean_provenance_evaluations +=
            weight * (attempt * reads.1 as f64 + original_share * (current + earlier) as f64);
        model.mean_reach_evaluations += weight
            * (attempt * (reads.0 + reads.1) as f64 + original_share * (current + earlier) as f64);
        let mut draws: Vec<(Goal<G>, f64, bool)> = admitted
            .iter()
            .map(|&(goal, count)| (goal, attempt * count as f64 / admitted_count as f64, true))
            .collect();
        draws.push((desired, original_share, false));
        for (goal, probability, relabeled) in draws {
            let p = weight * probability;
            let reached = states[t + 1] == goal;
            let truncated = !reached && e.steps()[t].step.time_limit_reached;
            let done = reached || (truncated && configuration.truncation == TruncationMode::Pyalcs);
            let reward = if reached { 1000.0 } else { 0.0 };
            let flags = [
                f64::from(states[t] == goal),
                f64::from(states[..=t].contains(&goal)),
                f64::from(done),
                f64::from(!candidates.contains(&goal)),
            ];
            model.already_reached += p * flags[0];
            model.after_counterfactual_end += p * flags[1];
            model.done += p * flags[2];
            model.outside_candidates += p * flags[3];
            model.mean_reward += p * reward;
            model.mean_reward_evaluations += p;
            model.mean_reach_evaluations += p;
            model.mean_scoring_evaluations += 2.0 * p;
            let route = if relabeled {
                &mut model.relabeled
            } else {
                &mut model.original
            };
            route.already_reached += p * flags[0];
            route.after_counterfactual_end += p * flags[1];
            route.done += p * flags[2];
            route.outside_candidates += p * flags[3];
            route.reward += p * reward;
        }
    }
    model
}

fn assert_close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{what}: {actual} != {expected}"
    );
}

fn assert_route_close(actual: RouteComposition, expected: RouteComposition, what: &str) {
    let RouteComposition {
        share,
        already_reached,
        after_counterfactual_end,
        done,
        outside_candidates,
        reward,
    } = actual;
    assert_close(share, expected.share, &format!("{what} share"));
    assert_close(
        already_reached,
        expected.already_reached,
        &format!("{what} already reached"),
    );
    assert_close(
        after_counterfactual_end,
        expected.after_counterfactual_end,
        &format!("{what} after end"),
    );
    assert_close(done, expected.done, &format!("{what} done"));
    assert_close(
        outside_candidates,
        expected.outside_candidates,
        &format!("{what} outside"),
    );
    assert_close(reward, expected.reward, &format!("{what} reward"));
    match (actual.given_route(), expected.share > 0.0) {
        (None, false) => {}
        (Some(means), true) => {
            assert_close(
                means.already_reached,
                expected.already_reached / expected.share,
                what,
            );
            assert_close(
                means.after_counterfactual_end,
                expected.after_counterfactual_end / expected.share,
                what,
            );
            assert_close(means.done, expected.done / expected.share, what);
            assert_close(
                means.outside_candidates,
                expected.outside_candidates / expected.share,
                what,
            );
            assert_close(means.reward, expected.reward / expected.share, what);
        }
        _ => panic!("{what}: a route is defined exactly when it has draws"),
    }
}

fn assert_composition_close(
    actual: ExpectedComposition,
    expected: ExpectedComposition,
    what: &str,
) {
    let ExpectedComposition {
        admissible_share,
        no_admissible_goal,
        original,
        relabeled,
        fallback,
        already_reached,
        after_counterfactual_end,
        done,
        outside_candidates,
        mean_reward,
        mean_reward_evaluations,
        mean_reach_evaluations,
        mean_scoring_evaluations,
        mean_selection_evaluations,
        mean_provenance_evaluations,
    } = actual;
    assert_close(admissible_share, expected.admissible_share, what);
    assert_close(no_admissible_goal, expected.no_admissible_goal, what);
    assert_route_close(original, expected.original, &format!("{what} original"));
    assert_route_close(relabeled, expected.relabeled, &format!("{what} relabeled"));
    assert_close(fallback, expected.fallback, what);
    assert_close(already_reached, expected.already_reached, what);
    assert_close(
        after_counterfactual_end,
        expected.after_counterfactual_end,
        what,
    );
    assert_close(done, expected.done, what);
    assert_close(outside_candidates, expected.outside_candidates, what);
    assert_close(mean_reward, expected.mean_reward, what);
    assert_close(
        mean_reward_evaluations,
        expected.mean_reward_evaluations,
        what,
    );
    assert_close(
        mean_reach_evaluations,
        expected.mean_reach_evaluations,
        what,
    );
    assert_close(
        mean_scoring_evaluations,
        expected.mean_scoring_evaluations,
        what,
    );
    assert_close(
        mean_selection_evaluations,
        expected.mean_selection_evaluations,
        what,
    );
    assert_close(
        mean_provenance_evaluations,
        expected.mean_provenance_evaluations,
        what,
    );
    assert_close(
        mean_scoring_evaluations + mean_selection_evaluations + mean_provenance_evaluations,
        actual.mean_objective_evaluations(),
        what,
    );
    assert_close(original.share + relabeled.share, 1.0, what);
    assert_close(
        original.already_reached + relabeled.already_reached,
        already_reached,
        what,
    );
    assert_close(
        original.after_counterfactual_end + relabeled.after_counterfactual_end,
        after_counterfactual_end,
        what,
    );
    assert_close(original.done + relabeled.done, done, what);
    assert_close(
        original.outside_candidates + relabeled.outside_candidates,
        outside_candidates,
        what,
    );
    assert_close(original.reward + relabeled.reward, mean_reward, what);
}

#[test]
fn exact_episode_composition_matches_an_independent_enumeration_in_every_field() {
    let mut store = TrajectoryStore::new(16);
    let ids = [
        episode(&mut store, &[1, 2, 1, 3, 1], 9, true),
        episode(&mut store, &[2, 3, 7], 7, false),
        episode(&mut store, &[4, 1, 3], 3, true),
    ];
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    let mut seen = [false; 6];
    for id in ids {
        let e = store.episode(id).unwrap();
        for strategy in STRATEGIES {
            for rule in RULES {
                for filter in [false, true] {
                    for share in [0.0, 0.3, 1.0] {
                        for truncation in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
                            let configuration = SamplerConfiguration {
                                truncation,
                                ..config(strategy, rule, filter, share)
                            };
                            let actual = episode_composition::<1, 1, 2>(
                                e,
                                store.candidates(),
                                configuration,
                                &evaluator,
                            )
                            .unwrap()
                            .expected;
                            let expected =
                                modeled_composition(e, store.candidates(), configuration);
                            assert_composition_close(
                                actual,
                                expected,
                                &format!("{id:?} {configuration:?}"),
                            );
                            seen[0] |= expected.fallback > 0.0 && expected.relabeled.share > 0.0;
                            seen[1] |= expected.relabeled.after_counterfactual_end
                                > expected.relabeled.already_reached;
                            seen[2] |=
                                expected.original.done > 0.0 && expected.relabeled.done > 0.0;
                            seen[3] |= expected.relabeled.outside_candidates > 0.0
                                && expected.relabeled.outside_candidates < expected.relabeled.share;
                            seen[4] |= expected.mean_selection_evaluations > 0.0
                                && expected.mean_provenance_evaluations
                                    > expected.original.share * 2.0;
                            seen[5] |= expected.original.reward > 0.0
                                && expected.relabeled.reward > 0.0
                                && expected.original.share > 0.0;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(seen, [true; 6]);
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
    assert_eq!(sampler.counters().pooled().draws, 0);
    assert_eq!(sampler.counters().failed_draws, 1);
    assert_eq!(
        sampler.counters().cost,
        CostByPurpose {
            scoring: ObjectiveCost {
                reward_evaluations: 1,
                reach_evaluations: 1
            },
            selection: ObjectiveCost::default(),
            provenance: ObjectiveCost::reach(1),
        }
    );
}

struct NegativeAt {
    value: u8,
    rewards: Cell<u64>,
    reached: Cell<u64>,
}
impl GoalObjective<1> for NegativeAt {
    fn reward(&self, achieved: &Goal<1>, desired: &Goal<1>) -> f64 {
        self.rewards.set(self.rewards.get() + 1);
        if value(achieved) == self.value {
            -1.0
        } else if achieved == desired {
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
fn a_call_failing_on_a_later_sample_claims_none_of_its_draws_but_counts_their_work() {
    let mut store = TrajectoryStore::new(6);
    episode(&mut store, &[0, 1, 2, 3, 4, 5, 6], 9, true);
    let configuration = config(GoalStrategy::Future, RULES[1], false, 0.5);
    let mut reference = Sampler::new(
        configuration,
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    let evaluator = ObjectiveEvaluator(&OBJECTIVE);
    let first = reference.draw::<1, 1, 2>(&store, 3, &evaluator).unwrap();
    let second = reference.draw::<1, 1, 2>(&store, 40, &evaluator).unwrap();
    let early: Vec<usize> = first
        .iter()
        .chain(&second[..2])
        .map(|draw| draw.transition)
        .collect();
    let failing = (0..6)
        .find(|t| !early.contains(t) && second.iter().any(|draw| draw.transition == *t))
        .expect("a transition first drawn late in the second call");
    let fails_at = second
        .iter()
        .position(|draw| draw.transition == failing)
        .unwrap();
    assert!(fails_at >= 2);
    let objective = NegativeAt {
        value: failing as u8 + 1,
        rewards: Cell::new(0),
        reached: Cell::new(0),
    };
    let failing_evaluator = ObjectiveEvaluator(&objective);
    let mut sampler = Sampler::new(
        configuration,
        ChaChaRandomSource::from_seed_and_stream(42, SAMPLER_STREAM),
    );
    assert_eq!(
        sampler
            .draw::<1, 1, 2>(&store, 3, &failing_evaluator)
            .unwrap(),
        first
    );
    let before = sampler.counters().clone();
    assert_eq!(before, recount(&first, 0));
    assert_eq!(
        sampler.draw::<1, 1, 2>(&store, 40, &failing_evaluator),
        Err(SampleError::NegativeReward)
    );
    let mut work = before.cost;
    for draw in &second[..=fails_at] {
        work.add(draw.cost);
    }
    let after = sampler.counters();
    assert_eq!(after.original, before.original);
    assert_eq!(after.relabeled, before.relabeled);
    assert_eq!(after.fallbacks, before.fallbacks);
    assert_eq!(after.strategies, before.strategies);
    assert_eq!(after.failed_draws, 1);
    assert_eq!(after.cost, work);
    assert_eq!(
        after.cost.total(),
        ObjectiveCost {
            reward_evaluations: objective.rewards.get(),
            reach_evaluations: objective.reached.get()
        }
    );
    assert_eq!(
        after,
        &ReplayCounters {
            failed_draws: 1,
            cost: work,
            ..recount(&first, 0)
        }
    );
}

#[test]
fn a_relabeled_episode_without_admissible_transitions_cannot_be_read_as_an_ending() {
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
    let block = position_goal((3, 3));
    assert!(states(e).iter().all(|state| *state == block));
    let evaluator = ObjectiveEvaluator(env.objective());
    for truncation in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        let non_goal =
            relabel_episode::<17, 2, 19>(e, &block, RULES[1], &evaluator, truncation).unwrap();
        assert!(non_goal.samples.is_empty());
        assert_eq!(non_goal.end, EpisodeEnd::NoAdmissibleTransition);
        assert_eq!(
            non_goal.cost,
            CostByPurpose {
                selection: ObjectiveCost::reach(4),
                ..CostByPurpose::default()
            }
        );
        let counterfactual =
            relabel_episode::<17, 2, 19>(e, &block, RULES[2], &evaluator, truncation).unwrap();
        assert!(counterfactual.samples.is_empty());
        assert_eq!(counterfactual.end, EpisodeEnd::AlreadyReachedAtStart);
        let every =
            relabel_episode::<17, 2, 19>(e, &block, RULES[0], &evaluator, truncation).unwrap();
        assert_eq!(every.samples.len(), 4);
        assert_eq!(every.end, EpisodeEnd::Terminated);
    }
    assert_relabeled_episodes::<17, 2, 19>(e, &[block, position_goal((1, 1))]);
}

#[test]
fn cost_purposes_follow_what_each_rule_reads() {
    let mut store = TrajectoryStore::new(4);
    let id = episode(&mut store, &[1, 2, 1, 3, 4], 9, true);
    let e = store.episode(id).unwrap();
    let objective = CountedObjective {
        rewards: Cell::new(0),
        reached: Cell::new(0),
    };
    let evaluator = ObjectiveEvaluator(&objective);
    let expectations = [
        (GoalStrategy::Future, RULES[0], 0, 4),
        (GoalStrategy::Future, RULES[1], 1, 3),
        (GoalStrategy::Future, RULES[2], 4, 0),
        (GoalStrategy::Episode, RULES[0], 0, 10),
        (GoalStrategy::Episode, RULES[1], 4, 6),
        (GoalStrategy::Episode, RULES[2], 10, 0),
    ];
    for (strategy, rule, selection_calls, provenance_calls) in expectations {
        let before = objective.reached.get();
        let distribution = goal_distribution(
            e,
            3,
            selection(strategy, rule, false),
            store.candidates(),
            &evaluator,
        );
        assert_eq!(
            distribution.cost,
            CostByPurpose {
                scoring: ObjectiveCost::default(),
                selection: ObjectiveCost::reach(selection_calls),
                provenance: ObjectiveCost::reach(provenance_calls),
            },
            "{strategy:?} {rule:?}"
        );
        assert_eq!(
            objective.reached.get() - before,
            selection_calls + provenance_calls
        );
    }
    let (facts, queries) = acs2_trajectory::selection::goal_facts(e, 3, &goal(9), &evaluator);
    assert_eq!(facts, GoalFacts::default());
    assert_eq!(
        queries,
        FactQueries {
            current_state: 1,
            earlier_states: 3
        }
    );
    assert_eq!(
        queries.as_provenance(),
        CostByPurpose {
            provenance: ObjectiveCost::reach(4),
            ..CostByPurpose::default()
        }
    );
    assert_eq!(objective.rewards.get(), 0);
}
