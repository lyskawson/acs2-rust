use super::*;
use acs2_core::ports::goal::{ExactMatch, Goal, GoalStart, GoalStep};
use acs2_core::representation::{classifier::Classifier, symbol::Symbol};
use acs2_core::support::measurement::{
    read_match_counters, start_match_counting, stop_match_counting,
};
use acs2_trajectory::{Admissibility, Episode, ObjectiveEvaluator};
use std::collections::BTreeMap;

type TestAgent = Agent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource, ChaChaRandomSource>;

fn perception<const N: usize>(values: [u8; N]) -> Perception<N> {
    Perception {
        symbols: values.map(Symbol::Token),
    }
}

fn configuration(
    heads: usize,
    temperature: u32,
    copy_update: CopyUpdate,
    admissibility: Admissibility,
    proportion: f64,
) -> ConsistencyConfiguration {
    ConsistencyConfiguration {
        heads,
        temperature,
        reward_on_reach: 1000.0,
        hindsight: HindsightConfiguration {
            copy_update,
            admissibility,
            relabeled_proportion: proportion,
        },
    }
}

fn agent(parameters: ConsistencyConfiguration) -> TestAgent {
    let config = Configuration {
        number_of_possible_actions: 2,
        epsilon: 0.0,
        do_ga: false,
        ..Configuration::default_protocol()
    };
    TestAgent::seeded(
        config,
        EpisodicConfiguration {
            capacity: 100,
            warmup: 1,
            updates_per_step: 3,
            truncation: acs2_core::trial::TruncationMode::Bootstrap,
        },
        parameters,
        42,
        1,
    )
}

fn classifier(
    state: u8,
    goal: u8,
    action: usize,
    value: f64,
    change: bool,
    config: &Configuration,
) -> Classifier<2> {
    let mut classifier = Classifier::general(Some(action), config);
    classifier.condition.symbols = [Symbol::Token(state), Symbol::Token(goal)];
    if change {
        classifier.effect.symbols[0] = Symbol::Token(8);
    }
    classifier.q = 1.0;
    classifier.r = value;
    classifier
}

fn fill_store(store: &mut TrajectoryStore<1, 1>) {
    for (desired, states) in [(9, vec![0, 1, 0, 2]), (8, vec![2, 0, 2])] {
        let mut episode = Episode::new(GoalStart {
            observation: perception([states[0]]),
            achieved: Goal::new([Symbol::Token(states[0])]),
            desired: Goal::new([Symbol::Token(desired)]),
        });
        for (index, &state) in states.iter().enumerate().skip(1) {
            episode.push(
                0,
                GoalStep {
                    observation: perception([state]),
                    achieved: Goal::new([Symbol::Token(state)]),
                    terminal_state: false,
                    time_limit_reached: index + 1 == states.len(),
                },
            );
        }
        store.insert(episode).unwrap();
    }
}

#[test]
fn priority_uses_population_variance_and_the_exact_integer_exponent() {
    assert_eq!(population_variance(&[0.0, 1000.0]), 250_000.0);
    assert_eq!(population_variance(&[0.0, 0.0, 1000.0, 1000.0]), 250_000.0);
    assert_eq!(population_variance(&[42.0]), 0.0);
    for temperature in 0..=MAX_TEMPERATURE {
        let maximum = 250_000.0;
        let base = maximum - 125_000.0 + priority::EPSILON;
        let mut expected = 1.0;
        let mut upper = 1.0;
        for _ in 0..temperature {
            expected *= base;
            upper *= maximum + priority::EPSILON;
        }
        assert_eq!(
            acceptance(125_000.0, maximum, temperature).to_bits(),
            (expected / upper).to_bits()
        );
        assert!(acceptance(maximum, maximum, temperature) > 0.0);
        assert_eq!(acceptance(0.0, maximum, temperature), 1.0);
    }
}

#[test]
fn no_change_classifiers_do_not_affect_values_or_variance_and_queries_are_read_only() {
    let mut learner = agent(configuration(
        2,
        4,
        CopyUpdate::Full,
        Admissibility::EveryTransition,
        2.0 / 3.0,
    ));
    let sample = ReplaySample {
        state: perception([0, 9]),
        next_state: perception([1, 9]),
        action: 0,
        reward: 0.0,
        done: false,
    };
    learner.heads[0].population =
        Population::from_classifiers(vec![classifier(0, 9, 0, 10.0, true, learner.config())]);
    learner.heads[1].population =
        Population::from_classifiers(vec![classifier(0, 9, 0, 30.0, true, learner.config())]);
    assert_eq!(learner.variance(&sample), 100.0);
    let mut rules = learner.heads[0].population.classifiers().to_vec();
    rules.push(classifier(0, 9, 0, 1000.0, false, learner.config()));
    learner.heads[0].population = Population::from_classifiers(rules);
    let before = learner.snapshot();
    assert_eq!(learner.variance(&sample), 100.0);
    assert_eq!(learner.snapshot(), before);
    let other = ReplaySample {
        action: 1,
        ..sample
    };
    assert_eq!(learner.variance(&other), 0.0);
}

struct NoRandom;
impl RandomSource for NoRandom {
    fn gen_bool(&mut self, _: f64) -> bool {
        panic!("probability one must consume no acceptance randomness")
    }
    fn gen_range(&mut self, _: usize) -> usize {
        panic!("unexpected range draw")
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("unexpected unit draw")
    }
}

#[test]
fn probability_one_keeps_the_proposal_order_and_the_sampler_stream() {
    for (heads, temperature) in [(1, 0), (1, 24), (2, 0), (4, 0)] {
        let parameters = configuration(
            heads,
            temperature,
            CopyUpdate::Full,
            Admissibility::EveryTransition,
            2.0 / 3.0,
        );
        let mut learner = Agent::<1, 1, 2, _, _, _>::new(
            agent(parameters).config,
            agent(parameters).replay,
            parameters,
            (0..heads)
                .map(|_| ChaChaRandomSource::from_seed(42))
                .collect(),
            ChaChaRandomSource::from_seed_and_stream(42, 7),
            NoRandom,
        );
        fill_store(&mut learner.store);
        let objective = ExactMatch {
            reward_on_reach: 1000.0,
        };
        let evaluator = ObjectiveEvaluator(&objective);
        let mut sampler = Sampler::new(
            learner.sampler.configuration(),
            ChaChaRandomSource::from_seed_and_stream(42, 7),
        );
        let expected = sampler
            .draw::<1, 1, 2>(&learner.store, 60, &evaluator)
            .unwrap();
        let actual = learner.draw_batch(60, &evaluator);
        assert_eq!(actual, expected);
        assert_eq!(
            learner.sampler.random_source().capture_state(),
            sampler.random_source().capture_state()
        );
        assert_eq!(learner.sampler.counters(), sampler.counters());
        assert_eq!(learner.priority.acceptance_coins, 0);
    }
}

#[test]
fn retained_future_proposals_reweighted_by_priority_match_independent_enumeration() {
    let objective = ExactMatch {
        reward_on_reach: 1000.0,
    };
    let evaluator = ObjectiveEvaluator(&objective);
    for rule in [
        Admissibility::EveryTransition,
        Admissibility::CounterfactualEpisode,
    ] {
        for proportion in [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0] {
            let mut learner = agent(configuration(2, 4, CopyUpdate::Full, rule, proportion));
            fill_store(&mut learner.store);
            let mut rules = vec![];
            for state in 0..=2 {
                for goal in [0, 1, 2, 8, 9] {
                    let value = 50.0 * f64::from(1 + state + goal);
                    rules.push(classifier(
                        state,
                        goal,
                        0,
                        value.min(1000.0),
                        true,
                        learner.config(),
                    ));
                }
            }
            learner.heads[1].population = Population::from_classifiers(rules);
            let mut mass = BTreeMap::new();
            for (desired, states) in [(9, vec![0, 1, 0, 2]), (8, vec![2, 0, 2])] {
                for t in 0..states.len() - 1 {
                    let mut future = vec![];
                    for &goal in &states[t + 1..] {
                        if rule == Admissibility::EveryTransition || !states[..=t].contains(&goal) {
                            future.push(goal);
                        }
                    }
                    let original =
                        1.0 - proportion + if future.is_empty() { proportion } else { 0.0 };
                    *mass
                        .entry((states[t], states[t + 1], desired, false))
                        .or_insert(0.0) += original / 5.0;
                    for &goal in &future {
                        *mass
                            .entry((states[t], states[t + 1], goal, true))
                            .or_insert(0.0) += proportion / 5.0 / future.len() as f64;
                    }
                }
            }
            let mut weights = BTreeMap::new();
            let mut mean_acceptance = 0.0;
            for (key, proposal) in &mass {
                let value = (50.0 * f64::from(1 + key.0 + key.2)).min(1000.0);
                let variance = value * value / 4.0;
                let ratio = (250_000.0 - variance + 1e-6) / (250_000.0 + 1e-6);
                let weight = proposal * ratio * ratio * ratio * ratio;
                weights.insert(*key, weight);
                mean_acceptance += weight;
            }
            let samples = 30_000;
            let draws = learner.draw_batch(samples, &evaluator);
            if rule == Admissibility::CounterfactualEpisode && proportion > 0.0 {
                assert!(learner.accepted.fallbacks > 0);
            }
            let mut observed = BTreeMap::new();
            for drawn in &draws {
                let symbols = |p: Perception<2>, index: usize| match p.symbols[index] {
                    Symbol::Token(v) => v,
                    _ => panic!("token"),
                };
                let key = (
                    symbols(drawn.scored.sample.state, 0),
                    symbols(drawn.scored.sample.next_state, 0),
                    symbols(drawn.scored.sample.state, 1),
                    drawn.provenance.origin == SampleOrigin::Relabeled,
                );
                *observed.entry(key).or_insert(0) += 1;
                assert_eq!(
                    drawn.scored.sample.reward,
                    if key.1 == key.2 { 1000.0 } else { 0.0 }
                );
                assert_eq!(drawn.scored.sample.done, key.1 == key.2);
            }
            for (key, weight) in weights {
                let probability = weight / mean_acceptance;
                let actual = f64::from(*observed.get(&key).unwrap_or(&0)) / samples as f64;
                let tolerance = 7.0 * (probability * (1.0 - probability) / samples as f64).sqrt()
                    + 1.0 / samples as f64;
                assert!(
                    (actual - probability).abs() <= tolerance,
                    "{rule:?} {proportion}: {key:?}: {actual} vs {probability}"
                );
            }
            let attempts = learner.priority.proposals as f64 / samples as f64;
            assert!((attempts - 1.0 / mean_acceptance).abs() < 0.15);
        }
    }
}

#[test]
fn all_priority_queries_precede_the_unweighted_hindsight_steps() {
    let mut learner = agent(configuration(
        2,
        4,
        CopyUpdate::Full,
        Admissibility::EveryTransition,
        2.0 / 3.0,
    ));
    fill_store(&mut learner.store);
    let objective = ExactMatch {
        reward_on_reach: 1000.0,
    };
    let evaluator = ObjectiveEvaluator(&objective);
    start_match_counting();
    let draws = learner.draw_batch(30, &evaluator);
    let formations = read_match_counters().unwrap().formations;
    stop_match_counting();
    assert_eq!(formations, learner.priority.value_queries);
    assert_eq!(learner.counts().full, 0);
    assert!(learner.heads.iter().all(|head| head.population.is_empty()));
    let mut expected = Population::new();
    let mut rng = ChaChaRandomSource::from_seed_and_stream(42, 1);
    for drawn in &draws {
        replay_learning_step(
            &mut expected,
            learner.config(),
            &MaxFitnessBootstrap,
            &drawn.scored.sample,
            100,
            &mut rng,
        );
        learner.learn_draw(0, drawn, 100);
    }
    assert_eq!(
        format!("{:?}", expected.classifiers()),
        format!("{:?}", learner.heads[0].population.classifiers())
    );
    assert!(learner.heads[1].population.is_empty());
    assert_eq!(learner.heads[0].rng.capture_state(), rng.capture_state());
    assert_eq!(learner.counts().full, 30);
}

#[test]
fn seeded_heads_and_acceptance_have_distinct_streams_and_head_zero_uses_the_agent_stream() {
    let learner = agent(configuration(
        4,
        1,
        CopyUpdate::Full,
        Admissibility::EveryTransition,
        2.0 / 3.0,
    ));
    assert_eq!(
        learner.heads[0].rng.capture_state(),
        ChaChaRandomSource::from_seed_and_stream(42, 1).capture_state()
    );
    for (head, stream) in learner.heads.iter().zip([1, 8, 9, 10]) {
        assert_eq!(head.rng.capture_state().unwrap().stream, stream);
    }
    assert_eq!(
        learner
            .sampler
            .random_source()
            .capture_state()
            .unwrap()
            .stream,
        7
    );
    assert_eq!(learner.acceptance_rng.capture_state().unwrap().stream, 12);
    let before = learner.snapshot();
    assert_eq!(learner.snapshot(), before);
}

struct FixedEpisode {
    steps: usize,
    length: usize,
    resets: usize,
}

impl acs2_core::ports::environment::Environment<2> for FixedEpisode {
    fn reset(&mut self) -> Perception<2> {
        self.steps = 0;
        self.resets += 1;
        perception([0, 9])
    }
    fn step(&mut self, _: usize) -> acs2_core::ports::environment::StepOutcome<2> {
        self.steps += 1;
        acs2_core::ports::environment::StepOutcome {
            observation: perception([(self.steps % 3) as u8, 9]),
            reward: 0.0,
            terminated: false,
            truncated: self.steps == self.length,
            info: (),
        }
    }
}

impl GoalEvaluator<1, 1> for FixedEpisode {
    fn outcome(
        &self,
        step: &GoalStep<1, 1>,
        goal: &Goal<1>,
    ) -> acs2_core::ports::goal::GoalOutcome {
        step.outcome(
            &ExactMatch {
                reward_on_reach: 1000.0,
            },
            goal,
        )
    }
    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        achieved == desired
    }
}

impl ReplayEnvironment<1, 1, 2> for FixedEpisode {
    fn episode_start(&self) -> GoalStart<1, 1> {
        GoalStart {
            observation: perception([0]),
            achieved: Goal::new([Symbol::Token(0)]),
            desired: Goal::new([Symbol::Token(9)]),
        }
    }
    fn last_step(&self) -> GoalStep<1, 1> {
        let state = (self.steps % 3) as u8;
        GoalStep {
            observation: perception([state]),
            achieved: Goal::new([Symbol::Token(state)]),
            terminal_state: false,
            time_limit_reached: self.steps == self.length,
        }
    }
}

#[test]
fn one_joint_interaction_budget_rotates_heads_and_only_the_acting_head_learns_three_draws_per_step()
{
    for heads in [1, 2, 4] {
        let mut learner = agent(configuration(
            heads,
            4,
            CopyUpdate::Full,
            Admissibility::EveryTransition,
            2.0 / 3.0,
        ));
        let mut env = FixedEpisode {
            steps: 0,
            length: 1,
            resets: 0,
        };
        start_match_counting();
        for episode in 0..12 {
            assert_eq!(learner.playing_head(), episode % heads);
            let before: Vec<_> = learner
                .heads
                .iter()
                .map(|head| {
                    (
                        head.population.classifiers().to_vec(),
                        head.counts,
                        head.rng.capture_state(),
                    )
                })
                .collect();
            learner.train_episode(&mut env, episode as u64);
            assert_eq!(env.steps, 1);
            assert_eq!(env.resets, episode + 1);
            for (index, head) in learner.heads.iter().enumerate() {
                if index == episode % heads {
                    assert_eq!(head.counts.full, before[index].1.full + 3);
                    assert_eq!(
                        head.counts.environment_steps,
                        before[index].1.environment_steps + 1
                    );
                } else {
                    assert_eq!(
                        format!("{:?}", head.population.classifiers()),
                        format!("{:?}", before[index].0)
                    );
                    assert_eq!(head.counts, before[index].1);
                    assert_eq!(head.rng.capture_state(), before[index].2);
                }
            }
        }
        let counters = read_match_counters().unwrap();
        stop_match_counting();
        assert_eq!(learner.counts().environment_steps, 12);
        assert_eq!(learner.counts().full, 36);
        assert_eq!(
            counters.formations,
            12 + 3 * 36 + learner.priority.value_queries
        );
        assert_eq!(learner.store.len(), 12);
        for head in &learner.heads {
            assert_eq!(head.counts.full, 36 / heads as u64);
        }
    }
}

#[test]
fn train_episode_freezes_all_draw_priorities_before_any_head_learns() {
    use acs2_core::ports::environment::Environment;
    let parameters = configuration(
        2,
        24,
        CopyUpdate::Full,
        Admissibility::EveryTransition,
        2.0 / 3.0,
    );
    let make = || {
        let mut learner = agent(parameters);
        let mut rules = vec![];
        for state in 0..3 {
            for goal in [0, 1, 2, 9] {
                rules.push(classifier(state, goal, 0, 200.0, true, learner.config()));
            }
        }
        learner.heads[1].population = Population::from_classifiers(rules);
        learner
    };
    let mut actual = make();
    let mut expected = make();
    let mut env = FixedEpisode {
        steps: 0,
        length: 3,
        resets: 0,
    };
    actual.train_episode(&mut env, 0);
    let mut env = FixedEpisode {
        steps: 0,
        length: 3,
        resets: 0,
    };
    let mut state = env.reset();
    let mut episode = expected.store.begin_episode(env.episode_start());
    for _ in 0..3 {
        let action = expected.explore_action(&state);
        state = env.step(action).observation;
        episode.push(action, env.last_step());
    }
    expected.store.insert(episode).unwrap();
    expected.heads[0].counts.environment_steps = 3;
    expected.heads[0].episodes = 1;
    expected.episodes = 1;
    let frozen = expected.draw_batch(9, &env);
    for drawn in &frozen {
        expected.learn_draw(0, drawn, 3);
    }
    assert_eq!(actual.snapshot(), expected.snapshot());
}

#[test]
fn values_copies_use_the_same_unweighted_step_and_the_match_identity_includes_two_formations() {
    let parameters = configuration(
        2,
        4,
        CopyUpdate::ValuesOnly,
        Admissibility::CounterfactualEpisode,
        2.0 / 3.0,
    );
    let mut learner = agent(parameters);
    fill_store(&mut learner.store);
    let objective = ExactMatch {
        reward_on_reach: 1000.0,
    };
    let evaluator = ObjectiveEvaluator(&objective);
    let draws = learner.draw_batch(40, &evaluator);
    let mut expected = Population::new();
    let mut rng = ChaChaRandomSource::from_seed_and_stream(42, 1);
    start_match_counting();
    for drawn in &draws {
        if drawn.provenance.origin == SampleOrigin::Relabeled {
            values_learning_step(&mut expected, learner.config(), &drawn.scored.sample);
        } else {
            replay_learning_step(
                &mut expected,
                learner.config(),
                &MaxFitnessBootstrap,
                &drawn.scored.sample,
                50,
                &mut rng,
            );
        }
    }
    stop_match_counting();
    start_match_counting();
    for drawn in &draws {
        learner.learn_draw(0, drawn, 50);
    }
    let counters = read_match_counters().unwrap();
    stop_match_counting();
    assert!(learner.counts().full > 0 && learner.counts().values_only > 0);
    assert_eq!(
        counters.formations,
        3 * learner.counts().full + 2 * learner.counts().values_only
    );
    assert_eq!(
        format!("{:?}", expected.classifiers()),
        format!("{:?}", learner.heads[0].population.classifiers())
    );
    assert_eq!(rng.capture_state(), learner.heads[0].rng.capture_state());
}

#[test]
fn original_and_copied_rewards_and_endings_use_the_objective_and_the_agents_truncation_mode() {
    use acs2_core::trial::TruncationMode;
    for mode in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        for proportion in [0.0, 1.0] {
            let mut learner = agent(configuration(
                2,
                1,
                CopyUpdate::Full,
                Admissibility::EveryTransition,
                proportion,
            ));
            learner.replay.truncation = mode;
            let config = SamplerConfiguration {
                truncation: mode,
                ..learner.sampler.configuration()
            };
            learner.sampler = Sampler::new(config, ChaChaRandomSource::from_seed_and_stream(42, 7));
            fill_store(&mut learner.store);
            let objective = ExactMatch {
                reward_on_reach: 500.0,
            };
            let draws = learner.draw_batch(100, &ObjectiveEvaluator(&objective));
            assert_eq!(
                learner.sampler.configuration().truncation,
                learner.replay.truncation
            );
            for drawn in draws {
                let episode = learner
                    .store
                    .episodes()
                    .find(|episode| episode.id() == drawn.episode)
                    .unwrap();
                let step = &episode.steps()[drawn.transition].step;
                let reached = step.achieved == drawn.goal;
                let truncated = !reached && step.time_limit_reached;
                assert_eq!(
                    drawn.scored.sample.reward,
                    if reached { 500.0 } else { 0.0 }
                );
                assert_eq!(
                    drawn.scored.sample.done,
                    reached || (truncated && mode == TruncationMode::Pyalcs)
                );
                assert_eq!(drawn.scored.outcome.terminated, reached);
                assert_eq!(drawn.scored.outcome.truncated, truncated);
                assert_eq!(drawn.scored.sample.state.symbols[1], drawn.goal.symbols[0]);
                assert_eq!(
                    drawn.scored.sample.next_state.symbols[1],
                    drawn.goal.symbols[0]
                );
            }
        }
    }
}

struct ExploreGreedyBranch;

impl RandomSource for ExploreGreedyBranch {
    fn gen_bool(&mut self, probability: f64) -> bool {
        assert_eq!(probability, 0.8);
        false
    }
    fn gen_range(&mut self, _: usize) -> usize {
        0
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("unexpected unit draw")
    }
}

#[test]
fn every_rotating_head_explores_on_the_goal_join_under_epsilon_point_eight() {
    let parameters = configuration(
        4,
        1,
        CopyUpdate::Full,
        Admissibility::EveryTransition,
        2.0 / 3.0,
    );
    let config = Configuration {
        epsilon: 0.8,
        number_of_possible_actions: 2,
        do_ga: false,
        ..Configuration::default_protocol()
    };
    let replay = agent(parameters).replay_configuration();
    let populations = (0..4)
        .map(|head| {
            let mut rules = vec![];
            for goal in [7, 9] {
                for action in [0, 1] {
                    let best = if goal == 7 { head % 2 } else { 1 - head % 2 };
                    rules.push(classifier(
                        0,
                        goal,
                        action,
                        if action == best { 100.0 } else { 1.0 },
                        true,
                        &config,
                    ));
                }
            }
            Population::from_classifiers(rules)
        })
        .collect();
    let mut learner = Agent::<1, 1, 2, _, _, _>::new(
        config,
        replay,
        parameters,
        (0..4).map(|_| ExploreGreedyBranch).collect(),
        ChaChaRandomSource::from_seed_and_stream(42, 7),
        ChaChaRandomSource::from_seed_and_stream(42, 12),
    )
    .with_populations(populations);
    for head in 0..4 {
        learner.episodes = head as u64;
        assert_eq!(learner.explore_action(&perception([0, 7])), head % 2);
        assert_eq!(learner.explore_action(&perception([0, 9])), 1 - head % 2);
    }
}
