use super::*;
use crate::episodic_replay::{EpisodicAgent, EpisodicConfiguration};
use acs2_core::learning::replay::{replay_learning_step, ReplaySample};
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::ports::goal::{ExactMatch, GoalOutcome, GoalStart, GoalStep};
use acs2_core::representation::{classifier::Classifier, perception::Perception, symbol::Symbol};
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::TruncationMode;
use acs2_trajectory::{
    DrawnSample, Episode, EpisodeId, GoalStrategy, ObjectiveEvaluator, Provenance, Sampler,
    ScoredSample, TrajectoryStore,
};

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}

fn config() -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        ..Configuration::default_protocol()
    }
}

fn sample(desired: u8, origin: SampleOrigin, fallback: bool) -> DrawnSample<1, 3> {
    DrawnSample {
        episode: EpisodeId(0),
        transition: 0,
        goal: goal(desired),
        scored: ScoredSample {
            sample: ReplaySample {
                state: Perception::new([
                    Symbol::Token(0),
                    Symbol::Token(0),
                    Symbol::Token(desired),
                ]),
                next_state: Perception::new([
                    Symbol::Token(1),
                    Symbol::Token(0),
                    Symbol::Token(desired),
                ]),
                action: 0,
                reward: 1000.0,
                done: false,
            },
            outcome: GoalOutcome {
                reward: 1000.0,
                terminated: false,
                truncated: false,
            },
            cost: Default::default(),
        },
        provenance: Provenance {
            origin,
            fallback,
            strategy: GoalStrategy::Future,
            requested_strategy: GoalStrategy::Future,
            already_reached: false,
            after_counterfactual_end: false,
            done: false,
            outside_candidates: false,
        },
        cost: Default::default(),
    }
}

fn rule(action: usize, config: &Configuration) -> Classifier<3> {
    let mut classifier = Classifier::general(Some(action), config);
    classifier.condition.symbols[0] = Symbol::Token(0);
    classifier.effect.symbols[0] = Symbol::Token(1);
    classifier
}

fn replay() -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity: 100,
        warmup: 1,
        updates_per_step: 3,
        truncation: TruncationMode::Bootstrap,
    }
}

type TestAgent = Box<
    dyn EpisodicAgent<
        2,
        1,
        3,
        PolicyRandom = ChaChaRandomSource,
        SamplerRandom = ChaChaRandomSource,
    >,
>;

fn agent(shortened: bool, per_goal: bool, config: Configuration, goals: &[Goal<1>]) -> TestAgent {
    let rng = ChaChaRandomSource::from_seed(42);
    let sampler = ChaChaRandomSource::from_seed(43);
    match (shortened, per_goal) {
        (false, false) => Box::new(crate::acs2er::episodic::Agent::new(
            config,
            replay(),
            rng,
            sampler,
        )),
        (false, true) => Box::new(crate::acs2er::episodic::PerGoalRewardAgent::new(
            config,
            replay(),
            goals,
            rng,
            sampler,
        )),
        (true, false) => Box::new(crate::acs2her::Agent::new(
            config,
            replay(),
            crate::acs2her::Preset::Acs2HerShortened {
                proportion: 2.0 / 3.0,
            },
            rng,
            sampler,
        )),
        (true, true) => Box::new(crate::acs2her::per_goal_reward::Agent::new(
            config,
            replay(),
            goals,
            rng,
            sampler,
        )),
    }
}

fn materialized(
    population: &Population<3>,
    rewards: &PerGoalRewards<1>,
    desired: &Goal<1>,
) -> String {
    let rules: Vec<_> = population
        .classifiers()
        .iter()
        .enumerate()
        .map(|(index, classifier)| {
            let mut classifier = classifier.clone();
            classifier.r = rewards.entry(index, desired).r;
            classifier.ir = rewards.entry(index, desired).ir;
            classifier
        })
        .collect();
    format!("{rules:?}")
}

#[test]
fn one_goal_learning_preserves_full_base_state_and_rng_bits_through_alp_and_removal() {
    for seed in 42..47 {
        let config = config();
        let mut actual = Population::new();
        let mut expected = Population::new();
        let mut rewards = PerGoalRewards::new(&[goal(9)], &config, 0);
        let mut rng = ChaChaRandomSource::from_seed(seed);
        let mut oracle = ChaChaRandomSource::from_seed(seed);
        for time in 1..=120 {
            let mut draw = sample(
                9,
                if time % 3 == 0 {
                    SampleOrigin::Relabeled
                } else {
                    SampleOrigin::Original
                },
                time % 5 == 0,
            );
            draw.scored.sample.done = time % 4 == 0;
            draw.scored.sample.reward = (time % 7) as f64 * 100.0;
            draw.scored.sample.state.symbols[0] = Symbol::Token((time % 3) as u8);
            draw.scored.sample.next_state.symbols[0] = Symbol::Token(((time + 1) % 3) as u8);
            draw.scored.sample.state.symbols[1] = Symbol::Token((time % 2) as u8);
            draw.scored.sample.next_state.symbols[1] = Symbol::Token(((time / 3) % 2) as u8);
            replay_learning_step(
                &mut expected,
                &config,
                &MaxFitnessBootstrap,
                &draw.scored.sample,
                time,
                &mut oracle,
            );
            rewards.learn::<2, 3>(&mut actual, &config, &draw, time, &mut rng);
            assert_eq!(
                materialized(&actual, &rewards, &goal(9)),
                format!("{:?}", expected.classifiers())
            );
            assert_eq!(rng.capture_state(), oracle.capture_state());
        }
        assert!(rewards.updates().original > 0 && rewards.updates().relabeled > 0);
    }
}

#[test]
fn both_actual_new_arms_reproduce_one_goal_draws_actions_and_sampler_records() {
    for shortened in [false, true] {
        verify_one_goal_episode_driver(shortened);
        let mut base = agent(shortened, false, config(), &[goal(9)]);
        let mut actual = agent(shortened, true, config(), &[goal(9)]);
        assert!(actual.learner().per_goal_rewards().is_some());
        assert_eq!(actual.learner().config().initial_r, 0.5);
        let mut episode = Episode::new(GoalStart {
            observation: Perception::new([Symbol::Token(0); 2]),
            achieved: goal(9),
            desired: goal(9),
        });
        for index in 0..4 {
            episode.push(
                0,
                GoalStep {
                    observation: Perception::new([Symbol::Token(index % 2), Symbol::Token(0)]),
                    achieved: goal(9),
                    terminal_state: false,
                    time_limit_reached: index == 3,
                },
            );
        }
        let mut store = TrajectoryStore::new(100);
        store.insert(episode).unwrap();
        let mut sampler = Sampler::new(
            actual.learner().sampler().configuration(),
            ChaChaRandomSource::from_seed(43),
        );
        let mut oracle = Sampler::new(
            base.learner().sampler().configuration(),
            ChaChaRandomSource::from_seed(43),
        );
        let objective = ObjectiveEvaluator(&ExactMatch {
            reward_on_reach: 1000.0,
        });
        let draws = sampler.draw::<2, 1, 3>(&store, 160, &objective).unwrap();
        let expected = oracle.draw::<2, 1, 3>(&store, 160, &objective).unwrap();
        assert_eq!(format!("{draws:?}"), format!("{expected:?}"));
        assert_eq!(sampler.counters(), oracle.counters());
        assert_eq!(
            sampler.random_source().capture_state(),
            oracle.random_source().capture_state()
        );
        if shortened {
            assert!(draws.iter().any(|draw| draw.provenance.fallback));
        }
        for (index, draw) in draws.iter().enumerate() {
            assert_eq!(
                actual
                    .learner_mut()
                    .explore_action(&draw.scored.sample.state),
                base.learner_mut().explore_action(&draw.scored.sample.state)
            );
            actual.learner_mut().learn_draw(draw, index as u64);
            base.learner_mut().learn_draw(draw, index as u64);
            let mut rng = ChaChaRandomSource::from_seed(index as u64);
            let mut expected_rng = ChaChaRandomSource::from_seed(index as u64);
            let (action, value) = actual
                .learner()
                .greedy_action(&draw.scored.sample.state, &mut rng);
            let (expected_action, expected_value) = base
                .learner()
                .greedy_action(&draw.scored.sample.state, &mut expected_rng);
            assert_eq!(
                (action, value.to_bits()),
                (expected_action, expected_value.to_bits())
            );
            assert_eq!(rng.capture_state(), expected_rng.capture_state());
            assert_eq!(
                materialized(
                    actual.learner().population(),
                    actual.learner().per_goal_rewards().unwrap(),
                    &goal(9)
                ),
                format!("{:?}", base.learner().population().classifiers())
            );
            let actual_snapshot = actual.learner().snapshot();
            let base_snapshot = base.learner().snapshot();
            assert_eq!(
                actual_snapshot
                    .split('|')
                    .skip(1)
                    .take(10)
                    .collect::<Vec<_>>(),
                base_snapshot
                    .split('|')
                    .skip(1)
                    .take(10)
                    .collect::<Vec<_>>()
            );
        }
    }
}

struct OneGoalEnvironment {
    cycle: u8,
    start: GoalStart<2, 1>,
    step: GoalStep<2, 1>,
}

impl OneGoalEnvironment {
    fn new() -> Self {
        Self {
            cycle: 0,
            start: GoalStart {
                observation: Perception::new([Symbol::Token(0); 2]),
                achieved: goal(8),
                desired: goal(9),
            },
            step: GoalStep {
                observation: Perception::new([Symbol::Token(0); 2]),
                achieved: goal(9),
                terminal_state: false,
                time_limit_reached: false,
            },
        }
    }
}

impl acs2_core::ports::environment::Environment<3> for OneGoalEnvironment {
    fn reset(&mut self) -> Perception<3> {
        self.start.observation.symbols[0] = Symbol::Token(self.cycle % 3);
        self.start.achieved = goal(8 + self.cycle % 2);
        self.cycle += 1;
        acs2_core::ports::goal::GoalLayout::<2, 1, 3>::join(
            &self.start.observation,
            &self.start.desired,
        )
    }

    fn step(&mut self, action: usize) -> acs2_core::ports::environment::StepOutcome<3> {
        self.step.observation = Perception::new([
            Symbol::Token((self.cycle + action as u8) % 3),
            Symbol::Token(action as u8),
        ]);
        acs2_core::ports::environment::StepOutcome {
            observation: acs2_core::ports::goal::GoalLayout::<2, 1, 3>::join(
                &self.step.observation,
                &goal(9),
            ),
            reward: 1000.0,
            terminated: true,
            truncated: false,
            info: (),
        }
    }
}

impl acs2_trajectory::GoalEvaluator<2, 1> for OneGoalEnvironment {
    fn outcome(&self, step: &GoalStep<2, 1>, desired: &Goal<1>) -> GoalOutcome {
        step.outcome(
            &ExactMatch {
                reward_on_reach: 1000.0,
            },
            desired,
        )
    }
    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        achieved == desired
    }
}

impl crate::episodic_replay::ReplayEnvironment<2, 1, 3> for OneGoalEnvironment {
    fn episode_start(&self) -> GoalStart<2, 1> {
        self.start
    }
    fn last_step(&self) -> GoalStep<2, 1> {
        self.step
    }
}

fn verify_one_goal_episode_driver(shortened: bool) {
    let mut base = agent(shortened, false, config(), &[goal(9)]);
    let mut actual = agent(shortened, true, config(), &[goal(9)]);
    let mut base_env = OneGoalEnvironment::new();
    let mut actual_env = OneGoalEnvironment::new();
    for time in 0..120 {
        base.learner_mut().train_episode(&mut base_env, time);
        actual.learner_mut().train_episode(&mut actual_env, time);
        assert_eq!(
            materialized(
                actual.learner().population(),
                actual.learner().per_goal_rewards().unwrap(),
                &goal(9)
            ),
            format!("{:?}", base.learner().population().classifiers())
        );
        let actual_snapshot = actual.learner().snapshot();
        let base_snapshot = base.learner().snapshot();
        assert_eq!(
            actual_snapshot
                .split('|')
                .skip(1)
                .take(10)
                .collect::<Vec<_>>(),
            base_snapshot
                .split('|')
                .skip(1)
                .take(10)
                .collect::<Vec<_>>()
        );
    }
    assert_eq!(actual.learner().sampler().counters().pooled().draws, 360);
    assert!(
        actual
            .learner()
            .per_goal_rewards()
            .unwrap()
            .updates()
            .original
            > 0
    );
}

#[test]
fn draw_updates_only_its_action_set_goal_including_relabeling_and_fallbacks() {
    for (desired, origin, fallback) in [
        (8, SampleOrigin::Relabeled, false),
        (9, SampleOrigin::Original, false),
        (9, SampleOrigin::Original, true),
    ] {
        let config = config();
        let mut population = Population::from_classifiers(vec![rule(0, &config), rule(1, &config)]);
        let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 2);
        for reference in 0..2 {
            for value in [8, 9] {
                let entry = rewards.entry_mut(reference, &goal(value));
                entry.r = (reference * 100 + value as usize) as f64;
                entry.ir = value as f64 * 2.0;
            }
        }
        let before = rewards.clone();
        rewards.learn::<2, 3>(
            &mut population,
            &config,
            &sample(desired, origin, fallback),
            1,
            &mut ChaChaRandomSource::from_seed(42),
        );
        for reference in 0..2 {
            for value in [8, 9] {
                if reference == 0 && value == desired {
                    assert_ne!(
                        rewards.entry(reference, &goal(value)).r.to_bits(),
                        before.entry(reference, &goal(value)).r.to_bits()
                    );
                    assert_ne!(
                        rewards.entry(reference, &goal(value)).ir.to_bits(),
                        before.entry(reference, &goal(value)).ir.to_bits()
                    );
                    assert!(rewards.entry(reference, &goal(value)).updated);
                } else {
                    assert_eq!(
                        rewards.entry(reference, &goal(value)),
                        before.entry(reference, &goal(value))
                    );
                }
            }
        }
        assert_eq!(
            rewards.updates(),
            match origin {
                SampleOrigin::Original => RewardUpdates {
                    original: 1,
                    relabeled: 0
                },
                SampleOrigin::Relabeled => RewardUpdates {
                    original: 0,
                    relabeled: 1
                },
            }
        );
        assert_eq!(rewards.updated_entries(), 1);
        assert_eq!(rewards.existing_pairs(), 4);
    }
}

#[test]
fn equal_values_remain_separate_entries_and_only_the_draw_entry_is_marked_updated() {
    let config = config();
    let mut population = Population::from_classifiers(vec![rule(0, &config)]);
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 1);
    for desired in [goal(8), goal(9)] {
        *rewards.entry_mut(0, &desired) = GoalValue {
            r: 3.0,
            ir: 3.0,
            updated: false,
        };
    }
    let mut draw = sample(8, SampleOrigin::Original, false);
    draw.scored.sample.done = true;
    draw.scored.sample.reward = 3.0;
    rewards.learn::<2, 3>(
        &mut population,
        &config,
        &draw,
        1,
        &mut ChaChaRandomSource::from_seed(42),
    );
    assert_eq!(
        rewards.entry(0, &goal(8)),
        &GoalValue {
            r: 3.0,
            ir: 3.0,
            updated: true
        }
    );
    assert_eq!(
        rewards.entry(0, &goal(9)),
        &GoalValue {
            r: 3.0,
            ir: 3.0,
            updated: false
        }
    );
    draw.scored.sample.reward = 1000.0;
    rewards.learn::<2, 3>(
        &mut population,
        &config,
        &draw,
        2,
        &mut ChaChaRandomSource::from_seed(42),
    );
    assert!(rewards.entry(0, &goal(8)).r > 3.0);
    assert_eq!(rewards.entry(0, &goal(9)).r, 3.0);
}

#[test]
fn goal_values_flip_action_and_bootstrap_and_ignore_shared_rewards_numerosity_and_null_effects_correctly(
) {
    let config = config();
    let mut first = rule(0, &config);
    first.q = 0.5;
    first.num = 3;
    let mut second = rule(1, &config);
    second.q = 1.0;
    let mut null = Classifier::general(Some(0), &config);
    null.q = 1.0;
    let mut population = Population::from_classifiers(vec![first, second, null]);
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 3);
    for (reference, values) in [(0, [10.0, -10.0]), (1, [-20.0, 20.0]), (2, [1e200, 1e200])] {
        for (desired, r) in [8, 9].into_iter().zip(values) {
            rewards.entry_mut(reference, &goal(desired)).r = r;
        }
    }
    let matched = vec![0, 1, 2];
    for poison in [1e250, -1e250, f64::NAN] {
        for reference in 0..3 {
            population.get_mut(reference).r = poison;
            population.get_mut(reference).ir = poison;
        }
        for (desired, action, value) in [(8, 0, 5.0_f64), (9, 1, 20.0_f64)] {
            assert_eq!(
                rewards.best_action(
                    &population,
                    &matched,
                    &goal(desired),
                    2,
                    &mut ChaChaRandomSource::from_seed(42)
                ),
                action
            );
            assert_eq!(
                rewards
                    .bootstrap(&population, &matched, &goal(desired))
                    .to_bits(),
                value.to_bits()
            );
        }
    }
    rewards.entry_mut(1, &goal(8)).r = 8.0;
    assert_eq!(
        rewards.best_action(
            &population,
            &matched,
            &goal(8),
            2,
            &mut ChaChaRandomSource::from_seed(42)
        ),
        0
    );
    assert_eq!(rewards.bootstrap(&population, &matched, &goal(8)), 8.0);
}

fn alp_only(
    rewards: &mut PerGoalRewards<1>,
    population: &mut Population<3>,
    draw: &DrawnSample<1, 3>,
    config: &Configuration,
) {
    let matched = population.form_match_set(&draw.scored.sample.state);
    let mut sets = learning::AlpSets {
        action_set: population.form_action_set(&matched, draw.scored.sample.action),
        next_matched: population.form_match_set(&draw.scored.sample.next_state),
    };
    learning::apply_alp(
        rewards,
        population,
        &mut sets,
        &draw.scored.sample,
        10,
        config,
        &mut ChaChaRandomSource::from_seed(42),
    );
}

#[test]
fn general_and_covering_use_base_creation_values_even_when_initial_r_is_nonzero() {
    let mut config = config();
    config.initial_r = 7.0;
    config.initial_ir = 2.0;
    let general = PerGoalRewards::new(&[goal(8), goal(9)], &config, 1);
    assert_eq!(
        general.entry(0, &goal(8)),
        &GoalValue {
            r: 7.0,
            ir: 2.0,
            updated: false
        }
    );
    let mut population = Population::new();
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 0);
    alp_only(
        &mut rewards,
        &mut population,
        &sample(8, SampleOrigin::Original, false),
        &config,
    );
    assert_eq!(population.len(), 1);
    for desired in [goal(8), goal(9)] {
        assert_eq!(
            rewards.entry(0, &desired),
            &GoalValue {
                r: 0.0,
                ir: 2.0,
                updated: false
            }
        );
    }
}

#[test]
fn expected_and_unexpected_alp_children_copy_all_parent_values_and_reset_update_flags() {
    for expected in [true, false] {
        let config = config();
        let mut parent = rule(0, &config);
        if expected {
            parent.mark.attributes[2].insert(Symbol::Token(8));
        } else {
            parent.effect.symbols[0] = Symbol::Wildcard;
        }
        let mut population = Population::from_classifiers(vec![parent]);
        let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 1);
        *rewards.entry_mut(0, &goal(8)) = GoalValue {
            r: 23.0,
            ir: 17.0,
            updated: true,
        };
        *rewards.entry_mut(0, &goal(9)) = GoalValue {
            r: -4.0,
            ir: -9.0,
            updated: false,
        };
        alp_only(
            &mut rewards,
            &mut population,
            &sample(9, SampleOrigin::Original, false),
            &config,
        );
        assert_eq!(population.len(), 2);
        for desired in [goal(8), goal(9)] {
            let inherited = rewards.entry(0, &desired);
            assert_eq!(
                rewards.entry(1, &desired),
                &GoalValue {
                    updated: false,
                    ..*inherited
                }
            );
        }
    }
}

#[test]
fn removal_drops_victim_entries_and_remaps_retained_and_new_values_in_population_order() {
    let config = config();
    let mut victim = rule(0, &config);
    victim.q = 0.05;
    victim.effect.symbols[0] = Symbol::Wildcard;
    let retained = rule(1, &config);
    let mut population = Population::from_classifiers(vec![victim, retained]);
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 2);
    for reference in 0..2 {
        for desired in [8, 9] {
            rewards.entry_mut(reference, &goal(desired)).r =
                (100 * reference + desired as usize) as f64;
        }
    }
    rewards.entry_mut(0, &goal(8)).updated = true;
    alp_only(
        &mut rewards,
        &mut population,
        &sample(9, SampleOrigin::Original, false),
        &config,
    );
    assert_eq!(population.len(), 2);
    assert_eq!(population.get(0).action, Some(1));
    assert_eq!(population.get(1).action, Some(0));
    for desired in [8, 9] {
        assert_eq!(rewards.entry(0, &goal(desired)).r, 100.0 + desired as f64);
        assert_eq!(rewards.entry(1, &goal(desired)).r, desired as f64);
    }
    assert_eq!(rewards.updated_entries(), 0);
    assert_eq!(rewards.existing_pairs(), 4);
}

#[test]
fn subsumption_and_duplicate_children_keep_the_existing_owners_values() {
    let config = config();
    let mut parent = rule(0, &config);
    parent.mark.attributes[2].insert(Symbol::Token(8));
    let mut subsumer = rule(0, &config);
    subsumer.q = 0.99;
    subsumer.exp = 100;
    subsumer.condition.symbols[0] = Symbol::Wildcard;
    let mut population = Population::from_classifiers(vec![parent, subsumer]);
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 2);
    rewards.entry_mut(0, &goal(8)).r = 53.0;
    rewards.entry_mut(1, &goal(8)).r = 89.0;
    let before = rewards.clone();
    alp_only(
        &mut rewards,
        &mut population,
        &sample(9, SampleOrigin::Original, false),
        &config,
    );
    assert_eq!(population.len(), 2);
    assert_eq!(rewards, before);
    let mut duplicate_parent = rule(0, &config);
    duplicate_parent.effect.symbols[0] = Symbol::Wildcard;
    let mut population = Population::from_classifiers(vec![duplicate_parent]);
    let mut rewards = PerGoalRewards::new(&[goal(8), goal(9)], &config, 1);
    rewards.entry_mut(0, &goal(8)).r = 77.0;
    alp_only(
        &mut rewards,
        &mut population,
        &sample(9, SampleOrigin::Original, false),
        &config,
    );
    assert_eq!(population.len(), 2);
    assert_eq!(rewards.entry(1, &goal(8)).r, 77.0);
}

#[test]
fn both_new_arm_constructors_reject_ga_and_unregistered_goals_fail_before_learning() {
    for shortened in [false, true] {
        let mut config = config();
        config.do_ga = true;
        assert!(std::panic::catch_unwind(|| agent(shortened, true, config, &[goal(9)])).is_err());
        let mut actual = agent(shortened, true, self::config(), &[goal(9)]);
        let before = actual.learner().snapshot();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            actual
                .learner_mut()
                .learn_draw(&sample(8, SampleOrigin::Original, false), 1);
        }))
        .is_err());
        assert_eq!(actual.learner().snapshot(), before);
    }
}

#[test]
fn both_new_arms_ignore_shared_r_and_ir_in_learning_and_policy_after_every_draw_origin() {
    for shortened in [false, true] {
        let build = |poison| {
            let config = config();
            let mut actual = agent(shortened, true, config.clone(), &[goal(8), goal(9)]);
            let mut first = rule(0, &config);
            first.r = poison;
            first.ir = -poison;
            first.mark.attributes[2].insert(Symbol::Token(7));
            let mut next = rule(1, &config);
            next.condition.symbols[0] = Symbol::Token(1);
            next.q = 0.8;
            next.r = -poison;
            next.ir = poison;
            *actual.learner_mut() = crate::episodic_replay::EpisodicReplay::new(
                config,
                replay(),
                actual.learner().sampler().configuration(),
                actual.learner().copy_update(),
                ChaChaRandomSource::from_seed(42),
                ChaChaRandomSource::from_seed(43),
            )
            .with_population(Population::from_classifiers(vec![first, next]))
            .with_per_goal_rewards(&[goal(8), goal(9)]);
            let rewards = actual.learner_mut().per_goal_rewards_mut().unwrap();
            for reference in 0..2 {
                for desired in [8, 9] {
                    let entry = rewards.entry_mut(reference, &goal(desired));
                    entry.r = 10.0 + reference as f64 * desired as f64;
                    entry.ir = 3.0 * desired as f64;
                }
            }
            actual
        };
        let mut first = build(1e250);
        let mut second = build(-1e250);
        for (time, (desired, origin, fallback)) in [
            (8, SampleOrigin::Relabeled, false),
            (9, SampleOrigin::Original, true),
            (9, SampleOrigin::Original, false),
        ]
        .into_iter()
        .enumerate()
        {
            let draw = sample(desired, origin, fallback);
            first.learner_mut().learn_draw(&draw, time as u64);
            second.learner_mut().learn_draw(&draw, time as u64);
            assert_eq!(
                first.learner().per_goal_rewards(),
                second.learner().per_goal_rewards()
            );
            assert_eq!(
                first
                    .learner_mut()
                    .explore_action(&draw.scored.sample.state),
                second
                    .learner_mut()
                    .explore_action(&draw.scored.sample.state)
            );
            for desired in [goal(8), goal(9)] {
                assert_eq!(
                    materialized(
                        first.learner().population(),
                        first.learner().per_goal_rewards().unwrap(),
                        &desired
                    ),
                    materialized(
                        second.learner().population(),
                        second.learner().per_goal_rewards().unwrap(),
                        &desired
                    )
                );
            }
        }
    }
}
