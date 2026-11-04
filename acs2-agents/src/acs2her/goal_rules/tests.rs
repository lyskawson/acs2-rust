use super::*;
use crate::acs2her::{Agent, Preset};
use crate::episodic_replay::{EpisodicAgent, EpisodicConfiguration};
use acs2_core::learning::replay::replay_learning_step;
use acs2_core::ports::goal::{Goal, GoalOutcome};
use acs2_core::representation::perception::Perception;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::TruncationMode;
use acs2_trajectory::{DrawnSample, EpisodeId, GoalStrategy, Provenance, ScoredSample};

fn config(ga: bool) -> Configuration {
    Configuration {
        do_ga: ga,
        do_subsumption: false,
        theta_ga: 0,
        mu: 0.0,
        chi: 0.0,
        number_of_possible_actions: 2,
        ..Configuration::default_protocol()
    }
}

fn sample() -> ReplaySample<2> {
    ReplaySample {
        state: Perception::new([Symbol::Token(0), Symbol::Token(9)]),
        action: 0,
        next_state: Perception::new([Symbol::Token(1), Symbol::Token(9)]),
        reward: 1000.0,
        done: false,
    }
}

fn population(config: &Configuration) -> Population<2> {
    let mut blind = Classifier::general(Some(0), config);
    blind.condition.symbols[0] = Symbol::Token(0);
    blind.effect.symbols[0] = Symbol::Token(1);
    blind.r = 20.0;
    blind.ir = 3.0;
    let mut goal = blind.clone();
    goal.condition.symbols[1] = Symbol::Token(9);
    goal.r = 40.0;
    let mut next = blind.clone();
    next.action = Some(1);
    next.condition.symbols[0] = Symbol::Token(1);
    next.q = 0.8;
    next.r = 500.0;
    Population::from_classifiers(vec![blind, goal, next])
}

#[test]
fn keep_all_reproduces_full_learning_population_and_random_bits() {
    for ga in [false, true] {
        for seed in 42..47 {
            let config = config(ga);
            let mut actual = population(&config);
            let mut expected = population(&config);
            let mut rng = ChaChaRandomSource::from_seed(seed);
            let mut oracle_rng = ChaChaRandomSource::from_seed(seed);
            for time in 1..30 {
                let mut draw = sample();
                draw.done = time % 3 == 0;
                replay_learning_step(
                    &mut expected,
                    &config,
                    &MaxFitnessBootstrap,
                    &draw,
                    time,
                    &mut oracle_rng,
                );
                learning_step::<1, 2>(&mut actual, &config, &draw, time, &mut rng, |_| true);
                assert_eq!(
                    format!("{:?}", actual.classifiers()),
                    format!("{:?}", expected.classifiers())
                );
                assert_eq!(rng.capture_state(), oracle_rng.capture_state());
            }
        }
    }
}

fn draw(origin: SampleOrigin, fallback: bool) -> DrawnSample<1, 2> {
    DrawnSample {
        episode: EpisodeId(0),
        transition: 0,
        goal: Goal::new([Symbol::Token(9)]),
        scored: ScoredSample {
            sample: sample(),
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

fn agent(ga: bool) -> Agent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource> {
    let config = config(ga);
    let mut agent = Agent::new(
        config.clone(),
        EpisodicConfiguration {
            capacity: 100,
            warmup: 1,
            updates_per_step: 3,
            truncation: TruncationMode::Bootstrap,
        },
        Preset::Acs2HerShortenedGoalRules,
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    );
    *agent.learner_mut() = crate::episodic_replay::EpisodicReplay::new(
        config.clone(),
        agent.learner().replay_configuration(),
        agent.learner().sampler().configuration(),
        crate::episodic_replay::CopyUpdate::GoalRules,
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    )
    .with_population(population(&config));
    agent
}

#[test]
fn relabeled_rewards_filter_goal_rules_while_alp_and_ga_reach_both_kinds() {
    let mut agent = agent(true);
    agent
        .learner_mut()
        .learn_draw(&draw(SampleOrigin::Relabeled, false), 10);
    let rules = agent.learner().population().classifiers();
    assert_eq!(rules[0].r.to_bits(), 20.0_f64.to_bits());
    assert_eq!(rules[0].ir.to_bits(), 3.0_f64.to_bits());
    assert_eq!(
        rules[1].r.to_bits(),
        (40.0_f64 + 0.05 * (1000.0 + 0.95 * 400.0 - 40.0)).to_bits()
    );
    assert_eq!(
        rules[1].ir.to_bits(),
        (3.0_f64 + 0.05 * (1000.0 - 3.0)).to_bits()
    );
    for rule in &rules[..2] {
        assert_eq!(rule.exp, 2);
        assert!(rule.q > 0.5);
        assert_eq!(rule.talp, Some(10));
        assert_eq!(rule.tga, 10);
    }
    let counts = agent.learner().reward_update_counts().relabeled;
    assert_eq!(
        counts,
        RouteCounts {
            draws: 1,
            with_goal: 1,
            without_goal: 0,
            skipped_without_goal: 1,
            draws_with_goal: 1,
            action_set_rules: 2
        }
    );
}

#[test]
fn original_and_fallback_draws_update_both_kinds() {
    for fallback in [false, true] {
        let mut agent = agent(false);
        agent
            .learner_mut()
            .learn_draw(&draw(SampleOrigin::Original, fallback), 10);
        let rules = agent.learner().population().classifiers();
        assert_ne!(rules[0].r.to_bits(), 20.0_f64.to_bits());
        assert_ne!(rules[1].r.to_bits(), 40.0_f64.to_bits());
        assert!(rules[..2].iter().all(|rule| rule.ir > 3.0));
        let counts = agent.learner().reward_update_counts();
        assert_eq!(counts.original.with_goal, 1);
        assert_eq!(counts.original.without_goal, 1);
        assert_eq!(counts.original.skipped_without_goal, 0);
        assert_eq!(counts.relabeled, RouteCounts::default());
    }
}

#[test]
fn filter_and_counters_include_goal_specialization_inserted_by_alp() {
    let config = config(false);
    let mut blind = population(&config).get(0).clone();
    blind.mark.attributes[1].insert(Symbol::Token(8));
    let mut rules = Population::from_classifiers(vec![blind]);
    let counts = learning_step::<1, 2>(
        &mut rules,
        &config,
        &sample(),
        10,
        &mut ChaChaRandomSource::from_seed(42),
        specifies_goal::<1, 2>,
    );
    assert_eq!(rules.len(), 2);
    assert!(!specifies_goal::<1, 2>(rules.get(0)));
    assert!(specifies_goal::<1, 2>(rules.get(1)));
    assert_eq!(rules.get(0).r.to_bits(), 20.0_f64.to_bits());
    assert!(rules.get(1).r > 20.0);
    assert_eq!(counts.with_goal, 1);
    assert_eq!(counts.skipped_without_goal, 1);
    assert_eq!(counts.action_set_rules, 2);
}

#[test]
fn keep_all_replays_the_actual_shortened_sampler_including_fallbacks_bit_for_bit() {
    use acs2_core::ports::goal::{ExactMatch, GoalStart, GoalStep};
    use acs2_trajectory::{Episode, ObjectiveEvaluator, Sampler, TrajectoryStore};
    let config = config(false);
    let replay = EpisodicConfiguration {
        capacity: 100,
        warmup: 1,
        updates_per_step: 3,
        truncation: TruncationMode::Bootstrap,
    };
    let mut oracle = Agent::<1, 1, 2, _, _>::new(
        config.clone(),
        replay,
        Preset::Acs2HerShortened {
            proportion: 2.0 / 3.0,
        },
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    );
    let goal = |value| Goal::new([Symbol::Token(value)]);
    let mut episode = Episode::new(GoalStart {
        observation: Perception::new([Symbol::Token(0)]),
        achieved: goal(0),
        desired: goal(9),
    });
    for (index, value) in [0, 1, 0, 2, 2].iter().enumerate() {
        episode.push(
            0,
            GoalStep {
                observation: Perception::new([Symbol::Token(*value)]),
                achieved: goal(*value),
                terminal_state: false,
                time_limit_reached: index == 4,
            },
        );
    }
    let mut store = TrajectoryStore::new(100);
    store.insert(episode).unwrap();
    let mut sampler = Sampler::new(
        oracle.learner().sampler().configuration(),
        ChaChaRandomSource::from_seed(43),
    );
    let draws = sampler
        .draw::<1, 1, 2>(
            &store,
            300,
            &ObjectiveEvaluator(&ExactMatch {
                reward_on_reach: 1000.0,
            }),
        )
        .unwrap();
    assert!(draws.iter().any(|draw| draw.provenance.fallback));
    assert!(draws
        .iter()
        .any(|draw| draw.provenance.origin == SampleOrigin::Relabeled));
    let mut actual = Population::new();
    let mut rng = ChaChaRandomSource::from_seed(42);
    for draw in &draws {
        oracle.learner_mut().learn_draw(draw, 5);
        learning_step::<1, 2>(
            &mut actual,
            &config,
            &draw.scored.sample,
            5,
            &mut rng,
            |_| true,
        );
        assert_eq!(
            format!("{:?}", actual.classifiers()),
            format!("{:?}", oracle.learner().population().classifiers())
        );
    }
    assert!(oracle
        .learner()
        .snapshot()
        .contains(&format!("{:?}", rng.capture_state())));
}
