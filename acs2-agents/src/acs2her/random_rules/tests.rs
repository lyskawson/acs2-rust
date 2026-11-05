use super::*;
use crate::acs2her::{Agent, Preset};
use crate::episodic_replay::{EpisodicAgent, EpisodicConfiguration};
use acs2_core::learning::replay::replay_learning_step;
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::ports::goal::{ExactMatch, Goal, GoalOutcome, GoalStart, GoalStep};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::symbol::Symbol;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::TruncationMode;
use acs2_trajectory::{DrawnSample, Episode, EpisodeId, GoalStrategy, ObjectiveEvaluator, Provenance, Sampler, ScoredSample, TrajectoryStore};

fn config(ga: bool) -> Configuration {
    Configuration {
        do_ga: ga, do_subsumption: false, theta_ga: 0, mu: 0.0, chi: 0.0,
        number_of_possible_actions: 2, ..Configuration::default_protocol()
    }
}

fn sample() -> ReplaySample<2> {
    ReplaySample {
        state: Perception::new([Symbol::Token(0), Symbol::Token(9)]), action: 0,
        next_state: Perception::new([Symbol::Token(1), Symbol::Token(9)]), reward: 1000.0, done: false,
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

fn draw(origin: SampleOrigin, fallback: bool) -> DrawnSample<1, 2> {
    DrawnSample {
        episode: EpisodeId(0), transition: 0, goal: Goal::new([Symbol::Token(9)]),
        scored: ScoredSample { sample: sample(), outcome: GoalOutcome { reward: 1000.0, terminated: false, truncated: false }, cost: Default::default() },
        provenance: Provenance { origin, fallback, strategy: GoalStrategy::Future, requested_strategy: GoalStrategy::Future,
            already_reached: false, after_counterfactual_end: false, done: false, outside_candidates: false },
        cost: Default::default(),
    }
}

#[test]
fn zero_skip_matches_full_steps_with_ga_and_policy_random_bits() {
    for ga in [false, true] {
        for seed in 42..47 {
            let config = config(ga);
            let mut actual = population(&config);
            let mut expected = population(&config);
            let mut policy = ChaChaRandomSource::from_seed_and_stream(seed, 1);
            let mut oracle = ChaChaRandomSource::from_seed_and_stream(seed, 1);
            let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: 0.0 }, ChaChaRandomSource::from_seed_and_stream(seed, SKIP_STREAM));
            for time in 1..30 {
                let mut sample = sample();
                sample.done = time % 3 == 0;
                replay_learning_step(&mut expected, &config, &MaxFitnessBootstrap, &sample, time, &mut oracle);
                updates.learn::<1, 2>(&mut actual, &config, &sample, time, &mut policy, SampleOrigin::Relabeled);
                assert_eq!(format!("{:?}", actual.classifiers()), format!("{:?}", expected.classifiers()));
                assert_eq!(policy.capture_state(), oracle.capture_state());
            }
            assert_eq!(updates.counts.relabeled.skipped_with_goal, 0);
            assert_eq!(updates.counts.relabeled.updates.skipped_without_goal, 0);
        }
    }
}

#[test]
fn all_skipped_preserves_rewards_and_runs_alp_ga_on_both_kinds() {
    let config = config(true);
    let mut actual = population(&config);
    let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: 1.0 }, ChaChaRandomSource::from_seed(42));
    updates.learn::<1, 2>(&mut actual, &config, &sample(), 10, &mut ChaChaRandomSource::from_seed(43), SampleOrigin::Relabeled);
    for (index, reward) in [(0, 20.0_f64), (1, 40.0_f64)] {
        let rule = actual.get(index);
        assert_eq!(rule.r.to_bits(), reward.to_bits());
        assert_eq!(rule.ir.to_bits(), 3.0_f64.to_bits());
        assert_eq!(rule.exp, 2);
        assert!(rule.q > 0.5);
        assert_eq!(rule.talp, Some(10));
        assert_eq!(rule.tga, 10);
    }
    assert_eq!(updates.counts.relabeled.skipped_with_goal, 1);
    assert_eq!(updates.counts.relabeled.updates.skipped_without_goal, 1);
    assert_eq!(updates.counts.relabeled.updates.draws_with_goal, 1);
}

#[test]
fn unequal_rule_kinds_receive_independent_coins_with_the_same_probability() {
    let config = config(false);
    let source = population(&config);
    let mut actual = Population::from_classifiers(vec![source.get(0).clone(), source.get(1).clone(), source.get(0).clone(), source.get(0).clone(), source.get(2).clone()]);
    let probability = 0.6321070402944451;
    let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: probability }, ChaChaRandomSource::from_seed_and_stream(42, SKIP_STREAM));
    let mut expected = ChaChaRandomSource::from_seed_and_stream(42, SKIP_STREAM);
    let mut policy = ChaChaRandomSource::from_seed(43);
    let mut skipped = [0_u64; 4];
    for time in 1..=10_000 {
        let coins = std::array::from_fn::<_, 4, _>(|_| expected.gen_bool(probability));
        for (total, coin) in skipped.iter_mut().zip(coins) { *total += u64::from(coin); }
        updates.learn::<1, 2>(&mut actual, &config, &sample(), time, &mut policy, SampleOrigin::Relabeled);
    }
    let route = updates.counts.relabeled;
    assert_eq!(route.skipped_with_goal, skipped[1]);
    assert_eq!(route.updates.skipped_without_goal, skipped[0] + skipped[2] + skipped[3]);
    assert_eq!(updates.rng.capture_state(), expected.capture_state());
    assert_eq!(route.updates.action_set_rules, 40_000);
    assert_eq!(route.updates.draws_with_goal, 10_000);
    for share in [route.skipped_with_goal as f64 / 10_000.0, route.updates.skipped_without_goal as f64 / 30_000.0] {
        assert!((share - probability).abs() < 0.01);
    }
}

#[test]
fn original_and_fallback_update_every_rule_without_consuming_a_skip_coin() {
    for fallback in [false, true] {
        let config = config(false);
        let replay = EpisodicConfiguration { capacity: 100, warmup: 1, updates_per_step: 3, truncation: TruncationMode::Bootstrap };
        let mut agent = Agent::<1, 1, 2, _, _>::new(config.clone(), replay, Preset::Acs2HerShortenedRandomRules,
            ChaChaRandomSource::from_seed(42), ChaChaRandomSource::from_seed(43));
        *agent.learner_mut() = crate::episodic_replay::EpisodicReplay::new(
            config.clone(), replay, agent.learner().sampler().configuration(), crate::episodic_replay::CopyUpdate::RandomRules,
            ChaChaRandomSource::from_seed(42), ChaChaRandomSource::from_seed(43),
        ).with_population(population(&config));
        agent.learner_mut().set_random_rule_updates(RandomRuleConfiguration { skip_probability: 1.0 }, ChaChaRandomSource::from_seed(44));
        let before = agent.learner().random_rule_updates().unwrap().rng.capture_state();
        agent.learner_mut().learn_draw(&draw(SampleOrigin::Original, fallback), 10);
        let state = agent.learner().random_rule_updates().unwrap();
        assert_eq!(state.rng.capture_state(), before);
        assert_eq!(state.counts.original.updates.without_goal, 1);
        assert_eq!(state.counts.original.updates.with_goal, 1);
        assert_eq!(state.counts.original.skipped_with_goal, 0);
        assert_eq!(state.counts.original.updates.skipped_without_goal, 0);
        assert_eq!(state.counts.relabeled, RandomRouteCounts::default());
        for index in [0, 1] {
            assert!(agent.learner().population().get(index).r > 40.0);
            assert!(agent.learner().population().get(index).ir > 3.0);
        }
    }
}

#[test]
fn zero_skip_reproduces_shortened_sampler_draws_and_population_including_fallbacks() {
    let config = config(false);
    let replay = EpisodicConfiguration { capacity: 100, warmup: 1, updates_per_step: 3, truncation: TruncationMode::Bootstrap };
    let mut oracle = Agent::<1, 1, 2, _, _>::new(config.clone(), replay, Preset::Acs2HerShortened { proportion: 2.0 / 3.0 }, ChaChaRandomSource::from_seed(42), ChaChaRandomSource::from_seed(43));
    let mut actual = Agent::<1, 1, 2, _, _>::new(config, replay, Preset::Acs2HerShortenedRandomRules, ChaChaRandomSource::from_seed(42), ChaChaRandomSource::from_seed(43));
    actual.learner_mut().set_random_rule_updates(RandomRuleConfiguration { skip_probability: 0.0 }, ChaChaRandomSource::from_seed(44));
    let goal = |value| Goal::new([Symbol::Token(value)]);
    let mut episode = Episode::new(GoalStart { observation: Perception::new([Symbol::Token(0)]), achieved: goal(0), desired: goal(9) });
    for (index, value) in [0, 1, 0, 2, 2].iter().enumerate() {
        episode.push(0, GoalStep { observation: Perception::new([Symbol::Token(*value)]), achieved: goal(*value), terminal_state: false, time_limit_reached: index == 4 });
    }
    let mut store = TrajectoryStore::new(100);
    store.insert(episode).unwrap();
    let mut left = Sampler::new(actual.learner().sampler().configuration(), ChaChaRandomSource::from_seed(43));
    let mut right = Sampler::new(oracle.learner().sampler().configuration(), ChaChaRandomSource::from_seed(43));
    let evaluator = ObjectiveEvaluator(&ExactMatch { reward_on_reach: 1000.0 });
    let draws = left.draw::<1, 1, 2>(&store, 300, &evaluator).unwrap();
    let expected = right.draw::<1, 1, 2>(&store, 300, &evaluator).unwrap();
    assert_eq!(format!("{draws:?}"), format!("{expected:?}"));
    assert_eq!(left.random_source().capture_state(), right.random_source().capture_state());
    assert!(draws.iter().any(|draw| draw.provenance.fallback));
    assert!(draws.iter().any(|draw| draw.provenance.origin == SampleOrigin::Relabeled));
    for (draw, expected) in draws.iter().zip(expected) {
        actual.learner_mut().learn_draw(draw, 5);
        oracle.learner_mut().learn_draw(&expected, 5);
        assert_eq!(format!("{:?}", actual.learner().population().classifiers()), format!("{:?}", oracle.learner().population().classifiers()));
        assert_eq!(actual.learner_mut().explore_action(&sample().state), oracle.learner_mut().explore_action(&sample().state));
    }
}

#[test]
fn unfiltered_bootstrap_includes_a_blind_next_state_rule() {
    let config = config(false);
    let mut actual = population(&config);
    let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: 0.0 }, ChaChaRandomSource::from_seed(42));
    updates.learn::<1, 2>(&mut actual, &config, &sample(), 10, &mut ChaChaRandomSource::from_seed(43), SampleOrigin::Relabeled);
    assert_eq!(actual.get(1).r.to_bits(), (40.0_f64 + config.beta * (1000.0 + config.gamma * 400.0 - 40.0)).to_bits());
}

#[test]
fn invalid_skip_probabilities_are_rejected() {
    for value in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(std::panic::catch_unwind(|| RandomRuleConfiguration { skip_probability: value }.validate()).is_err());
    }
}

#[test]
fn skip_snapshot_includes_the_injected_random_position() {
    let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: 0.63 }, ChaChaRandomSource::from_seed_and_stream(42, SKIP_STREAM));
    let before = updates.snapshot();
    updates.rng.gen_bool(0.63);
    assert_ne!(updates.snapshot(), before);
}

#[test]
fn reward_skips_include_goal_rules_created_by_alp_before_the_reward_step() {
    let config = config(false);
    let mut blind = population(&config).get(0).clone();
    blind.mark.attributes[1].insert(Symbol::Token(8));
    let mut population = Population::from_classifiers(vec![blind]);
    let mut updates = RandomRuleUpdates::new(RandomRuleConfiguration { skip_probability: 1.0 }, ChaChaRandomSource::from_seed(42));
    updates.learn::<1, 2>(&mut population, &config, &sample(), 10, &mut ChaChaRandomSource::from_seed(43), SampleOrigin::Relabeled);
    assert_eq!(population.len(), 2);
    assert!(super::super::goal_rules::specifies_goal::<1, 2>(population.get(1)));
    assert_eq!(updates.counts.relabeled.skipped_with_goal, 1);
    assert_eq!(updates.counts.relabeled.updates.skipped_without_goal, 1);
    assert_eq!(updates.counts.relabeled.updates.draws_with_goal, 1);
}
