use super::*;
use acs2_core::learning::{action_selection::EpsilonGreedy, rl::MaxFitnessBootstrap};
use acs2_core::ports::environment::StepOutcome;
use acs2_core::representation::{classifier::Classifier, symbol::Symbol};
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::LearningAgent;

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}
fn state(value: u8, desired: u8) -> Perception<3> {
    Perception::new([
        Symbol::Token(value),
        Symbol::Token(0),
        Symbol::Token(desired),
    ])
}
fn config() -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        initial_r: 0.73,
        initial_ir: 0.27,
        ..Configuration::default_protocol()
    }
}
struct Tiny {
    tick: u8,
    episode: u8,
}
impl Environment<3> for Tiny {
    fn reset(&mut self) -> Perception<3> {
        self.tick = 0;
        self.episode += 1;
        state(self.episode % 3, 9)
    }
    fn step(&mut self, action: usize) -> StepOutcome<3> {
        self.tick += 1;
        let ended = self.tick == 5;
        StepOutcome {
            observation: state((self.episode + self.tick + action as u8) % 3, 9),
            reward: f64::from(self.tick) * 11.0,
            terminated: ended && self.episode.is_multiple_of(2),
            truncated: ended && !self.episode.is_multiple_of(2),
            info: (),
        }
    }
}

#[test]
fn actual_online_driver_equals_acs2_after_every_episode_including_delayed_updates_and_both_truncations(
) {
    for mode in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        for seed in 42..47 {
            let config = config();
            let mut actual = Agent::<2, 1, 3, _>::new(
                config.clone(),
                &[goal(9)],
                ChaChaRandomSource::from_seed(seed),
            )
            .with_truncation_mode(mode);
            let mut base = crate::acs2::Agent::<3, _>::new(
                config.clone(),
                ChaChaRandomSource::from_seed(seed),
            )
            .with_truncation_mode(mode);
            let selector = EpsilonGreedy {
                number_of_possible_actions: 2,
                epsilon: config.epsilon,
            };
            let mut left = Tiny {
                tick: 0,
                episode: 0,
            };
            let mut right = Tiny {
                tick: 0,
                episode: 0,
            };
            for episode in 0..80 {
                let a = actual.train_episode(&mut left, episode * 5);
                let b = base.run_explore_trial(
                    &mut right,
                    &selector,
                    &MaxFitnessBootstrap,
                    episode * 5,
                );
                assert_eq!((a.steps, a.reward.to_bits()), (b.steps, b.reward.to_bits()));
                let rules: Vec<_> = actual
                    .population
                    .classifiers()
                    .iter()
                    .enumerate()
                    .map(|(i, c)| {
                        let mut c = c.clone();
                        let entry = actual.rewards.entry(i, &goal(9));
                        c.r = entry.r;
                        c.ir = entry.ir;
                        c
                    })
                    .collect();
                assert_eq!(
                    format!("{rules:?}"),
                    format!("{:?}", base.population().classifiers())
                );
                assert_eq!(actual.rng.capture_state(), base.capture().rng.into());
            }
            assert!(actual.rewards.updates().original > 0);
            assert_eq!(actual.rewards.updates().relabeled, 0);
        }
    }
}

#[test]
fn current_goal_only_is_updated_and_shared_values_cannot_affect_choices_or_bootstrap() {
    let config = config();
    let mut rules = Vec::new();
    for action in 0..2 {
        let mut c = Classifier::general(Some(action), &config);
        c.effect.symbols[0] = Symbol::Token(1);
        c.q = 0.8;
        c.r = f64::NAN;
        c.ir = f64::NEG_INFINITY;
        rules.push(c);
    }
    let mut agent = Agent::<2, 1, 3, _>::new(
        config.clone(),
        &[goal(8), goal(9)],
        ChaChaRandomSource::from_seed(42),
    )
    .with_population(Population::from_classifiers(rules));
    for (g, left, right, action, value) in [(8, 30.0, 2.0, 0, 24.0_f64), (9, 1.0, 40.0, 1, 32.0)] {
        agent.rewards.entry_mut(0, &goal(g)).r = left;
        agent.rewards.entry_mut(1, &goal(g)).r = right;
        let (a, v) = agent.greedy_action(&state(0, g), &mut ChaChaRandomSource::from_seed(43));
        assert_eq!(a, action);
        assert_eq!(v.to_bits(), value.to_bits());
        assert_eq!(
            agent
                .rewards
                .bootstrap(&agent.population, &[0, 1], &goal(g))
                .to_bits(),
            value.to_bits()
        );
    }
    let saved = agent.rewards.clone();
    agent.learn(
        &ReplaySample {
            state: state(0, 8),
            next_state: state(1, 8),
            action: 0,
            reward: 100.0,
            done: false,
        },
        vec![0],
        1,
    );
    assert_eq!(agent.population.len(), 2);
    assert_ne!(agent.rewards.entry(0, &goal(8)), saved.entry(0, &goal(8)));
    assert_eq!(agent.rewards.entry(0, &goal(9)), saved.entry(0, &goal(9)));
    assert_eq!(agent.rewards.entry(1, &goal(8)), saved.entry(1, &goal(8)));
    assert_eq!(agent.rewards.entry(1, &goal(9)), saved.entry(1, &goal(9)));
    assert!(agent.population.get(0).r.is_nan());
    assert_eq!(agent.population.get(0).ir, f64::NEG_INFINITY);
}

#[test]
fn covering_initializes_all_goals_and_removal_drops_the_corresponding_entries() {
    let config = config();
    let mut agent = Agent::<2, 1, 3, _>::new(
        config.clone(),
        &[goal(8), goal(9)],
        ChaChaRandomSource::from_seed(42),
    );
    agent.learn(
        &ReplaySample {
            state: state(0, 8),
            next_state: state(1, 8),
            action: 0,
            reward: 0.0,
            done: true,
        },
        vec![],
        1,
    );
    assert_eq!(agent.rewards.entry(0, &goal(8)).r, 0.0);
    assert_eq!(agent.rewards.entry(0, &goal(9)).r, 0.0);
    assert_eq!(agent.rewards.entry(0, &goal(9)).ir, config.initial_ir);
    assert_eq!(
        agent.rewards.entry(0, &goal(8)).ir,
        config.initial_ir + config.beta * (0.0 - config.initial_ir)
    );
    agent.population.get_mut(0).q = 0.01;
    agent.rewards.entry_mut(0, &goal(9)).r = 123456.0;
    agent.learn(
        &ReplaySample {
            state: state(0, 8),
            next_state: state(2, 8),
            action: 0,
            reward: 0.0,
            done: true,
        },
        vec![0],
        2,
    );
    assert_eq!(agent.rewards.existing_pairs(), agent.population.len() * 2);
    for index in 0..agent.population.len() {
        assert_ne!(agent.rewards.entry(index, &goal(9)).r, 123456.0);
    }
}

#[test]
fn genetic_algorithm_and_any_restore_are_rejected() {
    let mut config = config();
    config.do_ga = true;
    assert!(std::panic::catch_unwind(|| Agent::<2, 1, 3, _>::new(
        config,
        &[goal(9)],
        ChaChaRandomSource::from_seed(42)
    ))
    .is_err());
    let mut agent = Agent::<2, 1, 3, _>::new(
        self::config(),
        &[goal(9)],
        ChaChaRandomSource::from_seed(42),
    );
    let base = crate::acs2::Agent::<3, _>::new(self::config(), ChaChaRandomSource::from_seed(42));
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || agent.restore(base.capture())
    ))
    .is_err());
}

#[test]
fn actual_online_updates_inherit_every_goal_and_subsumption_keeps_the_subsumers_values() {
    for expected in [true, false] {
        let config = config();
        let mut parent = Classifier::<3>::general(Some(0), &config);
        parent.condition.symbols[0] = Symbol::Token(0);
        if expected {
            parent.effect.symbols[0] = Symbol::Token(1);
            parent.mark.attributes[2].insert(Symbol::Token(8));
        }
        let mut agent = Agent::<2, 1, 3, _>::new(
            config.clone(),
            &[goal(8), goal(9)],
            ChaChaRandomSource::from_seed(42),
        )
        .with_population(Population::from_classifiers(vec![parent]));
        assert_eq!(agent.rewards.entry(0, &goal(8)).r, config.initial_r);
        agent.rewards.entry_mut(0, &goal(8)).r = 53.0;
        agent.rewards.entry_mut(0, &goal(8)).ir = 17.0;
        agent.rewards.entry_mut(0, &goal(9)).r = 23.0;
        agent.rewards.entry_mut(0, &goal(9)).ir = 11.0;
        agent.learn(
            &ReplaySample {
                state: state(0, 9),
                next_state: state(1, 9),
                action: 0,
                reward: 100.0,
                done: true,
            },
            vec![0],
            10,
        );
        assert_eq!(agent.population.len(), 2);
        assert_eq!(agent.rewards.entry(1, &goal(8)).r, 53.0);
        assert_eq!(agent.rewards.entry(1, &goal(8)).ir, 17.0);
        assert_eq!(
            agent.rewards.entry(1, &goal(9)).r,
            23.0 + config.beta * (100.0 - 23.0)
        );
        assert_eq!(
            agent.rewards.entry(1, &goal(9)).ir,
            11.0 + config.beta * (100.0 - 11.0)
        );
    }
    let mut config = config();
    config.beta = 0.0;
    let mut parent = Classifier::<3>::general(Some(0), &config);
    parent.condition.symbols[0] = Symbol::Token(0);
    parent.effect.symbols[0] = Symbol::Token(1);
    parent.mark.attributes[2].insert(Symbol::Token(8));
    let mut subsumer = parent.clone();
    subsumer.condition.symbols[0] = Symbol::Wildcard;
    subsumer.mark = acs2_core::representation::mark::Mark::new();
    subsumer.q = 0.99;
    subsumer.exp = 100;
    let mut agent = Agent::<2, 1, 3, _>::new(
        config,
        &[goal(8), goal(9)],
        ChaChaRandomSource::from_seed(42),
    )
    .with_population(Population::from_classifiers(vec![parent, subsumer]));
    agent.rewards.entry_mut(0, &goal(8)).r = 53.0;
    agent.rewards.entry_mut(1, &goal(8)).r = 89.0;
    let before = agent.rewards.entry(1, &goal(8)).to_owned();
    agent.learn(
        &ReplaySample {
            state: state(0, 9),
            next_state: state(1, 9),
            action: 0,
            reward: 100.0,
            done: true,
        },
        vec![0, 1],
        10,
    );
    assert_eq!(agent.population.len(), 2);
    assert_eq!(agent.rewards.entry(1, &goal(8)), &before);
}
