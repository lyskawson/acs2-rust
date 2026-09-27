use std::cell::{Cell, RefCell};

use acs2_core::acs2er::{Acs2ErAgent, ReplayConfiguration};
use acs2_core::action_selection::ActionSelector;
use acs2_core::agent::Agent;
use acs2_core::checkpoint::Checkpointed;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::population::{ClassifierRef, Population};
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource, RngState};
use acs2_core::symbol::Symbol;
use acs2_core::trial::{self, LearningAgent, TruncationMode};

fn state(value: u8) -> Perception<2> {
    Perception::new([Symbol::Token(value), Symbol::Token(0)])
}

fn config() -> Configuration {
    Configuration {
        number_of_possible_actions: 1,
        beta: 0.5,
        gamma: 0.75,
        ..Configuration::default_protocol()
    }
}

fn population() -> Population<2> {
    let classifier = |condition, effect, reward| {
        let mut cl = Classifier::general(Some(0), &config());
        cl.condition.set(0, Symbol::Token(condition));
        cl.effect.set(0, effect);
        cl.q = 0.8;
        cl.r = reward;
        cl.ir = 2.0;
        cl.exp = 4;
        cl
    };
    Population::from_classifiers(vec![
        classifier(0, Symbol::Token(1), 8.0),
        classifier(1, Symbol::Token(2), 40.0),
        classifier(1, Symbol::Wildcard, 1_000.0),
        classifier(2, Symbol::Token(3), 2_000.0),
    ])
}

struct Transition {
    start: u8,
    terminated: bool,
    truncated: bool,
    reward: f64,
    steps: u32,
}

impl Transition {
    fn new(terminated: bool, truncated: bool, reward: f64) -> Self {
        Self {
            start: 0,
            terminated,
            truncated,
            reward,
            steps: 0,
        }
    }
}

impl Environment<2> for Transition {
    fn reset(&mut self) -> Perception<2> {
        self.steps = 0;
        state(self.start)
    }

    fn step(&mut self, action: usize) -> StepOutcome<2> {
        assert_eq!(action, 0);
        self.steps += 1;
        assert_eq!(self.steps, 1, "an ended episode must not take another step");
        StepOutcome {
            observation: state(self.start + 1),
            reward: self.reward,
            terminated: self.terminated,
            truncated: self.truncated,
            info: (),
        }
    }
}

struct FirstRandom;

impl RandomSource for FirstRandom {
    fn gen_bool(&mut self, _probability: f64) -> bool {
        false
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        assert!(bound > 0);
        0
    }
    fn gen_unit(&mut self) -> f64 {
        0.0
    }
}

struct FirstAction;

impl ActionSelector<2> for FirstAction {
    fn select(
        &self,
        _population: &Population<2>,
        _set: &[ClassifierRef],
        _rng: &mut dyn RandomSource,
    ) -> usize {
        0
    }
}

#[derive(Clone, Copy, Debug)]
enum Path {
    Explore,
    Exploit,
    Replay,
    ReplayExploit,
}

fn assert_prediction(pop: &Population<2>, expected: f64, reward: f64, learned: bool, repeats: u32) {
    let acting = pop.get(0);
    let mut r = 8.0;
    let mut ir = 2.0;
    for _ in 0..repeats {
        r += 0.5 * (expected - r);
        ir += 0.5 * (reward - ir);
    }
    assert_eq!(acting.r.to_bits(), r.to_bits(), "reward prediction");
    assert_eq!(
        acting.ir.to_bits(),
        ir.to_bits(),
        "immediate reward prediction"
    );
    assert_eq!(acting.exp, 4 + if learned { repeats } else { 0 });
    assert_eq!(
        pop.get(1).r,
        40.0,
        "the next-state classifier is not updated"
    );
}

fn check(path: Path, terminated: bool, truncated: bool, reward: f64) {
    for mode in [
        None,
        Some(TruncationMode::Bootstrap),
        Some(TruncationMode::Pyalcs),
    ] {
        let terminal = terminated || mode == Some(TruncationMode::Pyalcs);
        let target = reward + if terminal { 0.0 } else { 0.75 * 32.0 };
        let mut env = Transition::new(terminated, truncated, reward);
        match path {
            Path::Explore | Path::Exploit => {
                let mut agent = Agent::with_population(config(), FirstRandom, population());
                if let Some(mode) = mode {
                    agent = agent.with_truncation_mode(mode);
                }
                let metrics = match path {
                    Path::Explore => {
                        agent.run_explore_trial(&mut env, &FirstAction, &MaxFitnessBootstrap, 100)
                    }
                    _ => agent.run_exploit_trial(&mut env, &MaxFitnessBootstrap, 100),
                };
                assert_eq!(metrics.steps, 1, "{path:?} {mode:?}");
                assert_eq!(metrics.reward, reward);
                assert_prediction(
                    agent.population(),
                    target,
                    reward,
                    matches!(path, Path::Explore),
                    1,
                );
            }
            Path::Replay | Path::ReplayExploit => {
                let replay = ReplayConfiguration {
                    buffer_size: 4,
                    min_samples: 2,
                    samples_number: 1,
                };
                let mut agent =
                    Acs2ErAgent::with_population(config(), replay, FirstRandom, population());
                if let Some(mode) = mode {
                    agent = agent.with_truncation_mode(mode);
                }
                if matches!(path, Path::ReplayExploit) {
                    let metrics = agent.run_exploit_trial(&mut env, &MaxFitnessBootstrap, 100);
                    assert_eq!(metrics.steps, 1);
                    assert_eq!(metrics.reward, reward);
                    assert!(agent.replay_memory().is_empty());
                    assert_prediction(agent.population(), target, reward, false, 1);
                } else {
                    let metrics =
                        agent.run_explore_trial(&mut env, &FirstAction, &MaxFitnessBootstrap, 100);
                    assert_eq!(metrics.steps, 1);
                    assert_eq!(metrics.reward, reward);
                    assert_eq!(agent.population().get(0).r, 8.0, "warmup does not learn");
                    let stored = agent.replay_memory().get(0);
                    assert_eq!(stored.done, terminal);
                    assert_eq!(stored.state, state(0));
                    assert_eq!(stored.next_state, state(1));
                    assert_eq!(stored.reward, reward);
                    let mut later = Transition {
                        start: 2,
                        ..Transition::new(true, false, 7.0)
                    };
                    for repeat in 1..=2 {
                        let metrics = agent.run_explore_trial(
                            &mut later,
                            &FirstAction,
                            &MaxFitnessBootstrap,
                            100 + repeat as u64,
                        );
                        assert_eq!(metrics.steps, 1);
                        assert_eq!(agent.replay_memory().get(0), stored);
                        assert_prediction(agent.population(), target, reward, true, repeat);
                    }
                }
            }
        }
    }
}

macro_rules! cases {
    ($truncated:ident, $terminated:ident, $both:ident, $success:ident, $path:expr) => {
        #[test]
        fn $truncated() {
            check($path, false, true, 4.0);
        }
        #[test]
        fn $terminated() {
            check($path, true, false, 4.0);
        }
        #[test]
        fn $both() {
            check($path, true, true, 4.0);
        }
        #[test]
        fn $success() {
            check($path, true, false, 1_000.0);
        }
    };
}

cases!(
    exploration_truncation,
    exploration_termination,
    exploration_both_flags,
    exploration_limit_success,
    Path::Explore
);
cases!(
    exploitation_truncation,
    exploitation_termination,
    exploitation_both_flags,
    exploitation_limit_success,
    Path::Exploit
);
cases!(
    replay_truncation,
    replay_termination,
    replay_both_flags,
    replay_limit_success,
    Path::Replay
);
cases!(
    replay_exploitation_truncation,
    replay_exploitation_termination,
    replay_exploitation_both_flags,
    replay_exploitation_limit_success,
    Path::ReplayExploit
);

struct TwoSteps {
    steps: u32,
}

impl Environment<2> for TwoSteps {
    fn reset(&mut self) -> Perception<2> {
        self.steps = 0;
        state(0)
    }
    fn step(&mut self, _action: usize) -> StepOutcome<2> {
        self.steps += 1;
        assert!(self.steps <= 2);
        StepOutcome {
            observation: state(self.steps as u8),
            reward: if self.steps == 1 { 4.0 } else { 0.0 },
            terminated: self.steps == 2,
            truncated: false,
            info: (),
        }
    }
}

#[derive(Default)]
struct InspectAction {
    calls: Cell<u32>,
    snapshot: RefCell<Option<(String, Option<RngState>)>>,
}

impl ActionSelector<2> for InspectAction {
    fn select(
        &self,
        pop: &Population<2>,
        _set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        let call = self.calls.get();
        self.calls.set(call + 1);
        if call == 1 {
            *self.snapshot.borrow_mut() =
                Some((format!("{:?}", pop.classifiers()), rng.capture_state()));
        }
        0
    }
}

#[test]
fn truncated_exploration_matches_a_continuing_transition_including_alp_ga_and_rng() {
    for do_ga in [false, true] {
        let cfg = Configuration {
            do_ga,
            theta_ga: 0,
            ..config()
        };
        let mut capped =
            Agent::with_population(cfg.clone(), ChaChaRandomSource::from_seed(42), population());
        let mut continuing =
            Agent::with_population(cfg, ChaChaRandomSource::from_seed(42), population());
        capped.run_explore_trial(
            &mut Transition::new(false, true, 4.0),
            &FirstAction,
            &MaxFitnessBootstrap,
            100,
        );
        let inspect = InspectAction::default();
        continuing.run_explore_trial(
            &mut TwoSteps { steps: 0 },
            &inspect,
            &MaxFitnessBootstrap,
            100,
        );
        let (classifiers, rng) = inspect.snapshot.into_inner().unwrap();
        assert_eq!(
            format!("{:?}", capped.population().classifiers()),
            classifiers
        );
        assert_eq!(Some(capped.capture().rng), rng);
        if do_ga {
            assert_eq!(capped.population().get(0).tga, 101);
        }
    }
}

#[test]
fn truncated_exploration_reforms_the_next_match_set_after_alp() {
    let mut general = Classifier::general(Some(0), &config());
    general.effect.set(0, Symbol::Token(1));
    general.q = 0.6;
    general.r = 40.0;
    let mut inadequate = general.clone();
    inadequate.effect.set(0, Symbol::Token(2));
    inadequate.q = 0.05;
    inadequate.r = 10_000.0;
    let mut agent = Agent::with_population(
        config(),
        FirstRandom,
        Population::from_classifiers(vec![inadequate, general]),
    );
    agent.run_explore_trial(
        &mut Transition::new(false, true, 4.0),
        &FirstAction,
        &MaxFitnessBootstrap,
        100,
    );
    assert_eq!(agent.population().len(), 1);
    let cl = agent.population().get(0);
    assert_eq!(cl.q, 0.8);
    assert_eq!(
        cl.r, 34.0,
        "bootstrap uses the updated quality and excludes the deleted classifier"
    );
}

#[test]
fn the_free_exploitation_function_uses_the_corrected_default() {
    let mut pop = population();
    let metrics = trial::run_exploit_trial(
        &mut pop,
        &config(),
        &mut FirstRandom,
        &mut Transition::new(false, true, 4.0),
        &MaxFitnessBootstrap,
    );
    assert_eq!(metrics.steps, 1);
    assert_prediction(&pop, 28.0, 4.0, false, 1);
}

#[test]
fn restore_preserves_the_mode_selected_by_the_caller() {
    for mode in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        let source =
            Agent::with_population(config(), ChaChaRandomSource::from_seed(42), population());
        let mut online =
            Agent::new(config(), ChaChaRandomSource::from_seed(1)).with_truncation_mode(mode);
        online.restore(source.capture());
        online.run_exploit_trial(
            &mut Transition::new(false, true, 4.0),
            &MaxFitnessBootstrap,
            100,
        );
        let replay = ReplayConfiguration {
            buffer_size: 4,
            min_samples: 1,
            samples_number: 1,
        };
        let source = Acs2ErAgent::with_population(
            config(),
            replay,
            ChaChaRandomSource::from_seed(42),
            population(),
        );
        let mut er = Acs2ErAgent::new(config(), replay, ChaChaRandomSource::from_seed(1))
            .with_truncation_mode(mode);
        er.restore(source.capture());
        er.run_explore_trial(
            &mut Transition::new(false, true, 4.0),
            &FirstAction,
            &MaxFitnessBootstrap,
            100,
        );
        let target = if mode == TruncationMode::Pyalcs {
            4.0
        } else {
            28.0
        };
        assert_prediction(online.population(), target, 4.0, false, 1);
        assert_prediction(er.population(), target, 4.0, true, 1);
    }
}

#[test]
fn new_agents_use_the_corrected_default_after_restore() {
    let source = Agent::with_population(config(), ChaChaRandomSource::from_seed(42), population());
    let mut online = Agent::new(config(), ChaChaRandomSource::from_seed(1));
    online.restore(source.capture());
    online.run_exploit_trial(
        &mut Transition::new(false, true, 4.0),
        &MaxFitnessBootstrap,
        100,
    );
    assert_prediction(online.population(), 28.0, 4.0, false, 1);
    let replay = ReplayConfiguration {
        buffer_size: 4,
        min_samples: 1,
        samples_number: 1,
    };
    let source = Acs2ErAgent::with_population(
        config(),
        replay,
        ChaChaRandomSource::from_seed(42),
        population(),
    );
    let mut er = Acs2ErAgent::new(config(), replay, ChaChaRandomSource::from_seed(1));
    er.restore(source.capture());
    er.run_explore_trial(
        &mut Transition::new(false, true, 4.0),
        &FirstAction,
        &MaxFitnessBootstrap,
        100,
    );
    assert_prediction(er.population(), 28.0, 4.0, true, 1);
}

#[test]
#[should_panic(expected = "truncation mode cannot change with stored replay samples")]
fn replay_mode_cannot_change_after_samples_have_been_stored() {
    let replay = ReplayConfiguration {
        buffer_size: 4,
        min_samples: 2,
        samples_number: 1,
    };
    let mut agent = Acs2ErAgent::with_population(config(), replay, FirstRandom, population());
    agent.run_explore_trial(
        &mut Transition::new(false, true, 4.0),
        &FirstAction,
        &MaxFitnessBootstrap,
        100,
    );
    agent.with_truncation_mode(TruncationMode::Pyalcs);
}
