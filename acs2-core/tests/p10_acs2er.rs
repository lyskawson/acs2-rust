use acs2_core::acs2er::replay::{ReplayConfiguration, ReplayMemory, ReplaySample};
use acs2_core::acs2er::Acs2ErAgent;
use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;

fn state(values: [u8; 2]) -> Perception<2> {
    Perception {
        symbols: values.map(Symbol::Token),
    }
}

fn sample(action: usize) -> ReplaySample<2> {
    ReplaySample {
        state: state([b'0', b'0']),
        action,
        reward: 0.5,
        next_state: state([b'1', b'1']),
        done: false,
    }
}

struct FixedLengthEnv {
    trial_length: u32,
    steps_in_trial: u32,
}

impl FixedLengthEnv {
    fn new(trial_length: u32) -> Self {
        Self {
            trial_length,
            steps_in_trial: 0,
        }
    }
}

impl Environment<2> for FixedLengthEnv {
    fn reset(&mut self) -> Perception<2> {
        self.steps_in_trial = 0;
        state([b'0', b'0'])
    }

    fn step(&mut self, _action: usize) -> StepOutcome<2> {
        self.steps_in_trial += 1;
        if self.steps_in_trial >= self.trial_length {
            self.steps_in_trial = 0;
            StepOutcome {
                observation: state([b'1', b'1']),
                reward: 10.0,
                terminated: true,
                truncated: false,
                info: (),
            }
        } else {
            StepOutcome {
                observation: state([b'1', b'0']),
                reward: 1.0,
                terminated: false,
                truncated: false,
                info: (),
            }
        }
    }
}

fn test_config() -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        ..Configuration::default_protocol()
    }
}

fn test_replay_config() -> ReplayConfiguration {
    ReplayConfiguration {
        buffer_size: 100,
        min_samples: 20,
        samples_number: 2,
    }
}

fn population_signature(classifiers: &[&Classifier<2>]) -> Vec<String> {
    classifiers
        .iter()
        .map(|classifier| {
            format!(
                "{:?}|{:?}|{:?}|{:.12}|{:.12}|{:.12}|{}|{}|{:?}|{:?}",
                classifier.condition.symbols,
                classifier.action,
                classifier.effect.symbols,
                classifier.q,
                classifier.r,
                classifier.ir,
                classifier.num,
                classifier.exp,
                classifier.talp,
                classifier.mark.attributes,
            )
        })
        .collect()
}

fn run_agent(seed: u64, trials: u32, trial_length: u32) -> Vec<String> {
    let mut env = FixedLengthEnv::new(trial_length);
    let mut agent = Acs2ErAgent::<2, _>::new(
        test_config(),
        test_replay_config(),
        ChaChaRandomSource::from_seed(seed),
    );
    let selector = EpsilonGreedy {
        number_of_possible_actions: 2,
        epsilon: 0.5,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut time: u64 = 0;
    for _ in 0..trials {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }
    let population = agent.population();
    let all: Vec<&Classifier<2>> = population.iter().collect();
    population_signature(&all)
}

#[test]
fn replay_memory_stores_a_single_sample() {
    let mut memory = ReplayMemory::<2>::new(5);
    memory.update(sample(1));
    assert_eq!(memory.len(), 1);
    assert!(!memory.is_empty());
    assert_eq!(memory.get(0).action, 1);
}

#[test]
fn replay_memory_drops_the_oldest_sample_when_full() {
    let mut memory = ReplayMemory::<2>::new(3);
    for action in 1..=4 {
        memory.update(sample(action));
    }
    assert_eq!(memory.len(), 3);
    assert_eq!(memory.get(0).action, 2);
    assert_eq!(memory.get(1).action, 3);
    assert_eq!(memory.get(2).action, 4);
}

#[test]
fn replay_memory_never_exceeds_its_bound() {
    let mut memory = ReplayMemory::<2>::new(4);
    for action in 0..50 {
        memory.update(sample(action));
        assert!(memory.len() <= memory.max_size());
    }
    assert_eq!(memory.len(), 4);
    assert_eq!(memory.get(0).action, 46);
    assert_eq!(memory.get(3).action, 49);
}

#[test]
fn sampled_indices_are_distinct_and_in_range() {
    let mut memory = ReplayMemory::<2>::new(64);
    for action in 0..64 {
        memory.update(sample(action));
    }
    let mut rng = ChaChaRandomSource::from_seed(7);
    for _ in 0..500 {
        let indices = memory.sample_indices(5, &mut rng);
        assert_eq!(indices.len(), 5);
        for (position, index) in indices.iter().enumerate() {
            assert!(*index < memory.len());
            assert!(!indices[..position].contains(index));
        }
    }
}

#[test]
fn sampled_indices_are_clamped_to_the_buffer_size() {
    let mut memory = ReplayMemory::<2>::new(64);
    for action in 0..3 {
        memory.update(sample(action));
    }
    let mut rng = ChaChaRandomSource::from_seed(7);
    let indices = memory.sample_indices(10, &mut rng);
    assert_eq!(indices.len(), 3);
}

struct ScriptedRandom {
    range_values: Vec<usize>,
    cursor: usize,
}

impl ScriptedRandom {
    fn new(range_values: Vec<usize>) -> Self {
        Self {
            range_values,
            cursor: 0,
        }
    }

    fn consumed(&self) -> usize {
        self.cursor
    }
}

impl RandomSource for ScriptedRandom {
    fn gen_bool(&mut self, _probability: f64) -> bool {
        unreachable!("replay sampling must not draw booleans")
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        let value = self.range_values[self.cursor];
        self.cursor += 1;
        assert!(value < bound);
        value
    }

    fn gen_unit(&mut self) -> f64 {
        unreachable!("replay sampling must not draw unit floats")
    }
}

#[test]
fn sampling_redraws_until_the_indices_are_distinct() {
    let mut memory = ReplayMemory::<2>::new(5);
    for action in 0..5 {
        memory.update(sample(action));
    }
    let mut rng = ScriptedRandom::new(vec![2, 2, 4, 4, 2, 0]);
    let indices = memory.sample_indices(3, &mut rng);
    assert_eq!(indices, vec![2, 4, 0]);
    assert_eq!(rng.consumed(), 6);
}

#[test]
fn sampling_draws_nothing_when_no_samples_are_requested() {
    let mut memory = ReplayMemory::<2>::new(5);
    memory.update(sample(0));
    let mut rng = ScriptedRandom::new(Vec::new());
    assert!(memory.sample_indices(0, &mut rng).is_empty());
    assert_eq!(rng.consumed(), 0);
}

#[test]
fn the_same_seed_replays_the_same_index_stream() {
    let mut memory = ReplayMemory::<2>::new(64);
    for action in 0..64 {
        memory.update(sample(action));
    }
    let mut first = ChaChaRandomSource::from_seed(11);
    let mut second = ChaChaRandomSource::from_seed(11);
    let mut third = ChaChaRandomSource::from_seed(12);
    let mut first_stream = Vec::new();
    let mut second_stream = Vec::new();
    let mut third_stream = Vec::new();
    for _ in 0..200 {
        first_stream.push(memory.sample_indices(3, &mut first));
        second_stream.push(memory.sample_indices(3, &mut second));
        third_stream.push(memory.sample_indices(3, &mut third));
    }
    assert_eq!(first_stream, second_stream);
    assert_ne!(first_stream, third_stream);
}

#[test]
fn population_stays_empty_while_the_buffer_is_below_min_samples() {
    let mut env = FixedLengthEnv::new(1);
    let mut agent = Acs2ErAgent::<2, _>::new(
        test_config(),
        test_replay_config(),
        ChaChaRandomSource::from_seed(42),
    );
    let selector = EpsilonGreedy {
        number_of_possible_actions: 2,
        epsilon: 0.5,
    };
    let bootstrap = MaxFitnessBootstrap;

    let mut time: u64 = 0;
    for _ in 0..10 {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
        assert!(agent.population().is_empty());
    }

    assert_eq!(agent.population().len(), 0);
    assert_eq!(agent.replay_memory().len(), 10);
}

#[test]
fn learning_starts_once_the_buffer_reaches_min_samples() {
    let mut env = FixedLengthEnv::new(3);
    let mut agent = Acs2ErAgent::<2, _>::new(
        test_config(),
        test_replay_config(),
        ChaChaRandomSource::from_seed(42),
    );
    let selector = EpsilonGreedy {
        number_of_possible_actions: 2,
        epsilon: 0.5,
    };
    let bootstrap = MaxFitnessBootstrap;

    let mut time: u64 = 0;
    let mut population_at_warmup_end = None;
    for trial in 0..50 {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
        if trial == 5 {
            population_at_warmup_end = Some(agent.population().len());
        }
    }

    assert_eq!(population_at_warmup_end, Some(0));
    assert!(agent.population().len() > 0);
    assert_eq!(agent.replay_memory().len(), 100);
}

#[test]
fn the_current_transition_is_never_learned_from_directly() {
    let mut env = FixedLengthEnv::new(4);
    let replay_config = ReplayConfiguration {
        buffer_size: 1_000,
        min_samples: usize::MAX,
        samples_number: 3,
    };
    let mut agent = Acs2ErAgent::<2, _>::new(
        test_config(),
        replay_config,
        ChaChaRandomSource::from_seed(3),
    );
    let selector = EpsilonGreedy {
        number_of_possible_actions: 2,
        epsilon: 0.5,
    };
    let bootstrap = MaxFitnessBootstrap;

    let mut time: u64 = 0;
    for _ in 0..200 {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }

    assert_eq!(agent.population().len(), 0);
    assert_eq!(agent.replay_memory().len(), 800);
}

#[test]
fn identical_seeds_produce_identical_populations() {
    let first = run_agent(2024, 60, 3);
    let second = run_agent(2024, 60, 3);
    assert!(!first.is_empty());
    assert_eq!(first, second);
}

#[test]
fn different_seeds_produce_different_populations() {
    let first = run_agent(2024, 60, 3);
    let second = run_agent(9001, 60, 3);
    assert_ne!(first, second);
}
