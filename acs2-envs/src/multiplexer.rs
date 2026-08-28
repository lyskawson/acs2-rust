use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::knowledge::Transition;
use acs2_core::perception::Perception;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;

const DIGIT_ZERO: u8 = b'0';
const VALIDATION_CLEAR: u8 = 0;
const VALIDATION_SET: u8 = 1;
const REWARD_CORRECT: f64 = 1000.0;
const REWARD_WRONG: f64 = 0.0;
const NUMBER_OF_POSSIBLE_ACTIONS: usize = 2;
const EXHAUSTIVE_INPUT_BITS_LIMIT: usize = 20;

pub const fn control_bits_for(perception_length: usize) -> usize {
    let mut control_bits = 0usize;
    loop {
        let observation_length = control_bits + (1usize << control_bits) + 1;
        if observation_length == perception_length {
            return control_bits;
        }
        if observation_length > perception_length {
            panic!("perception length is not a valid multiplexer size: it must equal control_bits + 2^control_bits + 1");
        }
        control_bits += 1;
    }
}

pub fn get_correct_answer(input_bits: &[u8], control_bits: usize) -> usize {
    let mut address = 0usize;
    for &bit in &input_bits[..control_bits] {
        address = (address << 1) | bit as usize;
    }
    input_bits[control_bits + address] as usize
}

pub struct Multiplexer<const N: usize> {
    perception: [Symbol; N],
    rng: Box<dyn RandomSource>,
}

impl<const N: usize> Multiplexer<N> {
    pub const CONTROL_BITS: usize = control_bits_for(N);
    pub const INPUT_BITS: usize = N - 1;
    pub const NUMBER_OF_POSSIBLE_ACTIONS: usize = NUMBER_OF_POSSIBLE_ACTIONS;

    pub fn new(rng: Box<dyn RandomSource>) -> Self {
        let _ = Self::CONTROL_BITS;
        Self {
            perception: [Symbol::Token(DIGIT_ZERO); N],
            rng,
        }
    }

    fn bit_at(&self, index: usize) -> u8 {
        match self.perception[index] {
            Symbol::Token(value) => value - DIGIT_ZERO,
            Symbol::Wildcard => 0,
        }
    }

    fn correct_answer(&self) -> usize {
        let input: [u8; N] = core::array::from_fn(|index| self.bit_at(index));
        get_correct_answer(&input[..Self::INPUT_BITS], Self::CONTROL_BITS)
    }
}

impl<const N: usize> Environment<N> for Multiplexer<N> {
    fn reset(&mut self) -> Perception<N> {
        for index in 0..Self::INPUT_BITS {
            let bit = self.rng.gen_range(2) as u8;
            self.perception[index] = Symbol::Token(DIGIT_ZERO + bit);
        }
        self.perception[N - 1] = Symbol::Token(DIGIT_ZERO + VALIDATION_CLEAR);
        Perception::new(self.perception)
    }

    fn step(&mut self, action: usize) -> StepOutcome<N> {
        let reward = if action == self.correct_answer() {
            self.perception[N - 1] = Symbol::Token(DIGIT_ZERO + VALIDATION_SET);
            REWARD_CORRECT
        } else {
            REWARD_WRONG
        };

        StepOutcome {
            observation: Perception::new(self.perception),
            reward,
            terminated: true,
            truncated: false,
            info: (),
        }
    }
}

fn decode_input<const N: usize>(value: usize, input_bits: usize) -> [u8; N] {
    let mut input = [0u8; N];
    for index in 0..input_bits {
        input[index] = ((value >> (input_bits - 1 - index)) & 1) as u8;
    }
    input
}

fn make_transition<const N: usize>(input: &[u8; N], action: usize, correct_answer: usize) -> Transition<N> {
    let mut p0 = [Symbol::Token(DIGIT_ZERO); N];
    for index in 0..N - 1 {
        p0[index] = Symbol::Token(DIGIT_ZERO + input[index]);
    }
    p0[N - 1] = Symbol::Token(DIGIT_ZERO + VALIDATION_CLEAR);

    let mut p1 = p0;
    if action == correct_answer {
        p1[N - 1] = Symbol::Token(DIGIT_ZERO + VALIDATION_SET);
    }

    Transition::new(Perception::new(p0), action, Perception::new(p1))
}

pub fn exhaustive_transitions<const N: usize>() -> impl Iterator<Item = Transition<N>> {
    let control_bits = control_bits_for(N);
    let input_bits = N - 1;
    let total_inputs = 1usize << input_bits;
    (0..total_inputs).flat_map(move |value| {
        let input = decode_input::<N>(value, input_bits);
        let correct_answer = get_correct_answer(&input[..input_bits], control_bits);
        [0usize, 1usize]
            .into_iter()
            .map(move |action| make_transition::<N>(&input, action, correct_answer))
    })
}

pub fn sampled_transitions<const N: usize>(sample_inputs: usize, seed: u64) -> Vec<Transition<N>> {
    let control_bits = control_bits_for(N);
    let input_bits = N - 1;
    let mut rng = ChaChaRandomSource::from_seed(seed);
    let mut transitions = Vec::with_capacity(sample_inputs * NUMBER_OF_POSSIBLE_ACTIONS);
    for _ in 0..sample_inputs {
        let mut input = [0u8; N];
        for index in 0..input_bits {
            input[index] = rng.gen_range(2) as u8;
        }
        let correct_answer = get_correct_answer(&input[..input_bits], control_bits);
        for action in 0..NUMBER_OF_POSSIBLE_ACTIONS {
            transitions.push(make_transition::<N>(&input, action, correct_answer));
        }
    }
    transitions
}

fn input_index_of<const N: usize>(input: &[u8; N], input_bits: usize) -> usize {
    let mut value = 0usize;
    for index in 0..input_bits {
        value = (value << 1) | input[index] as usize;
    }
    value
}

fn mark_covered_cells<const N: usize>(
    classifier: &acs2_core::classifier::Classifier<N>,
    control_bits: usize,
    input_bits: usize,
    covered: &mut [bool],
) {
    let action = match classifier.action {
        Some(action) => action,
        None => return,
    };

    if let Symbol::Token(value) = classifier.condition.get(N - 1) {
        if value != DIGIT_ZERO + VALIDATION_CLEAR {
            return;
        }
    }

    for index in 0..input_bits {
        if !classifier.effect.get(index).is_wildcard() {
            return;
        }
    }

    let anticipates_flip = match classifier.effect.get(N - 1) {
        Symbol::Wildcard => false,
        Symbol::Token(value) if value == DIGIT_ZERO + VALIDATION_SET => true,
        Symbol::Token(_) => return,
    };

    let mut base_input = [0u8; N];
    let mut free_positions: Vec<usize> = Vec::new();
    for index in 0..input_bits {
        match classifier.condition.get(index) {
            Symbol::Wildcard => free_positions.push(index),
            Symbol::Token(value) => base_input[index] = value - DIGIT_ZERO,
        }
    }

    for combination in 0..(1usize << free_positions.len()) {
        let mut input = base_input;
        for (bit, &position) in free_positions.iter().enumerate() {
            input[position] = ((combination >> bit) & 1) as u8;
        }
        let correct_answer = get_correct_answer(&input[..input_bits], control_bits);
        let predicts = if anticipates_flip {
            action == correct_answer
        } else {
            action != correct_answer
        };
        if predicts {
            let input_index = input_index_of::<N>(&input, input_bits);
            covered[(input_index << 1) | action] = true;
        }
    }
}

pub fn exhaustive_knowledge<const N: usize>(
    population: &acs2_core::population::Population<N>,
    theta_r: f64,
) -> f64 {
    let control_bits = control_bits_for(N);
    let input_bits = N - 1;
    let total_cells = (1usize << input_bits) << 1;
    let mut covered = vec![false; total_cells];

    for classifier in population.iter().filter(|classifier| classifier.is_reliable(theta_r)) {
        mark_covered_cells::<N>(classifier, control_bits, input_bits, &mut covered);
    }

    let count = covered.iter().filter(|&&cell| cell).count();
    count as f64 / total_cells as f64
}

pub fn evaluate_knowledge<const N: usize>(
    population: &acs2_core::population::Population<N>,
    theta_r: f64,
    sample_inputs: usize,
    sample_seed: u64,
) -> f64 {
    if N - 1 <= EXHAUSTIVE_INPUT_BITS_LIMIT {
        exhaustive_knowledge::<N>(population, theta_r)
    } else {
        acs2_core::knowledge::anticipation_fraction(
            population,
            theta_r,
            sampled_transitions::<N>(sample_inputs, sample_seed),
        )
    }
}

macro_rules! define_multiplexers {
    ($($alias:ident => $control_bits:literal),+ $(,)?) => {
        $(
            pub type $alias = Multiplexer<{ $control_bits + (1usize << $control_bits) + 1 }>;
        )+
    };
}

define_multiplexers! {
    Mpx6 => 2,
    Mpx11 => 3,
    Mpx20 => 4,
    Mpx37 => 5,
    Mpx70 => 6,
    Mpx135 => 7,
}

#[cfg(test)]
mod tests {
    use super::*;
    use acs2_core::action_selection::EpsilonGreedy;
    use acs2_core::agent::Agent;
    use acs2_core::classifier::Classifier;
    use acs2_core::config::Configuration;
    use acs2_core::knowledge::anticipation_fraction;
    use acs2_core::population::Population;
    use acs2_core::rl::MaxFitnessBootstrap;
    use acs2_core::trial::LearningAgent;

    fn token(value: u8) -> Symbol {
        Symbol::Token(b'0' + value)
    }

    #[test]
    fn get_correct_answer_matches_pyalcs_3bit() {
        assert_eq!(get_correct_answer(&[0, 1, 0, 0], 1), 1);
        assert_eq!(get_correct_answer(&[1, 1, 0, 0], 1), 0);
    }

    #[test]
    fn get_correct_answer_matches_pyalcs_6bit() {
        assert_eq!(get_correct_answer(&[1, 1, 0, 1, 0, 0, 0], 2), 0);
    }

    #[test]
    fn get_correct_answer_matches_pyalcs_11bit() {
        assert_eq!(get_correct_answer(&[1, 0, 1, 1, 0, 1, 1, 0, 1, 0], 3), 1);
    }

    fn assert_mpx<const N: usize>(expected_input_bits: usize) {
        let mut env = Multiplexer::<N>::new(Box::new(ChaChaRandomSource::from_seed(1)));
        let perception = env.reset();
        assert_eq!(perception.symbols.len(), expected_input_bits + 1);
        assert_eq!(N, expected_input_bits + 1);
        let control_bits = Multiplexer::<N>::CONTROL_BITS;
        assert_eq!(control_bits + (1usize << control_bits), expected_input_bits);
        assert_eq!(Multiplexer::<N>::INPUT_BITS, expected_input_bits);
    }

    #[test]
    fn perception_length_equals_input_bits_plus_one_for_every_size() {
        assert_mpx::<7>(6);
        assert_mpx::<12>(11);
        assert_mpx::<21>(20);
        assert_mpx::<38>(37);
        assert_mpx::<71>(70);
        assert_mpx::<136>(135);
    }

    #[test]
    fn multiplexer_uses_two_actions() {
        assert_eq!(Multiplexer::<7>::NUMBER_OF_POSSIBLE_ACTIONS, 2);
        assert_eq!(Configuration::mpx().number_of_possible_actions, 2);
    }

    #[test]
    fn reset_pins_validation_bit_and_step_flips_only_on_correct_answer() {
        let mut env = Multiplexer::<7>::new(Box::new(ChaChaRandomSource::from_seed(7)));
        let observation = env.reset();
        assert_eq!(observation.get(6), token(0));

        let correct = env.correct_answer();
        let outcome = env.step(correct);
        assert_eq!(outcome.observation.get(6), token(1));
        assert_eq!(outcome.reward, REWARD_CORRECT);
        assert!(outcome.terminated);
        assert!(!outcome.truncated);

        let mut env = Multiplexer::<7>::new(Box::new(ChaChaRandomSource::from_seed(7)));
        env.reset();
        let correct = env.correct_answer();
        let outcome = env.step(1 - correct);
        assert_eq!(outcome.observation.get(6), token(0));
        assert_eq!(outcome.reward, REWARD_WRONG);
        assert!(outcome.terminated);
    }

    #[test]
    fn step_leaves_input_bits_unchanged() {
        let mut env = Multiplexer::<12>::new(Box::new(ChaChaRandomSource::from_seed(11)));
        let before = env.reset();
        let correct = env.correct_answer();
        let after = env.step(correct).observation;
        for index in 0..Multiplexer::<12>::INPUT_BITS {
            assert_eq!(before.get(index), after.get(index));
        }
    }

    fn reliable_half_classifier<const N: usize>(action: usize, anticipates_change: bool) -> Classifier<N> {
        let mut classifier = Classifier::general(Some(action), &Configuration::mpx());
        classifier.condition.set(0, token(0));
        if anticipates_change {
            classifier.effect.set(N - 1, token(1));
        }
        classifier.q = 1.0;
        classifier
    }

    fn half_coverage_population<const N: usize>() -> Population<N> {
        let mut classifiers = Vec::new();
        for action in 0..NUMBER_OF_POSSIBLE_ACTIONS {
            classifiers.push(reliable_half_classifier::<N>(action, false));
            classifiers.push(reliable_half_classifier::<N>(action, true));
        }
        Population::from_classifiers(classifiers)
    }

    fn assert_sampler_agrees<const N: usize>() {
        const SAMPLE_INPUTS: usize = 200_000;
        const SAMPLE_SEED: u64 = 0x6D70_7831;
        const TOLERANCE: f64 = 0.005;

        let population = half_coverage_population::<N>();
        let theta_r = 0.9;

        let exhaustive = anticipation_fraction(&population, theta_r, exhaustive_transitions::<N>());
        let sampled =
            anticipation_fraction(&population, theta_r, sampled_transitions::<N>(SAMPLE_INPUTS, SAMPLE_SEED));

        assert!(
            (exhaustive - 0.5).abs() < 1e-12,
            "synthetic half-coverage population should yield exactly 0.5 exhaustively, got {exhaustive}"
        );
        assert!(
            (sampled - exhaustive).abs() <= TOLERANCE,
            "sampled {sampled} disagrees with exhaustive {exhaustive} beyond {TOLERANCE}"
        );
    }

    fn train_population<const N: usize>(explore_trials: u32, seed: u64) -> (Population<N>, f64) {
        let mut config = Configuration::mpx();
        config.epsilon = 0.8;
        let mut env = Multiplexer::<N>::new(Box::new(ChaChaRandomSource::from_seed(seed)));
        let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
        let selector = EpsilonGreedy {
            number_of_possible_actions: Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
            epsilon: 0.8,
        };
        let bootstrap = MaxFitnessBootstrap;
        let mut time = 0u64;
        for _ in 0..explore_trials {
            let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
            time += metrics.steps as u64;
        }
        let theta_r = agent.config().theta_r;
        (Population::from_classifiers(agent.population().iter().cloned().collect()), theta_r)
    }

    #[test]
    fn fast_exhaustive_matches_naive_on_trained_population() {
        let (population, theta_r) = train_population::<7>(8_000, 42);
        let fast = exhaustive_knowledge::<7>(&population, theta_r);
        let naive = anticipation_fraction(&population, theta_r, exhaustive_transitions::<7>());
        assert_eq!(fast, naive);
    }

    #[test]
    fn fast_exhaustive_matches_naive_on_synthetic_population() {
        let population = half_coverage_population::<12>();
        let fast = exhaustive_knowledge::<12>(&population, 0.9);
        let naive = anticipation_fraction(&population, 0.9, exhaustive_transitions::<12>());
        assert_eq!(fast, naive);
        assert_eq!(fast, 0.5);
    }

    #[test]
    fn sampled_estimator_agrees_with_exhaustive_at_k11() {
        assert_sampler_agrees::<12>();
    }

    #[test]
    fn sampled_estimator_agrees_with_exhaustive_at_k20() {
        assert_sampler_agrees::<21>();
    }
}
