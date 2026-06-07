use crate::classifier::Classifier;
use crate::config::Configuration;
use crate::perception::Perception;
use crate::population::{ClassifierRef, Population};
use crate::rng::RandomSource;

pub fn apply_ga<const N: usize>(
    time: u64,
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    state: &Perception<N>,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    todo!()
}

pub fn should_apply<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    time: u64,
    theta_ga: u32,
) -> bool {
    todo!()
}

pub fn roulette_wheel_selection<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    rng: &mut dyn RandomSource,
) -> (ClassifierRef, ClassifierRef) {
    todo!()
}

pub fn generalizing_mutation<const N: usize>(
    classifier: &mut Classifier<N>,
    mu: f64,
    rng: &mut dyn RandomSource,
) {
    todo!()
}

pub fn two_point_crossover<const N: usize>(
    first: &mut Classifier<N>,
    second: &mut Classifier<N>,
    rng: &mut dyn RandomSource,
) {
    todo!()
}
