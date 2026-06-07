use crate::classifier::Classifier;
use crate::config::Configuration;
use crate::perception::Perception;
use crate::population::{ClassifierRef, Population};
use crate::rng::RandomSource;

pub fn cover<const N: usize>(
    p0: &Perception<N>,
    action: usize,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
) -> Classifier<N> {
    todo!()
}

pub fn expected_case<const N: usize>(
    classifier: &mut Classifier<N>,
    p0: &Perception<N>,
    time: u64,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) -> Option<Classifier<N>> {
    todo!()
}

pub fn unexpected_case<const N: usize>(
    classifier: &mut Classifier<N>,
    p0: &Perception<N>,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
) -> Option<Classifier<N>> {
    todo!()
}

pub fn apply_alp<const N: usize>(
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    p0: &Perception<N>,
    action: usize,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    todo!()
}
