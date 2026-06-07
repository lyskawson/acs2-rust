use crate::classifier::Classifier;
use crate::population::{ClassifierRef, Population};

pub trait BootstrapEstimator<const N: usize> {
    fn estimate(&self, population: &Population<N>, match_set: &[ClassifierRef]) -> f64;
}

pub struct MaxFitnessBootstrap;

impl<const N: usize> BootstrapEstimator<N> for MaxFitnessBootstrap {
    fn estimate(&self, population: &Population<N>, match_set: &[ClassifierRef]) -> f64 {
        todo!()
    }
}

pub fn update_classifier<const N: usize>(
    classifier: &mut Classifier<N>,
    reward: f64,
    bootstrap: f64,
    beta: f64,
    gamma: f64,
) {
    todo!()
}

pub fn apply_reinforcement_learning<const N: usize>(
    population: &mut Population<N>,
    action_set: &[ClassifierRef],
    reward: f64,
    bootstrap: f64,
    beta: f64,
    gamma: f64,
) {
    todo!()
}
