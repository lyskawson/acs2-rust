use crate::classifier::Classifier;
use crate::population::{ClassifierRef, Population};

pub trait BootstrapEstimator<const N: usize> {
    fn estimate(&self, population: &Population<N>, match_set: &[ClassifierRef]) -> f64;
}

pub struct MaxFitnessBootstrap;

impl<const N: usize> BootstrapEstimator<N> for MaxFitnessBootstrap {
    fn estimate(&self, population: &Population<N>, match_set: &[ClassifierRef]) -> f64 {
        population.get_maximum_fitness(match_set)
    }
}

pub fn update_classifier<const N: usize>(
    classifier: &mut Classifier<N>,
    reward: f64,
    bootstrap: f64,
    beta: f64,
    gamma: f64,
) {
    let discounted_reward = reward + gamma * bootstrap;
    classifier.r += beta * (discounted_reward - classifier.r);
    classifier.ir += beta * (reward - classifier.ir);
}

pub fn apply_reinforcement_learning<const N: usize>(
    population: &mut Population<N>,
    action_set: &[ClassifierRef],
    reward: f64,
    bootstrap: f64,
    beta: f64,
    gamma: f64,
) {
    for &reference in action_set {
        update_classifier(population.get_mut(reference), reward, bootstrap, beta, gamma);
    }
}
