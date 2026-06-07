use crate::population::{ClassifierRef, Population};
use crate::rng::RandomSource;

pub trait ActionSelector<const N: usize> {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize;
}

pub struct EpsilonGreedy {
    pub number_of_possible_actions: usize,
    pub epsilon: f64,
}

pub struct BestAction {
    pub number_of_possible_actions: usize,
}

pub struct RandomAction {
    pub number_of_possible_actions: usize,
}

impl<const N: usize> ActionSelector<N> for EpsilonGreedy {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        todo!()
    }
}

impl<const N: usize> ActionSelector<N> for BestAction {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        todo!()
    }
}

impl<const N: usize> ActionSelector<N> for RandomAction {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        todo!()
    }
}
