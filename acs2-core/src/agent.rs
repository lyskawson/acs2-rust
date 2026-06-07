use crate::action_selection::ActionSelector;
use crate::config::Configuration;
use crate::environment::Environment;
use crate::population::Population;
use crate::rl::BootstrapEstimator;
use crate::rng::RandomSource;

pub struct TrialMetrics {
    pub steps: u32,
    pub reward: f64,
}

pub struct Agent<const N: usize, R: RandomSource> {
    population: Population<N>,
    config: Configuration,
    rng: R,
}

impl<const N: usize, R: RandomSource> Agent<N, R> {
    pub fn new(config: Configuration, rng: R) -> Self {
        todo!()
    }

    pub fn with_population(config: Configuration, rng: R, population: Population<N>) -> Self {
        todo!()
    }

    pub fn population(&self) -> &Population<N> {
        todo!()
    }

    pub fn config(&self) -> &Configuration {
        todo!()
    }

    pub fn run_explore_trial<E, S, B>(
        &mut self,
        env: &mut E,
        selector: &S,
        bootstrap: &B,
        time: u64,
    ) -> TrialMetrics
    where
        E: Environment<N>,
        S: ActionSelector<N>,
        B: BootstrapEstimator<N>,
    {
        todo!()
    }

    pub fn run_exploit_trial<E, B>(
        &mut self,
        env: &mut E,
        bootstrap: &B,
        time: u64,
    ) -> TrialMetrics
    where
        E: Environment<N>,
        B: BootstrapEstimator<N>,
    {
        todo!()
    }
}
