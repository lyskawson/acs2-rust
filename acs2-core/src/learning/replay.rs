use crate::learning::alp::apply_alp;
use crate::learning::ga::apply_ga;
use crate::learning::rl::{apply_reinforcement_learning, BootstrapEstimator};
use crate::representation::perception::Perception;
use crate::representation::population::{ClassifierRef, Population};
use crate::support::config::Configuration;
use crate::support::rng::RandomSource;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ReplaySample<const N: usize> {
    pub state: Perception<N>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Perception<N>,
    pub done: bool,
}

pub fn replay_learning_step<const N: usize, B>(
    population: &mut Population<N>,
    config: &Configuration,
    bootstrap: &B,
    sample: &ReplaySample<N>,
    time: u64,
    rng: &mut dyn RandomSource,
) where
    B: BootstrapEstimator<N>,
{
    let match_set = population.form_match_set(&sample.state);
    let mut action_set = population.form_action_set(&match_set, sample.action);
    let mut next_match_set = population.form_match_set(&sample.next_state);

    apply_alp(
        population,
        &mut next_match_set,
        &mut action_set,
        &sample.state,
        sample.action,
        &sample.next_state,
        time,
        config,
        rng,
    );

    next_match_set = population.form_match_set(&sample.next_state);
    let bootstrap_value = if sample.done {
        0.0
    } else {
        bootstrap.estimate(population, &next_match_set)
    };
    apply_reinforcement_learning(
        population,
        &action_set,
        sample.reward,
        bootstrap_value,
        config.beta,
        config.gamma,
    );

    if config.do_ga {
        let mut ga_match_set: Vec<ClassifierRef> = if sample.done {
            Vec::new()
        } else {
            next_match_set
        };
        apply_ga(
            time,
            population,
            &mut ga_match_set,
            &mut action_set,
            &sample.next_state,
            config,
            rng,
        );
    }
}
