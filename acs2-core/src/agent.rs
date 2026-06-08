use crate::action_selection::{ActionSelector, BestAction};
use crate::alp::apply_alp;
use crate::config::Configuration;
use crate::environment::Environment;
use crate::ga::apply_ga;
use crate::population::{ClassifierRef, Population};
use crate::rl::{apply_reinforcement_learning, BootstrapEstimator};
use crate::rng::RandomSource;

pub struct TrialMetrics {
    pub steps: u32,
    pub reward: f64,
}

struct PreviousStep<const N: usize> {
    action_set: Vec<ClassifierRef>,
    state: crate::perception::Perception<N>,
    action: usize,
    reward: f64,
}

pub struct Agent<const N: usize, R: RandomSource> {
    population: Population<N>,
    config: Configuration,
    rng: R,
}

impl<const N: usize, R: RandomSource> Agent<N, R> {
    pub fn new(config: Configuration, rng: R) -> Self {
        Self {
            population: Population::new(),
            config,
            rng,
        }
    }

    pub fn with_population(config: Configuration, rng: R, population: Population<N>) -> Self {
        Self {
            population,
            config,
            rng,
        }
    }

    pub fn population(&self) -> &Population<N> {
        &self.population
    }

    pub fn config(&self) -> &Configuration {
        &self.config
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
        let population = &mut self.population;
        let config = &self.config;
        let rng = &mut self.rng;

        let mut state = env.reset();
        let mut steps: u32 = 0;
        let mut total_reward = 0.0;
        let mut previous: Option<PreviousStep<N>> = None;

        loop {
            let mut match_set = population.form_match_set(&state);

            if let Some(mut prior) = previous.take() {
                apply_alp(
                    population,
                    &mut match_set,
                    &mut prior.action_set,
                    &prior.state,
                    prior.action,
                    &state,
                    time + steps as u64,
                    config,
                    rng,
                );
                match_set = population.form_match_set(&state);
                let bootstrap_value = bootstrap.estimate(population, &match_set);
                apply_reinforcement_learning(
                    population,
                    &prior.action_set,
                    prior.reward,
                    bootstrap_value,
                    config.beta,
                    config.gamma,
                );
                if config.do_ga {
                    apply_ga(
                        time + steps as u64,
                        population,
                        &mut match_set,
                        &mut prior.action_set,
                        &state,
                        config,
                        rng,
                    );
                    match_set = population.form_match_set(&state);
                }
            }

            let action = selector.select(population, &match_set, rng);
            let mut action_set = population.form_action_set(&match_set, action);
            let acting_state = state;

            let outcome = env.step(action);
            total_reward += outcome.reward;
            state = outcome.observation;
            steps += 1;

            if outcome.terminated || outcome.truncated {
                let mut terminal_match: Vec<ClassifierRef> = Vec::new();
                apply_alp(
                    population,
                    &mut terminal_match,
                    &mut action_set,
                    &acting_state,
                    action,
                    &state,
                    time + steps as u64,
                    config,
                    rng,
                );
                apply_reinforcement_learning(
                    population,
                    &action_set,
                    outcome.reward,
                    0.0,
                    config.beta,
                    config.gamma,
                );
                if config.do_ga {
                    apply_ga(
                        time + steps as u64,
                        population,
                        &mut terminal_match,
                        &mut action_set,
                        &state,
                        config,
                        rng,
                    );
                }
                break;
            }

            previous = Some(PreviousStep {
                action_set,
                state: acting_state,
                action,
                reward: outcome.reward,
            });
        }

        TrialMetrics {
            steps,
            reward: total_reward,
        }
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
        let _ = time;
        let population = &mut self.population;
        let config = &self.config;
        let rng = &mut self.rng;

        let best_action = BestAction {
            number_of_possible_actions: config.number_of_possible_actions,
        };

        let mut state = env.reset();
        let mut steps: u32 = 0;
        let mut total_reward = 0.0;
        let mut previous: Option<(Vec<ClassifierRef>, f64)> = None;

        loop {
            let match_set = population.form_match_set(&state);

            if let Some((action_set, reward)) = previous.take() {
                let bootstrap_value = bootstrap.estimate(population, &match_set);
                apply_reinforcement_learning(
                    population,
                    &action_set,
                    reward,
                    bootstrap_value,
                    config.beta,
                    config.gamma,
                );
            }

            let action = best_action.select(population, &match_set, rng);
            let action_set = population.form_action_set(&match_set, action);

            let outcome = env.step(action);
            total_reward += outcome.reward;
            state = outcome.observation;
            steps += 1;

            if outcome.terminated || outcome.truncated {
                apply_reinforcement_learning(
                    population,
                    &action_set,
                    outcome.reward,
                    0.0,
                    config.beta,
                    config.gamma,
                );
                break;
            }

            previous = Some((action_set, outcome.reward));
        }

        TrialMetrics {
            steps,
            reward: total_reward,
        }
    }
}
