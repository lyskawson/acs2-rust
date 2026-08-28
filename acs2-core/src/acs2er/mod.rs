pub mod replay;

use crate::action_selection::ActionSelector;
use crate::alp::apply_alp;
use crate::config::Configuration;
use crate::environment::Environment;
use crate::ga::apply_ga;
use crate::population::{ClassifierRef, Population};
use crate::rl::{apply_reinforcement_learning, BootstrapEstimator};
use crate::rng::RandomSource;
use crate::trial::{self, LearningAgent, TrialMetrics};

pub use replay::{ReplayConfiguration, ReplayMemory, ReplaySample};

pub struct Acs2ErAgent<const N: usize, R: RandomSource> {
    population: Population<N>,
    config: Configuration,
    replay_config: ReplayConfiguration,
    replay_memory: ReplayMemory<N>,
    rng: R,
}

impl<const N: usize, R: RandomSource> Acs2ErAgent<N, R> {
    pub fn new(config: Configuration, replay_config: ReplayConfiguration, rng: R) -> Self {
        Self {
            population: Population::new(),
            config,
            replay_config,
            replay_memory: ReplayMemory::new(replay_config.buffer_size),
            rng,
        }
    }

    pub fn with_population(
        config: Configuration,
        replay_config: ReplayConfiguration,
        rng: R,
        population: Population<N>,
    ) -> Self {
        Self {
            population,
            config,
            replay_config,
            replay_memory: ReplayMemory::new(replay_config.buffer_size),
            rng,
        }
    }

    pub fn replay_config(&self) -> &ReplayConfiguration {
        &self.replay_config
    }

    pub fn replay_memory(&self) -> &ReplayMemory<N> {
        &self.replay_memory
    }
}

impl<const N: usize, R: RandomSource> LearningAgent<N> for Acs2ErAgent<N, R> {
    fn population(&self) -> &Population<N> {
        &self.population
    }

    fn config(&self) -> &Configuration {
        &self.config
    }

    fn run_explore_trial<E, S, B>(
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
        let replay_config = &self.replay_config;
        let replay_memory = &mut self.replay_memory;
        let rng = &mut self.rng;

        let mut state = env.reset();
        let mut steps: u32 = 0;
        let mut total_reward = 0.0;

        loop {
            let match_set = population.form_match_set(&state);
            let action = selector.select(population, &match_set, rng);
            let acting_state = state;

            let outcome = env.step(action);
            total_reward += outcome.reward;
            state = outcome.observation;
            let done = outcome.terminated || outcome.truncated;

            replay_memory.update(ReplaySample {
                state: acting_state,
                action,
                reward: outcome.reward,
                next_state: state,
                done,
            });

            if replay_memory.len() >= replay_config.min_samples {
                let replayed = replay_memory.sample_indices(replay_config.samples_number, rng);
                for index in replayed {
                    let sample = replay_memory.get(index);
                    replay_learning_step(
                        population,
                        config,
                        bootstrap,
                        &sample,
                        time + steps as u64,
                        rng,
                    );
                }
            }

            steps += 1;

            if done {
                break;
            }
        }

        TrialMetrics {
            steps,
            reward: total_reward,
        }
    }

    fn run_exploit_trial<E, B>(&mut self, env: &mut E, bootstrap: &B, time: u64) -> TrialMetrics
    where
        E: Environment<N>,
        B: BootstrapEstimator<N>,
    {
        let _ = time;
        trial::run_exploit_trial(
            &mut self.population,
            &self.config,
            &mut self.rng,
            env,
            bootstrap,
        )
    }
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
