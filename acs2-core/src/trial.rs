use crate::action_selection::{ActionSelector, BestAction};
use crate::config::Configuration;
use crate::environment::Environment;
use crate::population::{ClassifierRef, Population};
use crate::rl::{apply_reinforcement_learning, BootstrapEstimator};
use crate::rng::RandomSource;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TruncationMode {
    #[default]
    Bootstrap,
    Pyalcs,
}

impl TruncationMode {
    pub fn is_terminal(self, terminated: bool, truncated: bool) -> bool {
        terminated || (truncated && self == Self::Pyalcs)
    }
}

pub struct TrialMetrics {
    pub steps: u32,
    pub reward: f64,
}

pub trait LearningAgent<const N: usize> {
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
        B: BootstrapEstimator<N>;

    fn run_exploit_trial<E, B>(&mut self, env: &mut E, bootstrap: &B, time: u64) -> TrialMetrics
    where
        E: Environment<N>,
        B: BootstrapEstimator<N>;

    fn population(&self) -> &Population<N>;

    fn config(&self) -> &Configuration;
}

pub fn run_exploit_trial<const N: usize, E, B>(
    population: &mut Population<N>,
    config: &Configuration,
    rng: &mut dyn RandomSource,
    env: &mut E,
    bootstrap: &B,
) -> TrialMetrics
where
    E: Environment<N>,
    B: BootstrapEstimator<N>,
{
    run_exploit_trial_with_mode(
        population,
        config,
        rng,
        env,
        bootstrap,
        TruncationMode::default(),
    )
}

pub fn run_exploit_trial_with_mode<const N: usize, E, B>(
    population: &mut Population<N>,
    config: &Configuration,
    rng: &mut dyn RandomSource,
    env: &mut E,
    bootstrap: &B,
    truncation_mode: TruncationMode,
) -> TrialMetrics
where
    E: Environment<N>,
    B: BootstrapEstimator<N>,
{
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
            let bootstrap_value =
                if truncation_mode.is_terminal(outcome.terminated, outcome.truncated) {
                    0.0
                } else {
                    let next_match_set = population.form_match_set(&state);
                    bootstrap.estimate(population, &next_match_set)
                };
            apply_reinforcement_learning(
                population,
                &action_set,
                outcome.reward,
                bootstrap_value,
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
