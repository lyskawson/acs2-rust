use acs2_agents::acs2::Agent;
use acs2_core::learning::action_selection::EpsilonGreedy;
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::knowledge::anticipation_fraction;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::LearningAgent;
use acs2_envs::multiplexer::{exhaustive_knowledge, exhaustive_transitions, Encoding, Multiplexer};

fn train_population<const N: usize>(explore_trials: u32, seed: u64) -> (Population<N>, f64) {
    let mut config = Configuration::mpx();
    config.epsilon = 0.8;
    let mut env = Multiplexer::<N>::new(Box::new(ChaChaRandomSource::from_seed(seed)));
    let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
    let selector = EpsilonGreedy {
        number_of_possible_actions: Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
        epsilon: 0.8,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut time = 0u64;
    for _ in 0..explore_trials {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }
    let theta_r = agent.config().theta_r;
    (Population::from_classifiers(agent.population().iter().cloned().collect()), theta_r)
}

#[test]
fn fast_exhaustive_matches_naive_on_trained_population() {
    let (population, theta_r) = train_population::<7>(8_000, 42);
    let fast = exhaustive_knowledge::<7>(&population, theta_r);
    let naive = anticipation_fraction(&population, theta_r, exhaustive_transitions::<7>(Encoding::Flip));
    assert_eq!(fast, naive);
}
