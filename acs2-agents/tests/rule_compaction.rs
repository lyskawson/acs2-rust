use acs2_agents::acs2::Agent;
use acs2_core::compaction::{compact_population, CompactionConfiguration, Decision};
use acs2_core::learning::action_selection::EpsilonGreedy;
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::support::config::Configuration;
use acs2_core::support::knowledge::anticipation_fraction;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_core::trial::LearningAgent;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_envs::maze::knowledge::pyalcs_transitions;
use acs2_envs::maze::{Maze, MAZE_PERCEPTION_LEN};

#[test]
fn compaction_of_a_trained_maze_population_keeps_its_knowledge() {
    let config = Configuration::default_protocol();
    let mut agent: Agent<MAZE_PERCEPTION_LEN, _> =
        Agent::new(config.clone(), ChaChaRandomSource::from_seed(42));
    let mut env = Maze::from_geometry(&MAZE4, Box::new(ChaChaRandomSource::from_seed(42)));
    let selector = EpsilonGreedy {
        number_of_possible_actions: config.number_of_possible_actions,
        epsilon: 0.8,
    };
    let mut time = 0u64;
    for _ in 0..300 {
        time += agent
            .run_explore_trial(&mut env, &selector, &MaxFitnessBootstrap, time)
            .steps as u64;
    }

    let population = agent.population();
    let (compacted, outcome) =
        compact_population(population, &CompactionConfiguration::from_learning(&config));

    assert!(compacted.len() < population.len());
    assert_eq!(compacted.len(), outcome.kept_indices().len());
    assert!(outcome
        .decisions
        .iter()
        .all(|decision| matches!(decision, Decision::Kept | Decision::Subsumed { .. })));
    assert_eq!(
        anticipation_fraction(population, config.theta_r, pyalcs_transitions(&MAZE4)),
        anticipation_fraction(&compacted, config.theta_r, pyalcs_transitions(&MAZE4))
    );
    assert!(anticipation_fraction(population, config.theta_r, pyalcs_transitions(&MAZE4)) > 0.0);
}
