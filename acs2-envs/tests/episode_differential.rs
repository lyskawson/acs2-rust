use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::trial::LearningAgent;
use acs2_core::config::Configuration;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::maze::{Maze, MAZE_PERCEPTION_LEN};
use acs2_envs::maze_data::geometry_by_id;
use serde_json::Value;

fn load_episodes() -> Value {
    let path = format!(
        "{}/../fixtures/differential_episode.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

struct RustEpisode {
    steps: u32,
    macro_population: usize,
    numerosity: u32,
    reliable: usize,
}

fn run_rust_episode(maze_id: &str, seed: u64) -> RustEpisode {
    let geometry = geometry_by_id(maze_id).unwrap();
    let config = Configuration::default_protocol();
    let mut env = Maze::from_geometry(geometry, Box::new(ChaChaRandomSource::from_seed(seed)));
    let mut agent =
        Agent::<MAZE_PERCEPTION_LEN, _>::new(config, ChaChaRandomSource::from_seed(seed));
    let selector = EpsilonGreedy {
        number_of_possible_actions: agent.config().number_of_possible_actions,
        epsilon: 0.8,
    };
    let bootstrap = MaxFitnessBootstrap;

    let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, 0);
    let population = agent.population();
    RustEpisode {
        steps: metrics.steps,
        macro_population: population.len(),
        numerosity: population.numerosity(),
        reliable: population.reliable_count(agent.config().theta_r),
    }
}

#[test]
fn explore_episode_metrics_are_reported_against_pyalcs() {
    let data = load_episodes();
    for episode in data["episodes"].as_array().unwrap() {
        let maze_id = episode["maze"].as_str().unwrap();
        let seed = episode["seed"].as_u64().unwrap();
        let cap = geometry_by_id(maze_id).unwrap().max_episode_steps;

        let rust = run_rust_episode(maze_id, seed);

        println!(
            "{maze_id} seed={seed} | pyalcs steps={} macro={} num={} reliable={} \
             || rust steps={} macro={} num={} reliable={}",
            episode["steps"].as_u64().unwrap(),
            episode["macro_population"].as_u64().unwrap(),
            episode["numerosity"].as_u64().unwrap(),
            episode["reliable"].as_u64().unwrap(),
            rust.steps,
            rust.macro_population,
            rust.numerosity,
            rust.reliable,
        );

        assert!(rust.steps <= cap, "{maze_id} steps within truncation cap");
        assert!(rust.macro_population >= 1, "{maze_id} covering grows population");
        assert!(
            rust.numerosity >= rust.macro_population as u32,
            "{maze_id} numerosity >= macro"
        );
    }
}
