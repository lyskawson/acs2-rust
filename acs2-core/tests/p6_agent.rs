use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::config::Configuration;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;

fn state(values: [u8; 4]) -> Perception<4> {
    Perception {
        symbols: values.map(|value| {
            if value == b'#' {
                Symbol::Wildcard
            } else {
                Symbol::Token(value)
            }
        }),
    }
}

struct ToggleMaze {
    step_count: u32,
}

impl ToggleMaze {
    fn new() -> Self {
        Self { step_count: 0 }
    }
}

impl Environment<4> for ToggleMaze {
    fn reset(&mut self) -> Perception<4> {
        self.step_count = 0;
        state([b'0', b'0', b'0', b'0'])
    }

    fn step(&mut self, _action: usize) -> StepOutcome<4> {
        self.step_count += 1;
        if self.step_count == 1 {
            StepOutcome {
                observation: state([b'1', b'0', b'0', b'0']),
                reward: 0.0,
                terminated: false,
                truncated: false,
                info: (),
            }
        } else {
            StepOutcome {
                observation: state([b'0', b'0', b'0', b'1']),
                reward: 1000.0,
                terminated: true,
                truncated: false,
                info: (),
            }
        }
    }
}

struct NeverEndingMaze {
    step_count: u32,
    cap: u32,
}

impl NeverEndingMaze {
    fn new(cap: u32) -> Self {
        Self { step_count: 0, cap }
    }
}

impl Environment<4> for NeverEndingMaze {
    fn reset(&mut self) -> Perception<4> {
        self.step_count = 0;
        state([b'0', b'0', b'0', b'0'])
    }

    fn step(&mut self, _action: usize) -> StepOutcome<4> {
        self.step_count += 1;
        let observation = if self.step_count % 2 == 0 {
            state([b'0', b'0', b'0', b'0'])
        } else {
            state([b'1', b'0', b'0', b'0'])
        };
        StepOutcome {
            observation,
            reward: 0.0,
            terminated: false,
            truncated: self.step_count >= self.cap,
            info: (),
        }
    }
}

#[test]
fn trials_honor_truncation_when_goal_unreachable() {
    let config = Configuration::default_protocol();
    let rng = ChaChaRandomSource::from_seed(0);
    let mut agent = Agent::<4, _>::new(config, rng);
    let actions = agent.config().number_of_possible_actions;
    let selector = EpsilonGreedy {
        number_of_possible_actions: actions,
        epsilon: 0.8,
    };
    let bootstrap = MaxFitnessBootstrap;

    let mut explore_env = NeverEndingMaze::new(5);
    let explore = agent.run_explore_trial(&mut explore_env, &selector, &bootstrap, 0);
    assert_eq!(explore.steps, 5);

    let mut exploit_env = NeverEndingMaze::new(5);
    let exploit = agent.run_exploit_trial(&mut exploit_env, &bootstrap, 100);
    assert_eq!(exploit.steps, 5);
}

#[test]
fn explore_then_exploit_grows_population_without_panic() {
    let config = Configuration::default_protocol();
    let rng = ChaChaRandomSource::from_seed(0);
    let mut agent = Agent::<4, _>::new(config, rng);

    let actions = agent.config().number_of_possible_actions;
    let selector = EpsilonGreedy {
        number_of_possible_actions: actions,
        epsilon: 0.8,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut env = ToggleMaze::new();

    let mut time: u64 = 0;
    let first = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
    time += first.steps as u64;
    let after_first = agent.population().len();
    assert!(after_first >= 1);

    for _ in 0..49 {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        assert!(metrics.steps >= 1);
        time += metrics.steps as u64;
    }

    assert!(agent.population().len() >= after_first);
    assert!(agent.population().len() >= 2);

    let exploit = agent.run_exploit_trial(&mut env, &bootstrap, time);
    assert!(exploit.steps >= 1);
}

#[test]
fn explore_loop_runs_with_ga_enabled() {
    let mut config = Configuration::default_protocol();
    config.do_ga = true;
    config.theta_ga = 1;
    let rng = ChaChaRandomSource::from_seed(7);
    let mut agent = Agent::<4, _>::new(config, rng);

    let actions = agent.config().number_of_possible_actions;
    let selector = EpsilonGreedy {
        number_of_possible_actions: actions,
        epsilon: 0.5,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut env = ToggleMaze::new();

    let mut time: u64 = 0;
    for _ in 0..30 {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }

    assert!(!agent.population().is_empty());
    let numerosity = agent.population().numerosity();
    assert!(numerosity >= agent.population().len() as u32);
}
