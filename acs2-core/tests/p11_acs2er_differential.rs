mod common;

use acs2_core::acs2er::{Acs2ErAgent, ReplayConfiguration};
use acs2_core::action_selection::ActionSelector;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::population::{ClassifierRef, Population};
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;

use common::{assert_classifier_matches, load};

const LENGTH: usize = 4;

struct CorridorEnv {
    state: usize,
    elapsed: u32,
    goal_state: usize,
    max_episode_steps: u32,
    goal_reward: f64,
}

impl CorridorEnv {
    fn encode(state: usize) -> Perception<LENGTH> {
        Perception {
            symbols: core::array::from_fn(|index| {
                let bit = (state >> (LENGTH - 1 - index)) & 1;
                Symbol::Token(b'0' + bit as u8)
            }),
        }
    }
}

impl Environment<LENGTH> for CorridorEnv {
    fn reset(&mut self) -> Perception<LENGTH> {
        self.state = 0;
        self.elapsed = 0;
        Self::encode(self.state)
    }

    fn step(&mut self, action: usize) -> StepOutcome<LENGTH> {
        self.elapsed += 1;
        if action == 1 {
            self.state = (self.state + 1).min(15);
        } else {
            self.state = self.state.saturating_sub(1);
        }

        let terminated = self.state == self.goal_state;
        let truncated = self.elapsed >= self.max_episode_steps;
        StepOutcome {
            observation: Self::encode(self.state),
            reward: if terminated { self.goal_reward } else { 0.0 },
            terminated,
            truncated,
            info: (),
        }
    }
}

struct ScriptedSelector {
    actions: Vec<usize>,
    cursor: std::cell::Cell<usize>,
}

impl ActionSelector<LENGTH> for ScriptedSelector {
    fn select(
        &self,
        _population: &Population<LENGTH>,
        _match_set: &[ClassifierRef],
        _rng: &mut dyn RandomSource,
    ) -> usize {
        let index = self.cursor.get();
        self.cursor.set(index + 1);
        self.actions[index]
    }
}

struct ReplayedRandom {
    draws: Vec<(usize, usize)>,
    cursor: std::rc::Rc<std::cell::Cell<usize>>,
}

impl RandomSource for ReplayedRandom {
    fn gen_bool(&mut self, _probability: f64) -> bool {
        panic!("the gated ACS2ER path must not draw booleans");
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        let index = self.cursor.get();
        let (expected_bound, value) = self.draws[index];
        assert_eq!(
            bound, expected_bound,
            "rng draw {index} asked for bound {bound}, pyalcs drew from {expected_bound}"
        );
        self.cursor.set(index + 1);
        value
    }

    fn gen_unit(&mut self) -> f64 {
        panic!("the gated ACS2ER path must not draw unit floats");
    }
}

fn canonical_key(classifier: &Classifier<LENGTH>) -> (String, usize, String) {
    let render = |symbols: &[Symbol; LENGTH]| -> String {
        symbols
            .iter()
            .map(|symbol| match symbol {
                Symbol::Wildcard => '#',
                Symbol::Token(value) => *value as char,
            })
            .collect()
    };
    (
        render(&classifier.condition.symbols),
        classifier.action.unwrap(),
        render(&classifier.effect.symbols),
    )
}

#[test]
fn acs2er_explore_run_matches_pyalcs() {
    let data = load("acs2er_differential.json");

    assert_eq!(
        data["alp_deletions"].as_u64().unwrap(),
        0,
        "fixture contains ALP deletions, which trigger the pyalcs mid-iteration \
         skip documented in docs/ARCHITECTURE.md; the exact-match gate is invalid"
    );

    let fixture_config = &data["config"];
    assert_eq!(fixture_config["classifier_length"].as_u64().unwrap() as usize, LENGTH);
    assert!(!fixture_config["do_ga"].as_bool().unwrap());

    let mut config = Configuration::default_protocol();
    config.number_of_possible_actions =
        fixture_config["number_of_possible_actions"].as_u64().unwrap() as usize;
    config.beta = fixture_config["beta"].as_f64().unwrap();
    config.gamma = fixture_config["gamma"].as_f64().unwrap();
    config.theta_i = fixture_config["theta_i"].as_f64().unwrap();
    config.theta_r = fixture_config["theta_r"].as_f64().unwrap();
    config.theta_exp = fixture_config["theta_exp"].as_u64().unwrap() as u32;
    config.theta_as = fixture_config["theta_as"].as_u64().unwrap() as u32;
    config.u_max = fixture_config["u_max"].as_u64().unwrap() as u32;
    config.initial_q = fixture_config["initial_q"].as_f64().unwrap();
    config.do_ga = fixture_config["do_ga"].as_bool().unwrap();
    config.do_subsumption = fixture_config["do_subsumption"].as_bool().unwrap();

    let replay_config = ReplayConfiguration {
        buffer_size: fixture_config["er_buffer_size"].as_u64().unwrap() as usize,
        min_samples: fixture_config["er_min_samples"].as_u64().unwrap() as usize,
        samples_number: fixture_config["er_samples_number"].as_u64().unwrap() as usize,
    };

    let environment = &data["environment"];
    let mut env = CorridorEnv {
        state: 0,
        elapsed: 0,
        goal_state: environment["goal_state"].as_u64().unwrap() as usize,
        max_episode_steps: environment["max_episode_steps"].as_u64().unwrap() as u32,
        goal_reward: environment["goal_reward"].as_f64().unwrap(),
    };

    let selector = ScriptedSelector {
        actions: data["actions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect(),
        cursor: std::cell::Cell::new(0),
    };

    let mut draws: Vec<(usize, usize)> = Vec::new();
    for event in data["rng_events"].as_array().unwrap() {
        let bound = event["bound"].as_u64().unwrap() as usize;
        for value in event["values"].as_array().unwrap() {
            draws.push((bound, value.as_u64().unwrap() as usize));
        }
    }
    let expected_draws = draws.len();

    let rng_cursor = std::rc::Rc::new(std::cell::Cell::new(0usize));
    let mut agent = Acs2ErAgent::<LENGTH, _>::new(
        config,
        replay_config,
        ReplayedRandom {
            draws,
            cursor: std::rc::Rc::clone(&rng_cursor),
        },
    );
    let bootstrap = MaxFitnessBootstrap;

    let expected_trial_steps: Vec<u32> = data["trial_steps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u32)
        .collect();

    let mut time: u64 = 0;
    let mut trial_steps: Vec<u32> = Vec::new();
    for _ in 0..data["trials"].as_u64().unwrap() {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
        trial_steps.push(metrics.steps);
    }

    assert_eq!(trial_steps, expected_trial_steps, "per-trial step counts");
    assert_eq!(time, data["total_steps"].as_u64().unwrap(), "total steps");
    assert_eq!(
        selector.cursor.get(),
        data["actions"].as_array().unwrap().len(),
        "action script consumed"
    );
    assert_eq!(
        agent.replay_memory().len(),
        data["replay_memory_size"].as_u64().unwrap() as usize,
        "replay memory size"
    );

    let expected_population = data["population_after"].as_array().unwrap();
    assert_eq!(
        agent.population().len(),
        expected_population.len(),
        "population size"
    );

    let mut sorted: Vec<&Classifier<LENGTH>> = agent.population().iter().collect();
    sorted.sort_by(|left, right| canonical_key(left).cmp(&canonical_key(right)));

    for (index, expected) in expected_population.iter().enumerate() {
        assert_classifier_matches(sorted[index], expected, &format!("classifier[{index}]"));
    }

    assert_eq!(
        rng_cursor.get(),
        expected_draws,
        "rust drew {} random values, pyalcs drew {expected_draws}",
        rng_cursor.get()
    );
}
