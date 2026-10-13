use acs2_core::learning::replay::{replay_learning_step, ReplaySample};
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::representation::{perception::Perception, population::Population};
use acs2_core::support::{config::Configuration, rng::RandomSource};
use serde_json::Value;

#[path = "../../../acs2-core/tests/common/mod.rs"]
mod common;

struct ValidationRandom {
    draws: Vec<(usize, usize)>,
    cursor: usize,
}
impl ValidationRandom {
    fn from_json(value: &Value) -> Self {
        Self {
            draws: value
                .as_array()
                .unwrap()
                .iter()
                .map(|draw| {
                    (
                        draw[0].as_u64().unwrap() as usize,
                        draw[1].as_u64().unwrap() as usize,
                    )
                })
                .collect(),
            cursor: 0,
        }
    }
}
impl RandomSource for ValidationRandom {
    fn gen_bool(&mut self, _: f64) -> bool {
        panic!("unexpected boolean draw")
    }
    fn gen_unit(&mut self) -> f64 {
        panic!("unexpected unit draw")
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        let (expected, position) = self.draws[self.cursor];
        assert_eq!(bound, expected);
        self.cursor += 1;
        position
    }
}

struct PyalcsValidation;
impl PyalcsValidation {
    fn samples(case: &Value, rng: &mut ValidationRandom) -> Vec<ReplaySample<8>> {
        let trial = case["trial"].as_array().unwrap();
        let index = case["transition"].as_u64().unwrap() as usize;
        let chosen = &trial[index];
        let state = common::perception::<4>(&chosen["state"]);
        let following = common::perception::<4>(&chosen["next_state"]);
        let join = |state: Perception<4>, goal: Perception<4>| Perception {
            symbols: core::array::from_fn(|i| {
                if i < 4 {
                    state.symbols[i]
                } else {
                    goal.symbols[i - 4]
                }
            }),
        };
        let sample = |goal, reward, done| ReplaySample {
            state: join(state, goal),
            next_state: join(following, goal),
            action: chosen["action"].as_u64().unwrap() as usize,
            reward,
            done,
        };
        let mut samples = vec![sample(
            common::perception::<4>(&case["real_goal"]),
            chosen["reward"].as_f64().unwrap(),
            chosen["done"].as_bool().unwrap(),
        )];
        let strategy = case["strategy"].as_str().unwrap();
        let (first, bound) = match strategy {
            "future" => (index, trial.len() - index),
            "episode" => (0, trial.len()),
            "final" => (trial.len() - 1, 1),
            _ => unreachable!(),
        };
        let count = if strategy == "final" {
            1
        } else {
            bound.min(case["goals_number"].as_u64().unwrap() as usize)
        };
        let mut positions = Vec::new();
        for _ in 0..count {
            let position = if strategy == "final" {
                0
            } else {
                rng.gen_range(bound)
            };
            assert!(!positions.contains(&position));
            positions.push(position);
            let goal = common::perception::<4>(&trial[first + position]["next_state"]);
            samples.push(sample(
                goal,
                if following == goal { 1.0 } else { 0.0 },
                false,
            ));
        }
        samples
    }

    fn learn(
        population: &mut Population<8>,
        config: &Configuration,
        sample: &ReplaySample<8>,
        time: u64,
        rng: &mut ValidationRandom,
    ) {
        replay_learning_step(population, config, &MaxFitnessBootstrap, sample, time, rng);
    }
}

#[test]
fn pyalcs_relabeling_and_random_replay_steps_match_complete_populations() {
    let data = common::load("her_differential.json");
    assert_eq!(data["coverage"]["deletions"], 0);
    assert_eq!(data["cases"].as_array().unwrap().len(), 1000);
    for name in [
        "original",
        "relabeled",
        "reached_copy",
        "unreached_copy",
        "expected",
        "unexpected",
        "covering",
        "rng_choice",
    ] {
        assert!(
            data["coverage"][name].as_u64().unwrap() > 0,
            "missing {name}"
        );
    }
    let parameters = &data["config"];
    let config = Configuration {
        number_of_possible_actions: 8,
        beta: parameters["beta"].as_f64().unwrap(),
        gamma: parameters["gamma"].as_f64().unwrap(),
        theta_i: 0.0,
        theta_r: parameters["theta_r"].as_f64().unwrap(),
        theta_exp: parameters["theta_exp"].as_u64().unwrap() as u32,
        theta_as: parameters["theta_as"].as_u64().unwrap() as u32,
        initial_q: parameters["initial_q"].as_f64().unwrap(),
        u_max: 100000,
        do_ga: false,
        do_pee: false,
        ..Configuration::default_protocol()
    };
    for (index, case) in data["cases"].as_array().unwrap().iter().enumerate() {
        let mut goals_rng = ValidationRandom::from_json(&case["goal_draws"]);
        let samples = PyalcsValidation::samples(case, &mut goals_rng);
        assert_eq!(samples.len(), case["samples"].as_array().unwrap().len());
        for (actual, expected) in samples.iter().zip(case["samples"].as_array().unwrap()) {
            assert_eq!(actual.state, common::perception::<8>(&expected["state"]));
            assert_eq!(
                actual.next_state,
                common::perception::<8>(&expected["next_state"])
            );
            assert_eq!(actual.action, expected["action"].as_u64().unwrap() as usize);
            assert_eq!(
                actual.reward.to_bits(),
                expected["reward"].as_f64().unwrap().to_bits()
            );
            assert_eq!(actual.done, expected["done"].as_bool().unwrap());
        }
        assert_eq!(goals_rng.cursor, goals_rng.draws.len());
        let mut population = Population::from_classifiers(
            case["population_before"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| {
                    let mut cl = common::classifier::<8>(value);
                    cl.ee = value["ee"].as_bool().unwrap();
                    cl
                })
                .collect(),
        );
        let mut learning_rng = ValidationRandom::from_json(&case["learning_draws"]);
        PyalcsValidation::learn(
            &mut population,
            &config,
            &samples[case["selected"].as_u64().unwrap() as usize],
            case["time"].as_u64().unwrap() + case["steps"].as_u64().unwrap(),
            &mut learning_rng,
        );
        let mut actual = population.classifiers().to_vec();
        actual.sort_by_key(|cl| {
            (
                format!("{:?}", cl.condition),
                cl.action,
                format!("{:?}", cl.effect),
            )
        });
        let mut expected = case["population_after"].as_array().unwrap().clone();
        expected.sort_by_key(|value| {
            let cl = common::classifier::<8>(value);
            (
                format!("{:?}", cl.condition),
                cl.action,
                format!("{:?}", cl.effect),
            )
        });
        assert_eq!(actual.len(), expected.len(), "case {index}");
        for (cl, value) in actual.iter().zip(expected) {
            common::assert_classifier_matches(cl, &value, &format!("case {index}"));
            assert_eq!(cl.ee, value["ee"].as_bool().unwrap());
        }
        assert_eq!(
            learning_rng.cursor,
            learning_rng.draws.len(),
            "case {index}: unconsumed randomness"
        );
    }
}
