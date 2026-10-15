use super::priority::{integer_power, population_variance, EPSILON};
use acs2_core::learning::replay::{replay_learning_step, ReplaySample};
use acs2_core::learning::rl::MaxFitnessBootstrap;
use acs2_core::representation::population::Population;
use acs2_core::support::{config::Configuration, rng::RandomSource};
use serde_json::Value;

use crate::differential_common as common;

struct DifferentialRandom {
    draws: Vec<(usize, usize)>,
    cursor: usize,
}
impl RandomSource for DifferentialRandom {
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

struct PyalcsPriority {
    maximum: f64,
    temperature: u32,
}
impl PyalcsPriority {
    fn priority(&mut self, populations: &[Population<8>], sample: &ReplaySample<8>) -> (f64, f64) {
        let values: Vec<_> = populations
            .iter()
            .map(|population| {
                let matched = population.form_match_set(&sample.state);
                let action_set = population.form_action_set(&matched, sample.action);
                action_set
                    .iter()
                    .map(|&index| population.get(index).fitness())
                    .fold(0.0, f64::max)
            })
            .collect();
        let variance = population_variance(&values);
        self.maximum = self.maximum.max(variance);
        (
            variance,
            integer_power(self.maximum - variance + EPSILON, self.temperature),
        )
    }
}

fn close(actual: f64, expected: &Value) {
    assert!(
        (actual - expected.as_f64().unwrap()).abs() <= 1e-9,
        "{actual} vs {expected}"
    );
}

#[test]
fn reference_priority_probabilities_and_one_learning_round_match_instrumented_v10() {
    let data = common::load("vcp_differential.json");
    assert_eq!(data["cases"].as_array().unwrap().len(), 1000);
    assert_eq!(data["coverage"]["deletions"], 0);
    for name in [
        "expected",
        "unexpected",
        "covering",
        "no_change_maximum",
        "duplicate_samples",
        "unchanged_priorities",
        "rng_choice",
    ] {
        assert!(
            data["coverage"][name].as_u64().unwrap() > 0,
            "missing {name}"
        );
    }
    let config = Configuration {
        number_of_possible_actions: 8,
        beta: data["config"]["beta"].as_f64().unwrap(),
        gamma: data["config"]["gamma"].as_f64().unwrap(),
        theta_i: 0.0,
        theta_r: data["config"]["theta_r"].as_f64().unwrap(),
        theta_exp: data["config"]["theta_exp"].as_u64().unwrap() as u32,
        theta_as: data["config"]["theta_as"].as_u64().unwrap() as u32,
        initial_q: data["config"]["initial_q"].as_f64().unwrap(),
        u_max: 100000,
        do_ga: false,
        do_pee: false,
        ..Configuration::default_protocol()
    };
    for (index, case) in data["cases"].as_array().unwrap().iter().enumerate() {
        let samples: Vec<_> = case["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| ReplaySample {
                state: common::perception::<8>(&value["state"]),
                action: value["action"].as_u64().unwrap() as usize,
                next_state: common::perception::<8>(&value["next_state"]),
                reward: value["reward"].as_f64().unwrap(),
                done: value["done"].as_bool().unwrap(),
            })
            .collect();
        let mut populations: Vec<_> = case["populations_before"]
            .as_array()
            .unwrap()
            .iter()
            .map(|values| {
                Population::from_classifiers(
                    values
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|value| {
                            let mut classifier = common::classifier::<8>(value);
                            classifier.ee = value["ee"].as_bool().unwrap();
                            classifier
                        })
                        .collect(),
                )
            })
            .collect();
        let mut oracle = PyalcsPriority {
            maximum: case["initial_maximum"].as_f64().unwrap(),
            temperature: case["temperature"].as_u64().unwrap() as u32,
        };
        let mut sigmas = vec![];
        let mut priorities = vec![];
        for (position, sample) in samples.iter().enumerate() {
            let (sigma, priority) = oracle.priority(&populations, sample);
            close(sigma, &case["sigmas_before"][position]);
            close(priority, &case["priorities_before"][position]);
            sigmas.push(sigma);
            priorities.push(priority);
        }
        close(oracle.maximum, &case["maximum_before"]);
        let sum: f64 = priorities.iter().sum();
        for (position, priority) in priorities.iter().enumerate() {
            close(priority / sum, &case["probabilities"][position]);
        }
        let mut rng = DifferentialRandom {
            draws: case["learning_draws"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| {
                    (
                        value[0].as_u64().unwrap() as usize,
                        value[1].as_u64().unwrap() as usize,
                    )
                })
                .collect(),
            cursor: 0,
        };
        let mut updated = vec![];
        for (head, indices) in case["learning_indices"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            for position in indices.as_array().unwrap() {
                let position = position.as_u64().unwrap() as usize;
                if !updated
                    .iter()
                    .any(|&prior| samples[prior] == samples[position])
                {
                    updated.push(position);
                }
                replay_learning_step(
                    &mut populations[head],
                    &config,
                    &MaxFitnessBootstrap,
                    &samples[position],
                    case["time"].as_u64().unwrap(),
                    &mut rng,
                );
            }
        }
        assert_eq!(rng.cursor, rng.draws.len(), "case {index}: random tape");
        assert_eq!(updated.len(), case["updates"].as_array().unwrap().len());
        for (order, &position) in updated.iter().enumerate() {
            let (sigma, priority) = oracle.priority(&populations, &samples[position]);
            assert_eq!(case["updates"][order]["index"], position);
            close(sigma, &case["updates"][order]["sigma"]);
            close(priority, &case["updates"][order]["priority"]);
            sigmas[position] = sigma;
            priorities[position] = priority;
        }
        close(oracle.maximum, &case["maximum_after"]);
        for (position, (&sigma, &priority)) in sigmas.iter().zip(&priorities).enumerate() {
            close(sigma, &case["sigmas_after"][position]);
            close(priority, &case["priorities_after"][position]);
        }
        for (head, population) in populations.iter().enumerate() {
            let mut actual = population.classifiers().to_vec();
            let mut expected = case["populations_after"][head].as_array().unwrap().clone();
            actual.sort_by_key(|cl| {
                (
                    format!("{:?}", cl.condition),
                    cl.action,
                    format!("{:?}", cl.effect),
                )
            });
            expected.sort_by_key(|value| {
                let cl = common::classifier::<8>(value);
                (
                    format!("{:?}", cl.condition),
                    cl.action,
                    format!("{:?}", cl.effect),
                )
            });
            assert_eq!(actual.len(), expected.len(), "case {index}, head {head}");
            for (classifier, value) in actual.iter().zip(expected) {
                common::assert_classifier_matches(
                    classifier,
                    &value,
                    &format!("case {index}, head {head}"),
                );
                assert_eq!(classifier.ee, value["ee"].as_bool().unwrap());
            }
        }
    }
}
