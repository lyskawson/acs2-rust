mod common;

use acs2_core::action_selection::{ActionSelector, BestAction, EpsilonGreedy, RandomAction};
use acs2_core::alp::{add_classifier, cover};
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::config::Configuration;
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::population::Population;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;

use common::{assert_classifier_matches, classifier, fixtures, perception};

const W: Symbol = Symbol::Wildcard;

fn token(value: u8) -> Symbol {
    Symbol::Token(value)
}

fn build(
    condition: [Symbol; 4],
    action: usize,
    effect: [Symbol; 4],
    q: f64,
    r: f64,
    num: u32,
) -> Classifier<4> {
    Classifier {
        condition: Condition { symbols: condition },
        action: Some(action),
        effect: Effect { symbols: effect },
        mark: Mark::new(),
        q,
        r,
        ir: 0.0,
        num,
        exp: 1,
        talp: None,
        tga: 0,
        tav: 0.0,
        ee: false,
    }
}

#[test]
fn covering_fixtures() {
    fn run<const N: usize>(input: &serde_json::Value, expected: &serde_json::Value, id: &str) {
        let p0 = perception::<N>(&input["p0"]);
        let p1 = perception::<N>(&input["p1"]);
        let action = input["action"].as_u64().unwrap() as usize;
        let time = input["time"].as_u64().unwrap();
        let config = Configuration::default_protocol();
        let classifier = cover(&p0, action, &p1, time, &config);
        assert_classifier_matches(&classifier, &expected["classifier"], id);
    }

    for fixture in fixtures("alp_covering.json") {
        let id = fixture["id"].as_str().unwrap();
        let input = &fixture["input"];
        let expected = &fixture["expected"];
        match input["p0"].as_array().unwrap().len() {
            4 => run::<4>(input, expected, id),
            8 => run::<8>(input, expected, id),
            length => panic!("unsupported length {length} in {id}"),
        }
    }
}

#[test]
fn add_classifier_fixtures() {
    let config = Configuration::default_protocol();
    for fixture in fixtures("numerosity_collapse.json") {
        if fixture["operation"].as_str().unwrap() != "alp.add_classifier" {
            continue;
        }
        let id = fixture["id"].as_str().unwrap();
        let input = &fixture["input"];

        let stored: Vec<Classifier<8>> = input["population"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| classifier::<8>(value))
            .collect();
        let mut population = Population::from_classifiers(stored);
        let mut new_list: Vec<Classifier<8>> = input["new_list"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| classifier::<8>(value))
            .collect();
        let child = classifier::<8>(&input["child"]);

        let search_set: Vec<usize> = (0..population.len()).collect();
        add_classifier(child, &mut population, &search_set, &mut new_list, &config);

        let expected = &fixture["expected"];
        assert_eq!(
            new_list.len(),
            expected["new_list_size"].as_u64().unwrap() as usize,
            "{id} new_list_size"
        );
        assert_eq!(
            population.len(),
            expected["population_size"].as_u64().unwrap() as usize,
            "{id} population_size"
        );
        for (index, expected_classifier) in expected["population"].as_array().unwrap().iter().enumerate() {
            assert_classifier_matches(population.get(index), expected_classifier, id);
        }
    }
}

#[test]
fn add_classifier_population_subsumer_absorbs_child() {
    let config = Configuration::default_protocol();
    let mut subsumer = build([W, W, W, token(b'0')], 3, [W, W, token(b'1'), W], 0.95, 1.0, 1);
    subsumer.exp = 30;
    let child = build([token(b'1'), W, W, token(b'0')], 3, [W, W, token(b'1'), W], 0.5, 1.0, 1);

    let mut population = Population::from_classifiers(vec![subsumer]);
    let mut new_list: Vec<Classifier<4>> = Vec::new();

    let search_set: Vec<usize> = (0..population.len()).collect();
    add_classifier(child, &mut population, &search_set, &mut new_list, &config);

    assert!(new_list.is_empty());
    assert_eq!(population.len(), 1);
    assert!(common::approx(population.get(0).q, 0.9525));
    assert_eq!(population.get(0).num, 1);
}

#[test]
fn add_classifier_new_list_identity_increases_quality() {
    let config = Configuration::default_protocol();
    let existing = build([token(b'1'), W, W, W], 2, [W, token(b'0'), W, W], 0.5, 0.5, 1);
    let duplicate = build([token(b'1'), W, W, W], 2, [W, token(b'0'), W, W], 0.5, 0.5, 1);

    let mut population = Population::<4>::new();
    let mut new_list = vec![existing];

    let search_set: Vec<usize> = (0..population.len()).collect();
    add_classifier(duplicate, &mut population, &search_set, &mut new_list, &config);

    assert!(population.is_empty());
    assert_eq!(new_list.len(), 1);
    assert!(common::approx(new_list[0].q, 0.525));
}

fn sample_population() -> Population<4> {
    Population::from_classifiers(vec![
        build([token(b'1'), W, W, W], 0, [W, W, W, token(b'0')], 0.95, 1.0, 2),
        build([token(b'1'), W, W, W], 1, [W, token(b'0'), W, W], 0.5, 0.5, 1),
        build([token(b'0'), W, W, W], 0, [W, W, W, W], 0.8, 0.5, 1),
    ])
}

#[test]
fn match_and_action_set_formation() {
    let population = sample_population();
    let state = perception::<4>(&serde_json::json!(["1", "0", "1", "0"]));

    let match_set = population.form_match_set(&state);
    assert_eq!(match_set, vec![0, 1]);

    let action_set = population.form_action_set(&match_set, 0);
    assert_eq!(action_set, vec![0]);

    assert!(common::approx(population.get_maximum_fitness(&match_set), 0.95));
    assert_eq!(population.numerosity(), 4);
    assert_eq!(population.reliable_count(0.9), 1);
}

#[test]
fn maximum_fitness_ignores_non_change_anticipating() {
    let population = sample_population();
    let only_passthrough = vec![2usize];
    assert_eq!(population.get_maximum_fitness(&only_passthrough), 0.0);
}

#[test]
fn best_action_selects_highest_fitness_times_numerosity() {
    let population = sample_population();
    let state = perception::<4>(&serde_json::json!(["1", "0", "1", "0"]));
    let match_set = population.form_match_set(&state);
    let selector = BestAction {
        number_of_possible_actions: 8,
    };
    for seed in 0..8u64 {
        let mut rng = ChaChaRandomSource::from_seed(seed);
        assert_eq!(selector.select(&population, &match_set, &mut rng), 0);
    }
}

#[test]
fn best_action_falls_back_to_random_when_no_change_anticipated() {
    let population = sample_population();
    let passthrough_only = vec![2usize];
    let selector = BestAction {
        number_of_possible_actions: 8,
    };
    for seed in 0..16u64 {
        let mut rng = ChaChaRandomSource::from_seed(seed);
        let action = selector.select(&population, &passthrough_only, &mut rng);
        assert!(action < 8);
    }
}

#[test]
fn random_action_stays_in_range() {
    let population = sample_population();
    let state = perception::<4>(&serde_json::json!(["1", "0", "1", "0"]));
    let match_set = population.form_match_set(&state);
    let selector = RandomAction {
        number_of_possible_actions: 8,
    };
    for seed in 0..16u64 {
        let mut rng = ChaChaRandomSource::from_seed(seed);
        assert!(selector.select(&population, &match_set, &mut rng) < 8);
    }
}

#[test]
fn epsilon_greedy_degenerates_to_best_and_random() {
    let population = sample_population();
    let state = perception::<4>(&serde_json::json!(["1", "0", "1", "0"]));
    let match_set = population.form_match_set(&state);

    let exploit = EpsilonGreedy {
        number_of_possible_actions: 8,
        epsilon: 0.0,
    };
    let explore = EpsilonGreedy {
        number_of_possible_actions: 8,
        epsilon: 1.0,
    };
    for seed in 0..8u64 {
        let mut rng = ChaChaRandomSource::from_seed(seed);
        assert_eq!(exploit.select(&population, &match_set, &mut rng), 0);
        let mut rng = ChaChaRandomSource::from_seed(seed);
        assert!(explore.select(&population, &match_set, &mut rng) < 8);
    }
}
