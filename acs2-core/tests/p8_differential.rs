mod common;

use acs2_core::alp::apply_alp;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::population::{ClassifierRef, Population};
use acs2_core::rl::apply_reinforcement_learning;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;

use common::{approx, assert_classifier_matches, classifier, load, perception};

fn symbols_to_string<const N: usize>(symbols: &[Symbol; N]) -> String {
    symbols
        .iter()
        .map(|symbol| match symbol {
            Symbol::Wildcard => '#',
            Symbol::Token(value) => *value as char,
        })
        .collect()
}

fn canonical_key(classifier: &Classifier<8>) -> (String, usize, String) {
    (
        symbols_to_string(&classifier.condition.symbols),
        classifier.action.unwrap(),
        symbols_to_string(&classifier.effect.symbols),
    )
}

fn index_list(value: &serde_json::Value) -> Vec<ClassifierRef> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|item| item.as_u64().unwrap() as ClassifierRef)
        .collect()
}

#[test]
fn deterministic_learning_step_matches_pyalcs() {
    let config = Configuration::default_protocol();
    let data = load("differential_cases.json");

    let kept = data["deterministic_kept"].as_u64().unwrap();
    let cases = data["cases"].as_array().unwrap();
    assert_eq!(kept as usize, cases.len());

    for case in cases {
        let id = case["id"].as_str().unwrap();
        let input = &case["input"];
        let expected = &case["expected"];

        let p0 = perception::<8>(&input["p0"]);
        let p1 = perception::<8>(&input["p1"]);
        let action = input["action"].as_u64().unwrap() as usize;
        let time = input["time"].as_u64().unwrap();
        let reward = input["reward"].as_f64().unwrap();
        let bootstrap = input["bootstrap"].as_f64().unwrap();

        let stored: Vec<Classifier<8>> = input["population"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| classifier::<8>(value))
            .collect();
        let mut population = Population::from_classifiers(stored);

        let match_set = population.form_match_set(&p0);
        assert_eq!(match_set, index_list(&expected["match_set"]), "{id} match_set");

        let action_set = population.form_action_set(&match_set, action);
        assert_eq!(action_set, index_list(&expected["action_set"]), "{id} action_set");

        let mut match_set_next = population.form_match_set(&p1);
        assert_eq!(
            match_set_next,
            index_list(&expected["match_set_next"]),
            "{id} match_set_next"
        );

        let computed_bootstrap = population.get_maximum_fitness(&match_set_next);
        assert!(
            approx(computed_bootstrap, bootstrap),
            "{id} bootstrap: rust={computed_bootstrap} pyalcs={bootstrap}"
        );

        let mut prior_action_set = action_set.clone();
        let mut rng = ChaChaRandomSource::from_seed(0);
        apply_alp(
            &mut population,
            &mut match_set_next,
            &mut prior_action_set,
            &p0,
            action,
            &p1,
            time,
            &config,
            &mut rng,
        );
        apply_reinforcement_learning(
            &mut population,
            &prior_action_set,
            reward,
            bootstrap,
            config.beta,
            config.gamma,
        );

        let expected_population = expected["population_after"].as_array().unwrap();
        assert_eq!(
            population.len(),
            expected_population.len(),
            "{id} population_after size"
        );

        let mut sorted: Vec<&Classifier<8>> = population.iter().collect();
        sorted.sort_by(|left, right| canonical_key(left).cmp(&canonical_key(right)));

        for (index, expected_classifier) in expected_population.iter().enumerate() {
            assert_classifier_matches(sorted[index], expected_classifier, id);
        }
    }
}

#[test]
fn rust_processes_whole_action_set_where_pyalcs_skips_after_deletion() {
    let config = Configuration::default_protocol();
    let p0 = perception::<8>(&serde_json::json!(["0", "0", "0", "0", "0", "0", "0", "0"]));
    let p1 = p0;

    let inadequate = classifier::<8>(&serde_json::json!({
        "condition": ["0", "#", "#", "#", "#", "#", "#", "#"], "action": 0,
        "effect": ["1", "#", "#", "#", "#", "#", "#", "#"],
        "mark": [[], [], [], [], [], [], [], []],
        "q": 0.05, "r": 0.5, "ir": 0.0, "num": 1, "exp": 5,
        "talp": 0, "tga": 0, "tav": 0.0
    }));
    let second = classifier::<8>(&serde_json::json!({
        "condition": ["0", "0", "#", "#", "#", "#", "#", "#"], "action": 0,
        "effect": ["1", "#", "#", "#", "#", "#", "#", "#"],
        "mark": [[], [], [], [], [], [], [], []],
        "q": 0.5, "r": 0.5, "ir": 0.0, "num": 1, "exp": 5,
        "talp": 0, "tga": 0, "tav": 0.0
    }));

    let mut population = Population::from_classifiers(vec![inadequate, second]);
    let match_set = population.form_match_set(&p0);
    let action_set = population.form_action_set(&match_set, 0);
    assert_eq!(action_set, vec![0, 1]);

    let mut match_set_next = population.form_match_set(&p1);
    let mut prior_action_set = action_set;
    let mut rng = ChaChaRandomSource::from_seed(0);
    apply_alp(
        &mut population,
        &mut match_set_next,
        &mut prior_action_set,
        &p0,
        0,
        &p1,
        10,
        &config,
        &mut rng,
    );

    let surviving_second = population
        .iter()
        .find(|cl| symbols_to_string(&cl.condition.symbols) == "00######")
        .expect("second action-set classifier must survive");
    assert_eq!(surviving_second.exp, 6, "second classifier must be processed (exp incremented)");
    assert!(
        surviving_second.mark.is_marked(),
        "second classifier must be marked; pyalcs skips it via mid-iteration deletion"
    );
    assert!(
        population
            .iter()
            .all(|cl| symbols_to_string(&cl.condition.symbols) != "0#######"),
        "inadequate first classifier must be deleted"
    );
}
