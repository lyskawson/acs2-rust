mod common;

use acs2_core::alp::{apply_alp, expected_case, unexpected_case};
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::config::Configuration;
use acs2_core::effect::Effect;
use acs2_core::ga::{apply_ga, generalizing_mutation, should_apply, two_point_crossover};
use acs2_core::mark::Mark;
use acs2_core::population::Population;
use acs2_core::rl::{apply_reinforcement_learning, update_classifier};
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::symbol::Symbol;

use common::{approx, assert_classifier_matches, classifier, effect, fixtures, mark, perception};

const W: Symbol = Symbol::Wildcard;

fn token(value: u8) -> Symbol {
    Symbol::Token(value)
}

fn scalar(q: f64, r: f64, ir: f64, exp: u32, talp: Option<u64>, tav: f64) -> Classifier<1> {
    Classifier {
        condition: Condition { symbols: [W] },
        action: Some(0),
        effect: Effect { symbols: [W] },
        mark: Mark::new(),
        q,
        r,
        ir,
        num: 1,
        exp,
        talp,
        tga: 0,
        tav,
        ee: false,
    }
}

#[test]
fn updates_qrir_tav_fixtures() {
    for fixture in fixtures("updates_qrir_tav.json") {
        let id = fixture["id"].as_str().unwrap();
        let operation = fixture["operation"].as_str().unwrap();
        let input = &fixture["input"];
        let expected = &fixture["expected"];

        match operation {
            "Classifier.increase_quality" => {
                let mut cl = scalar(input["q"].as_f64().unwrap(), 0.0, 0.0, 1, None, 0.0);
                cl.increase_quality(input["beta"].as_f64().unwrap());
                assert!(approx(cl.q, expected["q"].as_f64().unwrap()), "{id} q");
            }
            "Classifier.decrease_quality" => {
                let mut cl = scalar(input["q"].as_f64().unwrap(), 0.0, 0.0, 1, None, 0.0);
                cl.decrease_quality(input["beta"].as_f64().unwrap());
                assert!(approx(cl.q, expected["q"].as_f64().unwrap()), "{id} q");
            }
            "Classifier.update_application_average" => {
                let mut cl = scalar(
                    0.5,
                    0.0,
                    0.0,
                    input["exp"].as_u64().unwrap() as u32,
                    Some(input["talp"].as_u64().unwrap()),
                    input["tav"].as_f64().unwrap(),
                );
                cl.update_application_average(
                    input["time"].as_u64().unwrap(),
                    input["beta"].as_f64().unwrap(),
                );
                assert!(approx(cl.tav, expected["tav"].as_f64().unwrap()), "{id} tav");
                assert_eq!(cl.talp, Some(expected["talp"].as_u64().unwrap()), "{id} talp");
            }
            "rl.update_classifier" => {
                let mut cl = scalar(
                    0.5,
                    input["r"].as_f64().unwrap(),
                    input["ir"].as_f64().unwrap(),
                    1,
                    None,
                    0.0,
                );
                update_classifier(
                    &mut cl,
                    input["reward"].as_f64().unwrap(),
                    input["max_fitness"].as_f64().unwrap(),
                    input["beta"].as_f64().unwrap(),
                    input["gamma"].as_f64().unwrap(),
                );
                assert!(approx(cl.r, expected["r"].as_f64().unwrap()), "{id} r");
                assert!(approx(cl.ir, expected["ir"].as_f64().unwrap()), "{id} ir");
            }
            other => panic!("unknown operation {other}"),
        }
    }
}

#[test]
fn expected_case_fixtures() {
    let config = Configuration::default_protocol();
    for fixture in fixtures("alp_expected_case.json") {
        let id = fixture["id"].as_str().unwrap();
        let input = &fixture["input"];
        let expected = &fixture["expected"];
        let p0 = perception::<8>(&input["p0"]);
        let time = input["time"].as_u64().unwrap();
        let after = &expected["input_after"];

        if expected["mode"].as_str().unwrap() == "exact" {
            let mut cl = classifier::<8>(&input["classifier"]);
            let mut rng = ChaChaRandomSource::from_seed(0);
            let child = expected_case(&mut cl, &p0, time, &config, &mut rng);
            assert!(child.is_none(), "{id} expected no child");
            assert!(approx(cl.q, after["q"].as_f64().unwrap()), "{id} parent q");
            assert_eq!(cl.mark, mark::<8>(&after["mark"]), "{id} parent mark");
        } else {
            let child_spec = &expected["child"];
            let parent_effect = effect::<8>(&input["classifier"]["effect"]);
            let parent_action = input["classifier"]["action"].as_u64().map(|a| a as usize);
            let allowed_specificity: Vec<usize> = child_spec["allowed_specificity"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_u64().unwrap() as usize)
                .collect();
            for seed in 0..16u64 {
                let mut cl = classifier::<8>(&input["classifier"]);
                let mut rng = ChaChaRandomSource::from_seed(seed);
                let child = expected_case(&mut cl, &p0, time, &config, &mut rng)
                    .unwrap_or_else(|| panic!("{id} seed {seed} expected child"));
                assert_eq!(child.effect, parent_effect, "{id} seed {seed} effect");
                assert_eq!(child.action, parent_action, "{id} seed {seed} action");
                assert!(approx(child.q, 0.5), "{id} seed {seed} child q");
                assert!(!child.is_marked(), "{id} seed {seed} child marked");
                assert!(
                    allowed_specificity.contains(&child.condition.specificity()),
                    "{id} seed {seed} specificity {}",
                    child.condition.specificity()
                );
                assert!(approx(cl.q, after["q"].as_f64().unwrap()), "{id} seed {seed} parent q");
                assert_eq!(cl.mark, mark::<8>(&after["mark"]), "{id} seed {seed} parent mark");
            }
        }
    }
}

#[test]
fn unexpected_case_fixtures() {
    let config = Configuration::default_protocol();
    for fixture in fixtures("alp_unexpected_case.json") {
        let id = fixture["id"].as_str().unwrap();
        let input = &fixture["input"];
        let expected = &fixture["expected"];
        let p0 = perception::<8>(&input["p0"]);
        let p1 = perception::<8>(&input["p1"]);
        let time = input["time"].as_u64().unwrap();

        let mut cl = classifier::<8>(&input["classifier"]);
        let child = unexpected_case(&mut cl, &p0, &p1, time, &config);

        assert_eq!(child.is_some(), expected["returns_child"].as_bool().unwrap(), "{id} child presence");
        if let Some(child) = child {
            assert_classifier_matches(&child, &expected["child"], id);
        }
        let after = &expected["input_after"];
        assert!(approx(cl.q, after["q"].as_f64().unwrap()), "{id} parent q");
        assert_eq!(cl.mark, mark::<8>(&after["mark"]), "{id} parent mark");
    }
}

#[test]
fn learning_step_fixtures() {
    let config = Configuration::default_protocol();
    for fixture in fixtures("learning_step.json") {
        let id = fixture["id"].as_str().unwrap();
        let input = &fixture["input"];
        let p0 = perception::<8>(&input["p0"]);
        let p1 = perception::<8>(&input["p1"]);
        let action = input["action"].as_u64().unwrap() as usize;
        let time = input["time"].as_u64().unwrap();
        let reward = input["reward"].as_f64().unwrap();
        let bootstrap = input["max_fitness"].as_f64().unwrap();

        let stored: Vec<Classifier<8>> = input["population"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| classifier::<8>(value))
            .collect();
        let mut population = Population::from_classifiers(stored);
        let mut match_set = population.form_match_set(&p0);
        let mut action_set = population.form_action_set(&match_set, action);

        let mut rng = ChaChaRandomSource::from_seed(0);
        apply_alp(
            &mut population,
            &mut match_set,
            &mut action_set,
            &p0,
            action,
            &p1,
            time,
            &config,
            &mut rng,
        );
        apply_reinforcement_learning(
            &mut population,
            &action_set,
            reward,
            bootstrap,
            config.beta,
            config.gamma,
        );

        let expected = fixture["expected"]["population_after"].as_array().unwrap();
        assert_eq!(population.len(), expected.len(), "{id} population size");
        for (index, expected_classifier) in expected.iter().enumerate() {
            assert_classifier_matches(population.get(index), expected_classifier, id);
        }
    }
}

#[test]
fn apply_alp_deletes_single_inadequate_victim_and_remaps() {
    let config = Configuration::default_protocol();
    let p0 = perception::<4>(&serde_json::json!(["0", "0", "0", "0"]));
    let p1 = perception::<4>(&serde_json::json!(["1", "0", "0", "0"]));

    let victim = ga_classifier([W, W, W, W], 0, 0.1, 1);
    let mut population = Population::from_classifiers(vec![victim]);
    let mut match_set = population.form_match_set(&p0);
    let mut action_set = population.form_action_set(&match_set, 0);

    let mut rng = ChaChaRandomSource::from_seed(0);
    apply_alp(&mut population, &mut match_set, &mut action_set, &p0, 0, &p1, 5, &config, &mut rng);

    assert!(population.iter().all(|classifier| classifier.does_anticipate_change()));
    assert!(action_set.iter().all(|&reference| reference < population.len()));
    assert!(match_set.iter().all(|&reference| reference < population.len()));
    assert!(!population.is_empty());
}

#[test]
fn apply_alp_deletes_two_inadequate_victims_and_remaps() {
    let config = Configuration::default_protocol();
    let p0 = perception::<4>(&serde_json::json!(["0", "0", "0", "0"]));
    let p1 = perception::<4>(&serde_json::json!(["1", "0", "0", "0"]));

    let first_victim = ga_classifier([W, W, W, W], 0, 0.1, 1);
    let second_victim = ga_classifier([token(b'0'), W, W, W], 0, 0.1, 1);
    let mut population = Population::from_classifiers(vec![first_victim, second_victim]);
    let mut match_set = population.form_match_set(&p0);
    let mut action_set = population.form_action_set(&match_set, 0);
    assert_eq!(action_set.len(), 2);

    let mut rng = ChaChaRandomSource::from_seed(0);
    apply_alp(&mut population, &mut match_set, &mut action_set, &p0, 0, &p1, 5, &config, &mut rng);

    assert!(population.iter().all(|classifier| classifier.does_anticipate_change()));
    assert!(action_set.iter().all(|&reference| reference < population.len()));
    assert!(match_set.iter().all(|&reference| reference < population.len()));
    assert!(!population.is_empty());
}

#[test]
fn ga_should_apply_respects_threshold() {
    let stale = Population::from_classifiers(vec![ga_classifier([token(b'1'), W, W, W], 0, 0.5, 1)]);
    let action_set = vec![0usize];
    assert!(should_apply(&stale, &action_set, 200, 100));
    assert!(!should_apply(&stale, &action_set, 50, 100));
    assert!(!should_apply(&stale, &[], 200, 100));
}

#[test]
fn ga_generalizing_mutation_extremes() {
    let mut always = ga_classifier([token(b'1'), token(b'0'), W, token(b'1')], 0, 0.5, 1);
    let mut rng = ChaChaRandomSource::from_seed(1);
    generalizing_mutation(&mut always, 1.0, &mut rng);
    assert_eq!(always.condition.specificity(), 0);

    let mut never = ga_classifier([token(b'1'), token(b'0'), W, token(b'1')], 0, 0.5, 1);
    let before = never.condition.specificity();
    generalizing_mutation(&mut never, 0.0, &mut rng);
    assert_eq!(never.condition.specificity(), before);
}

#[test]
fn ga_two_point_crossover_preserves_combined_symbols() {
    let mut first = ga_classifier([token(b'1'), token(b'1'), W, W], 0, 0.5, 1);
    let mut second = ga_classifier([W, W, token(b'0'), token(b'0')], 0, 0.5, 1);

    let mut combined_before = combined_symbols(&first, &second);
    combined_before.sort();

    let mut rng = ChaChaRandomSource::from_seed(3);
    two_point_crossover(&mut first, &mut second, &mut rng);

    let mut combined_after = combined_symbols(&first, &second);
    combined_after.sort();

    assert_eq!(combined_before, combined_after);
}

#[test]
fn ga_apply_runs_and_keeps_action_set_bounded() {
    let mut population = Population::from_classifiers(vec![
        ga_classifier([token(b'1'), token(b'0'), W, W], 0, 0.6, 1),
        ga_classifier([token(b'1'), W, token(b'1'), W], 0, 0.7, 1),
    ]);
    let mut match_set = vec![0usize, 1usize];
    let mut action_set = vec![0usize, 1usize];
    let state = perception::<4>(&serde_json::json!(["1", "0", "1", "0"]));
    let config = Configuration::default_protocol();
    let mut rng = ChaChaRandomSource::from_seed(7);

    apply_ga(
        500,
        &mut population,
        &mut match_set,
        &mut action_set,
        &state,
        &config,
        &mut rng,
    );

    let numerosity: u32 = action_set.iter().map(|&reference| population.get(reference).num).sum();
    assert!(numerosity <= config.theta_as);
    assert!(!population.is_empty());
}

fn ga_classifier(condition: [Symbol; 4], action: usize, q: f64, num: u32) -> Classifier<4> {
    Classifier {
        condition: Condition { symbols: condition },
        action: Some(action),
        effect: Effect::all_wildcard(),
        mark: Mark::new(),
        q,
        r: 0.5,
        ir: 0.0,
        num,
        exp: 1,
        talp: Some(0),
        tga: 0,
        tav: 0.0,
        ee: false,
    }
}

fn combined_symbols(first: &Classifier<4>, second: &Classifier<4>) -> Vec<Symbol> {
    let mut symbols = Vec::new();
    for index in 0..4 {
        symbols.push(first.condition.get(index));
        symbols.push(second.condition.get(index));
    }
    symbols
}
