mod common;

use std::collections::BTreeSet;

use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::subsumption::{does_subsume, is_subsumer};
use acs2_core::symbol::Symbol;
use serde_json::Value;

use common::{classifier, condition, effect, fixtures, mark, perception, symbols};

#[test]
fn condition_matching_fixtures() {
    fn run<const N: usize>(operation: &str, left: &Value, right: &Value) -> bool {
        let cond = condition::<N>(left);
        match operation {
            "Condition.does_match" => cond.does_match(&perception::<N>(right)),
            "Condition.subsumes" => cond.subsumes(&condition::<N>(right)),
            other => panic!("unknown operation {other}"),
        }
    }

    for fixture in fixtures("condition_matching.json") {
        let id = fixture["id"].as_str().unwrap();
        let operation = fixture["operation"].as_str().unwrap();
        let left = &fixture["input"]["condition"];
        let right = &fixture["input"]["other"];
        let expected = fixture["expected"]["result"].as_bool().unwrap();
        let got = match left.as_array().unwrap().len() {
            4 => run::<4>(operation, left, right),
            8 => run::<8>(operation, left, right),
            length => panic!("unsupported length {length} in {id}"),
        };
        assert_eq!(got, expected, "fixture {id}");
    }
}

#[test]
fn effect_anticipation_fixtures() {
    for fixture in fixtures("effect_anticipation.json") {
        let id = fixture["id"].as_str().unwrap();
        let operation = fixture["operation"].as_str().unwrap();
        let eff = effect::<4>(&fixture["input"]["effect"]);
        let p0 = perception::<4>(&fixture["input"]["p0"]);
        let p1 = perception::<4>(&fixture["input"]["p1"]);
        let expected = fixture["expected"]["result"].as_bool().unwrap();
        let got = match operation {
            "Effect.anticipates_correctly" => eff.anticipates_correctly(&p0, &p1),
            "Effect.is_specializable" => eff.is_specializable(&p0, &p1),
            other => panic!("unknown operation {other}"),
        };
        assert_eq!(got, expected, "fixture {id}");
    }
}

#[test]
fn mark_differences_fixtures() {
    for fixture in fixtures("mark_differences.json") {
        let id = fixture["id"].as_str().unwrap();
        let pmark = mark::<8>(&fixture["input"]["mark"]);
        let p0 = perception::<8>(&fixture["input"]["p0"]);
        let expected = &fixture["expected"];
        match expected["mode"].as_str().unwrap() {
            "exact" => {
                let want = symbols::<8>(&expected["diff"]);
                for seed in 0..4u64 {
                    let mut rng = ChaChaRandomSource::from_seed(seed);
                    let diff = pmark.get_differences(&p0, &mut rng);
                    assert_eq!(diff.symbols, want, "fixture {id} seed {seed}");
                }
            }
            "property" => {
                let allowed_indices: BTreeSet<usize> = expected["allowed_specialized_indices"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect();
                let allowed_specificity: BTreeSet<usize> = expected["allowed_specificity"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_u64().unwrap() as usize)
                    .collect();
                for seed in 0..16u64 {
                    let mut rng = ChaChaRandomSource::from_seed(seed);
                    let diff = pmark.get_differences(&p0, &mut rng);
                    let specialized: Vec<usize> =
                        (0..8).filter(|&index| !diff.symbols[index].is_wildcard()).collect();
                    assert!(
                        allowed_specificity.contains(&specialized.len()),
                        "fixture {id} seed {seed}: specificity {}",
                        specialized.len()
                    );
                    assert_eq!(specialized.len(), 1, "fixture {id} seed {seed}");
                    let chosen = specialized[0];
                    assert!(
                        allowed_indices.contains(&chosen),
                        "fixture {id} seed {seed}: index {chosen} not allowed"
                    );
                    assert_eq!(
                        diff.symbols[chosen], p0.symbols[chosen],
                        "fixture {id} seed {seed}: specialized value differs from p0"
                    );
                }
            }
            other => panic!("unknown mode {other}"),
        }
    }
}

#[test]
fn subsumption_fixtures() {
    for fixture in fixtures("subsumption.json") {
        let id = fixture["id"].as_str().unwrap();
        let operation = fixture["operation"].as_str().unwrap();
        let input = &fixture["input"];
        let theta_exp = input["theta_exp"].as_u64().unwrap() as u32;
        let theta_r = input["theta_r"].as_f64().unwrap();
        let cl = classifier::<8>(&input["cl"]);
        let expected = fixture["expected"]["result"].as_bool().unwrap();
        let got = match operation {
            "subsumption.is_subsumer" => is_subsumer(&cl, theta_exp, theta_r),
            "subsumption.does_subsume" => {
                does_subsume(&cl, &classifier::<8>(&input["other"]), theta_exp, theta_r)
            }
            other => panic!("unknown operation {other}"),
        };
        assert_eq!(got, expected, "fixture {id}");
    }
}

#[test]
fn does_subsume_symmetric_wildcard_superset_swallows_more_specific() {
    let general = Classifier::<4> {
        condition: Condition {
            symbols: [
                Symbol::Token(b'1'),
                Symbol::Token(b'1'),
                Symbol::Wildcard,
                Symbol::Wildcard,
            ],
        },
        action: Some(0),
        effect: Effect::all_wildcard(),
        mark: Mark::new(),
        q: 0.95,
        r: 1.0,
        ir: 0.0,
        num: 1,
        exp: 30,
        talp: None,
        tga: 0,
        tav: 0.0,
        ee: false,
    };
    let specific = Classifier::<4> {
        condition: Condition {
            symbols: [
                Symbol::Token(b'1'),
                Symbol::Wildcard,
                Symbol::Token(b'9'),
                Symbol::Token(b'9'),
            ],
        },
        action: Some(0),
        effect: Effect::all_wildcard(),
        mark: Mark::new(),
        q: 0.95,
        r: 1.0,
        ir: 0.0,
        num: 1,
        exp: 30,
        talp: None,
        tga: 0,
        tav: 0.0,
        ee: false,
    };
    assert!(does_subsume(&general, &specific, 20, 0.9));
    assert!(!does_subsume(&specific, &general, 20, 0.9));
}

#[test]
fn classifier_identity_fixtures() {
    for fixture in fixtures("numerosity_collapse.json") {
        if fixture["operation"].as_str().unwrap() != "Classifier.__eq__" {
            continue;
        }
        let id = fixture["id"].as_str().unwrap();
        let cl = classifier::<8>(&fixture["input"]["cl"]);
        let other = classifier::<8>(&fixture["input"]["other"]);
        let expected = fixture["expected"]["result"].as_bool().unwrap();
        assert_eq!(cl == other, expected, "fixture {id}");
    }
}
