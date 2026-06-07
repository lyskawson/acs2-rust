#![allow(dead_code)]

use std::collections::BTreeSet;

use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::perception::Perception;
use acs2_core::symbol::Symbol;
use serde_json::Value;

pub fn load(file: &str) -> Value {
    let path = format!("{}/../fixtures/{}", env!("CARGO_MANIFEST_DIR"), file);
    let text = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&text).unwrap()
}

pub fn fixtures(file: &str) -> Vec<Value> {
    load(file)["fixtures"].as_array().unwrap().clone()
}

pub fn symbol(value: &str) -> Symbol {
    if value == "#" {
        Symbol::Wildcard
    } else {
        Symbol::Token(value.as_bytes()[0])
    }
}

pub fn symbols<const N: usize>(value: &Value) -> [Symbol; N] {
    let array = value.as_array().unwrap();
    assert_eq!(array.len(), N);
    core::array::from_fn(|index| symbol(array[index].as_str().unwrap()))
}

pub fn condition<const N: usize>(value: &Value) -> Condition<N> {
    Condition {
        symbols: symbols::<N>(value),
    }
}

pub fn effect<const N: usize>(value: &Value) -> Effect<N> {
    Effect {
        symbols: symbols::<N>(value),
    }
}

pub fn perception<const N: usize>(value: &Value) -> Perception<N> {
    Perception {
        symbols: symbols::<N>(value),
    }
}

pub fn mark<const N: usize>(value: &Value) -> Mark<N> {
    let array = value.as_array().unwrap();
    assert_eq!(array.len(), N);
    let attributes = core::array::from_fn(|index| {
        array[index]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| symbol(item.as_str().unwrap()))
            .collect::<BTreeSet<Symbol>>()
    });
    Mark { attributes }
}

pub fn optional_usize(value: &Value) -> Option<usize> {
    match value {
        Value::Null => None,
        other => Some(other.as_u64().unwrap() as usize),
    }
}

pub fn optional_u64(value: &Value) -> Option<u64> {
    match value {
        Value::Null => None,
        other => Some(other.as_u64().unwrap()),
    }
}

pub fn classifier<const N: usize>(value: &Value) -> Classifier<N> {
    Classifier {
        condition: condition::<N>(&value["condition"]),
        action: optional_usize(&value["action"]),
        effect: effect::<N>(&value["effect"]),
        mark: mark::<N>(&value["mark"]),
        q: value["q"].as_f64().unwrap(),
        r: value["r"].as_f64().unwrap(),
        ir: value["ir"].as_f64().unwrap(),
        num: value["num"].as_u64().unwrap() as u32,
        exp: value["exp"].as_u64().unwrap() as u32,
        talp: optional_u64(&value["talp"]),
        tga: value["tga"].as_u64().unwrap(),
        tav: value["tav"].as_f64().unwrap(),
        ee: false,
    }
}

pub fn approx(left: f64, right: f64) -> bool {
    (left - right).abs() <= 1e-9
}

pub fn assert_classifier_matches<const N: usize>(
    actual: &Classifier<N>,
    expected: &Value,
    id: &str,
) {
    assert_eq!(actual.condition, condition::<N>(&expected["condition"]), "{id} condition");
    assert_eq!(actual.action, optional_usize(&expected["action"]), "{id} action");
    assert_eq!(actual.effect, effect::<N>(&expected["effect"]), "{id} effect");
    assert_eq!(actual.mark, mark::<N>(&expected["mark"]), "{id} mark");
    assert!(approx(actual.q, expected["q"].as_f64().unwrap()), "{id} q");
    assert!(approx(actual.r, expected["r"].as_f64().unwrap()), "{id} r");
    assert!(approx(actual.ir, expected["ir"].as_f64().unwrap()), "{id} ir");
    assert_eq!(actual.num, expected["num"].as_u64().unwrap() as u32, "{id} num");
    assert_eq!(actual.exp, expected["exp"].as_u64().unwrap() as u32, "{id} exp");
    assert_eq!(actual.talp, optional_u64(&expected["talp"]), "{id} talp");
    assert_eq!(actual.tga, expected["tga"].as_u64().unwrap(), "{id} tga");
    assert!(approx(actual.tav, expected["tav"].as_f64().unwrap()), "{id} tav");
}
