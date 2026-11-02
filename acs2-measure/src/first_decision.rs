use acs2_core::representation::{perception::Perception, population::Population, symbol::Symbol};
use serde_json::{json, Value};

pub fn profile<const S: usize, const G: usize, const M: usize>(
    population: &Population<M>,
    state: &Perception<M>,
    action: usize,
) -> Value {
    assert_eq!(S + G, M);
    let candidates: Vec<_> = population
        .classifiers()
        .iter()
        .filter(|classifier| classifier.does_match(state) && classifier.does_anticipate_change())
        .collect();
    let best = candidates
        .iter()
        .map(|classifier| classifier.fitness() * f64::from(classifier.num))
        .reduce(f64::max);
    let winners: Vec<_> = candidates.iter()
        .filter(|classifier| classifier.action == Some(action)
            && Some(classifier.fitness() * f64::from(classifier.num)) == best)
        .map(|classifier| json!({
            "specified_state_attributes": classifier.condition.symbols[..S].iter().filter(|&&symbol| symbol != Symbol::Wildcard).count(),
            "specified_goal_attributes": classifier.condition.symbols[S..].iter().filter(|&&symbol| symbol != Symbol::Wildcard).count(),
            "q": classifier.q, "r": classifier.r, "fitness": classifier.fitness(),
            "experience": classifier.exp, "numerosity": classifier.num,
            "selection_score": classifier.fitness() * f64::from(classifier.num),
        })).collect();
    assert!(
        best.is_none() || !winners.is_empty(),
        "chosen action must have a highest-scoring rule"
    );
    json!({"random_fallback": best.is_none(), "winners": winners})
}
