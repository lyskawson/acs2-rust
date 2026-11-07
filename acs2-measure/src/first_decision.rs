use acs2_core::representation::{perception::Perception, population::Population, symbol::Symbol};
use serde_json::{json, Value};

pub fn profile<const S: usize, const G: usize, const M: usize>(
    population: &Population<M>,
    state: &Perception<M>,
    action: usize,
) -> Value {
    profile_with_rewards::<S, G, M>(population, state, action, |reference| population.get(reference).r)
}

pub fn profile_with_rewards<const S: usize, const G: usize, const M: usize>(
    population: &Population<M>,
    state: &Perception<M>,
    action: usize,
    reward: impl Fn(usize) -> f64,
) -> Value {
    assert_eq!(S + G, M);
    let candidates: Vec<_> = population
        .classifiers()
        .iter()
        .enumerate()
        .filter(|(_, classifier)| classifier.does_match(state) && classifier.does_anticipate_change())
        .collect();
    let best = candidates
        .iter()
        .map(|(reference, classifier)| classifier.q * reward(*reference) * f64::from(classifier.num))
        .reduce(f64::max);
    let winners: Vec<_> = candidates.iter()
        .filter(|(reference, classifier)| classifier.action == Some(action)
            && Some(classifier.q * reward(*reference) * f64::from(classifier.num)) == best)
        .map(|(reference, classifier)| json!({
            "specified_state_attributes": classifier.condition.symbols[..S].iter().filter(|&&symbol| symbol != Symbol::Wildcard).count(),
            "specified_goal_attributes": classifier.condition.symbols[S..].iter().filter(|&&symbol| symbol != Symbol::Wildcard).count(),
            "q": classifier.q, "r": reward(*reference), "fitness": classifier.q * reward(*reference),
            "experience": classifier.exp, "numerosity": classifier.num,
            "selection_score": classifier.q * reward(*reference) * f64::from(classifier.num),
        })).collect();
    assert!(
        best.is_none() || !winners.is_empty(),
        "chosen action must have a highest-scoring rule"
    );
    json!({"random_fallback": best.is_none(), "winners": winners})
}
