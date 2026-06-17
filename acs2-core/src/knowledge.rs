use crate::classifier::Classifier;
use crate::perception::Perception;
use crate::population::Population;

pub struct Transition<const N: usize> {
    pub p0: Perception<N>,
    pub action: usize,
    pub p1: Perception<N>,
}

impl<const N: usize> Transition<N> {
    pub fn new(p0: Perception<N>, action: usize, p1: Perception<N>) -> Self {
        Self { p0, action, p1 }
    }
}

fn predicts_successfully<const N: usize>(
    classifier: &Classifier<N>,
    transition: &Transition<N>,
) -> bool {
    classifier.action == Some(transition.action)
        && classifier.does_match(&transition.p0)
        && classifier.does_anticipate_correctly(&transition.p0, &transition.p1)
}

pub fn anticipation_fraction<const N: usize, I>(
    population: &Population<N>,
    theta_r: f64,
    transitions: I,
) -> f64
where
    I: IntoIterator<Item = Transition<N>>,
{
    let reliable: Vec<&Classifier<N>> = population
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .collect();

    let mut total = 0usize;
    let mut correctly_anticipated = 0usize;
    for transition in transitions {
        total += 1;
        if reliable
            .iter()
            .any(|classifier| predicts_successfully(classifier, &transition))
        {
            correctly_anticipated += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }
    correctly_anticipated as f64 / total as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Configuration;
    use crate::effect::Effect;
    use crate::symbol::Symbol;

    const VALIDATION: usize = 2;

    fn token(value: u8) -> Symbol {
        Symbol::Token(b'0' + value)
    }

    fn perception(input_bit: u8, validation_bit: u8) -> Perception<3> {
        Perception::new([token(input_bit), token(input_bit), token(validation_bit)])
    }

    fn reliable_classifier(
        condition_input_bit: u8,
        action: usize,
        anticipates_change: bool,
    ) -> Classifier<3> {
        let mut classifier = Classifier::general(Some(action), &Configuration::mpx());
        classifier.condition.set(0, token(condition_input_bit));
        classifier.condition.set(1, token(condition_input_bit));
        if anticipates_change {
            classifier.effect.set(VALIDATION, token(1));
        }
        classifier.q = 1.0;
        classifier
    }

    fn toy_transitions() -> Vec<Transition<3>> {
        let mut transitions = Vec::new();
        for input_bit in [0u8, 1u8] {
            let correct_action = input_bit as usize;
            for action in [0usize, 1usize] {
                let p0 = perception(input_bit, 0);
                let validation_after = if action == correct_action { 1 } else { 0 };
                let p1 = perception(input_bit, validation_after);
                transitions.push(Transition::new(p0, action, p1));
            }
        }
        transitions
    }

    #[test]
    fn wildcard_effect_credits_identity_transition() {
        let no_change_effect: Effect<3> = Effect::all_wildcard();
        let p0 = perception(1, 0);
        assert!(no_change_effect.anticipates_correctly(&p0, &p0));

        let p1 = perception(1, 1);
        assert!(!no_change_effect.anticipates_correctly(&p0, &p1));
    }

    #[test]
    fn complete_population_reaches_full_knowledge() {
        let mut classifiers = Vec::new();
        for input_bit in [0u8, 1u8] {
            let correct_action = input_bit as usize;
            let wrong_action = 1 - correct_action;
            classifiers.push(reliable_classifier(input_bit, correct_action, true));
            classifiers.push(reliable_classifier(input_bit, wrong_action, false));
        }
        let population = Population::from_classifiers(classifiers);

        let knowledge = anticipation_fraction(&population, 0.9, toy_transitions());
        assert_eq!(knowledge, 1.0);
    }

    #[test]
    fn change_only_population_caps_at_half() {
        let mut classifiers = Vec::new();
        for input_bit in [0u8, 1u8] {
            let correct_action = input_bit as usize;
            classifiers.push(reliable_classifier(input_bit, correct_action, true));
        }
        let population = Population::from_classifiers(classifiers);

        let knowledge = anticipation_fraction(&population, 0.9, toy_transitions());
        assert_eq!(knowledge, 0.5);
    }

    #[test]
    fn unreliable_classifiers_are_ignored() {
        let mut change = reliable_classifier(0, 0, true);
        change.q = 0.5;
        let population = Population::from_classifiers(vec![change]);

        let knowledge = anticipation_fraction(&population, 0.9, toy_transitions());
        assert_eq!(knowledge, 0.0);
    }
}
