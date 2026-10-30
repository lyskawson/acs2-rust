use crate::compaction::rule::CompactableRule;
use crate::representation::population::Population;
use crate::support::config::Configuration;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct CompactionConfiguration {
    pub theta_exp: u32,
    pub theta_r: f64,
}

impl CompactionConfiguration {
    pub fn new(theta_exp: u32, theta_r: f64) -> Self {
        Self { theta_exp, theta_r }
    }

    pub fn from_learning(config: &Configuration) -> Self {
        Self::new(config.theta_exp, config.theta_r)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterReason {
    Inexperienced,
    NoAnticipatedChange,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Decision {
    Kept,
    Subsumed { by: usize },
    Filtered(FilterReason),
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct CompactionOutcome {
    pub decisions: Vec<Decision>,
}

impl CompactionOutcome {
    pub fn kept_indices(&self) -> Vec<usize> {
        self.decisions
            .iter()
            .enumerate()
            .filter(|(_, decision)| **decision == Decision::Kept)
            .map(|(index, _)| index)
            .collect()
    }

    pub fn compression_rate(&self) -> f64 {
        if self.decisions.is_empty() {
            return 0.0;
        }
        1.0 - self.kept_indices().len() as f64 / self.decisions.len() as f64
    }
}

pub fn is_subsumed<R: CompactableRule>(
    candidate: &R,
    subsumer: &R,
    config: &CompactionConfiguration,
) -> bool {
    subsumer.condition_includes(candidate)
        && subsumer.same_action(candidate)
        && subsumer.effect_includes(candidate)
        && (subsumer.quality() > config.theta_r || subsumer.quality() >= candidate.quality())
        && subsumer.mark_permits(candidate)
}

fn filter_reason<R: CompactableRule>(
    rule: &R,
    config: &CompactionConfiguration,
) -> Option<FilterReason> {
    if rule.action_sequence_len() <= 1 {
        None
    } else if rule.experience() < config.theta_exp {
        Some(FilterReason::Inexperienced)
    } else if !rule.anticipates_change() {
        Some(FilterReason::NoAnticipatedChange)
    } else {
        None
    }
}

pub fn compact<R: CompactableRule>(
    rules: &[R],
    config: &CompactionConfiguration,
) -> CompactionOutcome {
    let mut decisions = vec![Decision::Kept; rules.len()];
    for candidate in 0..rules.len() {
        let subsumer = (0..rules.len()).find(|&other| {
            other != candidate
                && decisions[other] == Decision::Kept
                && is_subsumed(&rules[candidate], &rules[other], config)
        });
        decisions[candidate] = match subsumer {
            Some(by) => Decision::Subsumed { by },
            None => filter_reason(&rules[candidate], config)
                .map_or(Decision::Kept, Decision::Filtered),
        };
    }
    CompactionOutcome { decisions }
}

pub fn compact_population<const N: usize>(
    population: &Population<N>,
    config: &CompactionConfiguration,
) -> (Population<N>, CompactionOutcome) {
    let outcome = compact(population.classifiers(), config);
    let kept = outcome
        .kept_indices()
        .into_iter()
        .map(|index| population.get(index).clone())
        .collect();
    (Population::from_classifiers(kept), outcome)
}

pub fn reliable_mean_specificity<const N: usize>(
    population: &Population<N>,
    theta_r: f64,
) -> Option<f64> {
    let specificities: Vec<f64> = population
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .map(|classifier| classifier.condition.specificity() as f64 / N as f64)
        .collect();
    if specificities.is_empty() {
        return None;
    }
    Some(specificities.iter().sum::<f64>() / specificities.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::classifier::Classifier;
    use crate::representation::perception::Perception;
    use crate::representation::symbol::Symbol;
    use crate::support::knowledge::{anticipation_fraction, Transition};
    use crate::support::rng::{ChaChaRandomSource, RandomSource};

    const THETA_R: f64 = 0.9;

    fn config() -> CompactionConfiguration {
        CompactionConfiguration::new(20, THETA_R)
    }

    fn symbol(byte: u8) -> Symbol {
        if byte == b'#' {
            Symbol::Wildcard
        } else {
            Symbol::Token(byte)
        }
    }

    fn classifier(condition: &[u8; 3], action: usize, effect: &[u8; 3], q: f64) -> Classifier<3> {
        let mut classifier = Classifier::general(Some(action), &Configuration::default_protocol());
        for index in 0..3 {
            classifier.condition.set(index, symbol(condition[index]));
            classifier.effect.set(index, symbol(effect[index]));
        }
        classifier.q = q;
        classifier
    }

    fn perception(text: &[u8; 3]) -> Perception<3> {
        Perception::new(text.map(symbol))
    }

    #[test]
    fn reliable_general_classifier_removes_specific_one() {
        let rules = vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        let outcome = compact(&rules, &config());
        assert_eq!(
            outcome.decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );
        assert_eq!(outcome.compression_rate(), 0.5);
    }

    #[test]
    fn identical_classifiers_keep_the_later_one() {
        let rules = vec![
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        let outcome = compact(&rules, &config());
        assert_eq!(
            outcome.decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );
    }

    #[test]
    fn different_action_or_effect_keeps_both() {
        let different_action = vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 1, b"##1", 0.95),
        ];
        assert_eq!(compact(&different_action, &config()).kept_indices(), vec![0, 1]);

        let different_effect = vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"###", 0.95),
        ];
        assert_eq!(compact(&different_effect, &config()).kept_indices(), vec![0, 1]);
    }

    #[test]
    fn subsumer_needs_reliability_or_no_lower_quality() {
        let weaker = vec![
            classifier(b"10#", 0, b"##1", 0.6),
            classifier(b"1##", 0, b"##1", 0.5),
        ];
        assert_eq!(compact(&weaker, &config()).kept_indices(), vec![0, 1]);

        let unreliable_but_not_weaker = vec![
            classifier(b"10#", 0, b"##1", 0.5),
            classifier(b"1##", 0, b"##1", 0.5),
        ];
        assert_eq!(
            compact(&unreliable_but_not_weaker, &config()).decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );

        let reliable_but_weaker = vec![
            classifier(b"10#", 0, b"##1", 0.99),
            classifier(b"1##", 0, b"##1", 0.95),
        ];
        assert_eq!(
            compact(&reliable_but_weaker, &config()).decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );
    }

    #[test]
    fn marked_subsumer_needs_an_equal_mark() {
        let mut specific = classifier(b"10#", 0, b"##1", 0.95);
        let mut general = classifier(b"1##", 0, b"##1", 0.95);
        general.set_mark(&perception(b"110"));
        assert_eq!(compact(&[specific.clone(), general.clone()], &config()).kept_indices(), vec![0, 1]);

        specific.mark = general.mark.clone();
        assert_eq!(
            compact(&[specific.clone(), general.clone()], &config()).decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );

        let unmarked_general = classifier(b"1##", 0, b"##1", 0.95);
        assert_eq!(
            compact(&[specific, unmarked_general], &config()).decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );
    }

    #[test]
    fn discarded_classifiers_do_not_subsume() {
        let a = classifier(b"###", 0, b"##1", 0.95);
        let b = classifier(b"1##", 0, b"##1", 0.95);
        let c = classifier(b"10#", 0, b"##1", 0.95);

        let general_first = compact(&[a.clone(), b.clone(), c.clone()], &config());
        assert_eq!(general_first.kept_indices(), vec![0]);
        assert_eq!(general_first.decisions[1], Decision::Subsumed { by: 0 });
        assert_eq!(general_first.decisions[2], Decision::Subsumed { by: 0 });

        let general_last = compact(&[c, b, a], &config());
        assert_eq!(general_last.kept_indices(), vec![2]);
        assert_eq!(general_last.decisions[0], Decision::Subsumed { by: 1 });
        assert_eq!(general_last.decisions[1], Decision::Subsumed { by: 2 });
    }

    #[derive(Clone)]
    struct SequenceRule {
        inner: Classifier<3>,
        sequence_len: usize,
    }

    impl SequenceRule {
        fn new(condition: &[u8; 3], effect: &[u8; 3], sequence_len: usize, exp: u32) -> Self {
            let mut rule = classifier(condition, 0, effect, 0.95);
            rule.exp = exp;
            Self {
                inner: rule,
                sequence_len,
            }
        }
    }

    impl CompactableRule for SequenceRule {
        fn condition_includes(&self, other: &Self) -> bool {
            self.inner.condition_includes(&other.inner)
        }
        fn same_action(&self, other: &Self) -> bool {
            self.sequence_len == other.sequence_len
        }
        fn effect_includes(&self, other: &Self) -> bool {
            self.inner.effect_includes(&other.inner)
        }
        fn quality(&self) -> f64 {
            self.inner.q
        }
        fn experience(&self) -> u32 {
            self.inner.exp
        }
        fn mark_permits(&self, other: &Self) -> bool {
            self.inner.mark_permits(&other.inner)
        }
        fn action_sequence_len(&self) -> usize {
            self.sequence_len
        }
        fn anticipates_change(&self) -> bool {
            self.inner.anticipates_change()
        }
    }

    #[test]
    fn filter_removes_only_untested_or_changeless_sequences() {
        let rules = vec![
            SequenceRule::new(b"1##", b"##1", 2, 5),
            SequenceRule::new(b"0##", b"###", 2, 50),
            SequenceRule::new(b"#1#", b"##1", 2, 50),
            SequenceRule::new(b"#0#", b"###", 1, 0),
        ];
        let outcome = compact(&rules, &config());
        assert_eq!(
            outcome.decisions,
            vec![
                Decision::Filtered(FilterReason::Inexperienced),
                Decision::Filtered(FilterReason::NoAnticipatedChange),
                Decision::Kept,
                Decision::Kept,
            ]
        );
    }

    #[test]
    fn subsumption_takes_precedence_over_filtering() {
        let rules = vec![
            SequenceRule::new(b"10#", b"##1", 2, 5),
            SequenceRule::new(b"1##", b"##1", 2, 50),
        ];
        assert_eq!(
            compact(&rules, &config()).decisions,
            vec![Decision::Subsumed { by: 1 }, Decision::Kept]
        );
    }

    #[test]
    fn compact_population_leaves_input_untouched_and_is_deterministic() {
        let population = Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"0##", 1, b"###", 0.4),
        ]);
        let (first, first_outcome) = compact_population(&population, &config());
        let (second, second_outcome) = compact_population(&population, &config());
        assert_eq!(population.len(), 3);
        assert_eq!(first.len(), 2);
        assert_eq!(first.classifiers(), second.classifiers());
        assert_eq!(first_outcome, second_outcome);
    }

    #[test]
    fn reliable_mean_specificity_averages_reliable_conditions() {
        let population = Population::from_classifiers(vec![
            classifier(b"10#", 0, b"##1", 0.95),
            classifier(b"1##", 0, b"##1", 0.95),
            classifier(b"101", 1, b"###", 0.4),
        ]);
        assert_eq!(reliable_mean_specificity(&population, THETA_R), Some(0.5));
        assert_eq!(reliable_mean_specificity(&Population::<3>::new(), THETA_R), None);
    }

    fn random_symbol(rng: &mut dyn RandomSource, wildcard_probability: f64) -> Symbol {
        if rng.gen_bool(wildcard_probability) {
            Symbol::Wildcard
        } else {
            Symbol::Token(b'0' + rng.gen_range(2) as u8)
        }
    }

    fn random_perception(rng: &mut dyn RandomSource) -> Perception<3> {
        Perception::new(core::array::from_fn(|_| Symbol::Token(b'0' + rng.gen_range(2) as u8)))
    }

    fn random_classifier(rng: &mut dyn RandomSource) -> Classifier<3> {
        let mut classifier =
            Classifier::general(Some(rng.gen_range(2)), &Configuration::default_protocol());
        for index in 0..3 {
            classifier.condition.set(index, random_symbol(rng, 0.5));
            classifier.effect.set(index, random_symbol(rng, 0.7));
        }
        classifier.q = if rng.gen_bool(0.5) {
            0.9 + 0.1 * rng.gen_unit()
        } else {
            rng.gen_unit()
        };
        classifier.exp = rng.gen_range(50) as u32;
        if rng.gen_bool(0.3) {
            let marked = random_perception(rng);
            classifier.set_mark(&marked);
        }
        classifier
    }

    fn all_transitions() -> Vec<Transition<3>> {
        let perceptions: Vec<Perception<3>> = (0..8u8)
            .map(|bits| {
                Perception::new(core::array::from_fn(|index| {
                    Symbol::Token(b'0' + ((bits >> index) & 1))
                }))
            })
            .collect();
        let mut transitions = Vec::new();
        for p0 in &perceptions {
            for action in 0..2 {
                for p1 in &perceptions {
                    transitions.push(Transition::new(*p0, action, *p1));
                }
            }
        }
        transitions
    }

    fn final_survivor(outcome: &CompactionOutcome, mut index: usize) -> usize {
        let mut hops = 0;
        while let Decision::Subsumed { by } = outcome.decisions[index] {
            hops += 1;
            assert!(hops <= outcome.decisions.len());
            index = by;
        }
        index
    }

    #[test]
    fn random_populations_keep_their_knowledge() {
        let mut rng = ChaChaRandomSource::from_seed(7);
        for _ in 0..300 {
            let size = 1 + rng.gen_range(40);
            let classifiers: Vec<Classifier<3>> =
                (0..size).map(|_| random_classifier(&mut rng)).collect();
            let population = Population::from_classifiers(classifiers);
            let (compacted, outcome) = compact_population(&population, &config());

            assert_eq!(
                anticipation_fraction(&population, THETA_R, all_transitions()),
                anticipation_fraction(&compacted, THETA_R, all_transitions())
            );

            for (index, decision) in outcome.decisions.iter().enumerate() {
                assert_ne!(*decision, Decision::Filtered(FilterReason::Inexperienced));
                assert_ne!(*decision, Decision::Filtered(FilterReason::NoAnticipatedChange));
                let survivor = final_survivor(&outcome, index);
                assert_eq!(outcome.decisions[survivor], Decision::Kept);
                let removed = population.get(index);
                let kept = population.get(survivor);
                assert!(kept.condition.includes(&removed.condition));
                assert_eq!(kept.action, removed.action);
                assert_eq!(kept.effect, removed.effect);
                if removed.is_reliable(THETA_R) {
                    assert!(kept.is_reliable(THETA_R));
                }
            }
        }
    }
}
