use acs2_core::learning::alp::apply_alp;
use acs2_core::learning::ga::apply_ga;
use acs2_core::learning::replay::ReplaySample;
use acs2_core::learning::rl::{update_classifier, BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::population::Population;
use acs2_core::representation::symbol::Symbol;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::SampleOrigin;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RouteCounts {
    pub draws: u64,
    pub with_goal: u64,
    pub without_goal: u64,
    pub skipped_without_goal: u64,
    pub draws_with_goal: u64,
    pub action_set_rules: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RewardUpdateCounts {
    pub original: RouteCounts,
    pub relabeled: RouteCounts,
}

impl RewardUpdateCounts {
    pub(crate) fn observe(&mut self, origin: SampleOrigin, value: RouteCounts) {
        let route = match origin {
            SampleOrigin::Original => &mut self.original,
            SampleOrigin::Relabeled => &mut self.relabeled,
        };
        route.draws += value.draws;
        route.with_goal += value.with_goal;
        route.without_goal += value.without_goal;
        route.skipped_without_goal += value.skipped_without_goal;
        route.draws_with_goal += value.draws_with_goal;
        route.action_set_rules += value.action_set_rules;
    }
}

pub fn specifies_goal<const S: usize, const M: usize>(classifier: &Classifier<M>) -> bool {
    classifier.condition.symbols[S..]
        .iter()
        .any(|symbol| *symbol != Symbol::Wildcard)
}

pub(crate) fn learning_step<const S: usize, const M: usize>(
    population: &mut Population<M>,
    config: &Configuration,
    sample: &ReplaySample<M>,
    time: u64,
    rng: &mut dyn RandomSource,
    keep: impl Fn(&Classifier<M>) -> bool,
) -> RouteCounts {
    let matched = population.form_match_set(&sample.state);
    let mut action_set = population.form_action_set(&matched, sample.action);
    let mut next_matched = population.form_match_set(&sample.next_state);
    apply_alp(
        population,
        &mut next_matched,
        &mut action_set,
        &sample.state,
        sample.action,
        &sample.next_state,
        time,
        config,
        rng,
    );
    next_matched = population.form_match_set(&sample.next_state);
    let bootstrap = if sample.done {
        0.0
    } else {
        MaxFitnessBootstrap.estimate(population, &next_matched)
    };
    let mut counts = RouteCounts {
        draws: 1,
        action_set_rules: action_set.len() as u64,
        ..Default::default()
    };
    for &reference in &action_set {
        let classifier = population.get_mut(reference);
        let goal = specifies_goal::<S, M>(classifier);
        if goal {
            counts.draws_with_goal = 1;
        }
        if keep(classifier) {
            update_classifier(
                classifier,
                sample.reward,
                bootstrap,
                config.beta,
                config.gamma,
            );
            if goal {
                counts.with_goal += 1;
            } else {
                counts.without_goal += 1;
            }
        } else {
            counts.skipped_without_goal += 1;
        }
    }
    if config.do_ga {
        let mut ga_matched = if sample.done {
            Vec::new()
        } else {
            next_matched
        };
        apply_ga(
            time,
            population,
            &mut ga_matched,
            &mut action_set,
            &sample.next_state,
            config,
            rng,
        );
    }
    counts
}

#[cfg(test)]
mod tests;
