use acs2_core::learning::alp::{add_classifier, cover, expected_case, unexpected_case};
use acs2_core::learning::replay::ReplaySample;
use acs2_core::representation::population::{remap_after_removal, ClassifierRef, Population};
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;

use super::PerGoalRewards;

pub(super) struct AlpSets {
    pub action_set: Vec<ClassifierRef>,
    pub next_matched: Vec<ClassifierRef>,
}

pub(super) fn apply_alp<const G: usize, const M: usize>(
    rewards: &mut PerGoalRewards<G>,
    population: &mut Population<M>,
    sets: &mut AlpSets,
    sample: &ReplaySample<M>,
    time: u64,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    let AlpSets {
        next_matched,
        action_set,
    } = sets;
    let original_action_set = action_set.clone();
    let mut new_list = Vec::new();
    let mut new_values = Vec::new();
    let mut was_expected_case = false;
    let mut victims = Vec::new();
    for &reference in &original_action_set {
        let classifier = population.get_mut(reference);
        classifier.increase_experience();
        classifier.update_application_average(time, config.beta);
        let expected = classifier.does_anticipate_correctly(&sample.state, &sample.next_state);
        let child = if expected {
            expected_case(classifier, &sample.state, time, config, rng)
        } else {
            unexpected_case(classifier, &sample.state, &sample.next_state, time, config)
        };
        if expected {
            was_expected_case = true;
        } else if classifier.is_inadequate(config.theta_i) {
            victims.push(reference);
        }
        if let Some(mut child) = child {
            child.tga = time;
            let previous = new_list.len();
            add_classifier(
                child,
                population,
                &original_action_set,
                &mut new_list,
                config,
            );
            if new_list.len() > previous {
                let mut values = rewards.entries[reference].clone();
                values.iter_mut().for_each(|entry| entry.updated = false);
                new_values.push(values);
            }
        }
    }
    if !was_expected_case {
        let child = cover(
            &sample.state,
            sample.action,
            &sample.next_state,
            time,
            config,
        );
        let previous = new_list.len();
        add_classifier(
            child,
            population,
            &original_action_set,
            &mut new_list,
            config,
        );
        if new_list.len() > previous {
            new_values.push(rewards.start_values(0.0, config.initial_ir));
        }
    }
    assert_eq!(new_list.len(), new_values.len());
    for (classifier, values) in new_list.into_iter().zip(new_values) {
        let matches_next = classifier.does_match(&sample.next_state);
        let reference = population.insert(classifier);
        rewards.entries.push(values);
        action_set.push(reference);
        if matches_next {
            next_matched.push(reference);
        }
    }
    if !victims.is_empty() {
        victims.sort_unstable();
        victims.dedup();
        remap_after_removal(next_matched, &victims);
        remap_after_removal(action_set, &victims);
        population.remove_many(&victims);
        for &reference in victims.iter().rev() {
            rewards.entries.remove(reference);
        }
    }
    assert_eq!(population.len(), rewards.entries.len());
}
