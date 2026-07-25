use crate::classifier::Classifier;
use crate::condition::Condition;
use crate::config::{AlpGenVariant, Configuration};
use crate::perception::Perception;
use crate::population::{remap_after_removal, ClassifierRef, Population};
use crate::rng::RandomSource;
use crate::subsumption::does_subsume;

pub fn cover<const N: usize>(
    p0: &Perception<N>,
    action: usize,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
) -> Classifier<N> {
    let mut classifier = Classifier::covering(action, time, config);
    classifier.specialize(p0, p1, false);
    classifier
}

pub fn add_classifier<const N: usize>(
    child: Classifier<N>,
    population: &mut Population<N>,
    search_set: &[ClassifierRef],
    new_list: &mut Vec<Classifier<N>>,
    config: &Configuration,
) {
    let mut subsumer: Option<ClassifierRef> = None;
    for &reference in search_set {
        if does_subsume(population.get(reference), &child, config.theta_exp, config.theta_r) {
            let replaces = match subsumer {
                None => true,
                Some(current) => population.get(reference).is_more_general(population.get(current)),
            };
            if replaces {
                subsumer = Some(reference);
            }
        }
    }

    if let Some(reference) = subsumer {
        population.get_mut(reference).increase_quality(config.beta);
        return;
    }

    if let Some(existing) = new_list.iter_mut().find(|classifier| **classifier == child) {
        existing.increase_quality(config.beta);
        return;
    }

    for &reference in search_set {
        if *population.get(reference) == child {
            population.get_mut(reference).increase_quality(config.beta);
            return;
        }
    }

    new_list.push(child);
}

pub fn expected_case<const N: usize>(
    classifier: &mut Classifier<N>,
    p0: &Perception<N>,
    time: u64,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) -> Option<Classifier<N>> {
    let mut difference = classifier.mark.get_differences(p0, rng);

    if difference.specificity() == 0 {
        classifier.increase_quality(config.beta);
        return None;
    }

    let mut child = Classifier::copy_from(classifier, time);

    match config.alp_gen_variant {
        AlpGenVariant::Pyalcs => {
            generalize_over_specification_pyalcs(classifier, &mut difference, config.u_max, rng)
        }
        AlpGenVariant::Butz => {
            generalize_over_specification_butz(&mut child, &mut difference, config.u_max, rng)
        }
    }

    child.condition.specialize_with(&difference);
    if child.q < 0.5 {
        child.q = 0.5;
    }
    Some(child)
}

fn generalize_over_specification_pyalcs<const N: usize>(
    parent: &mut Classifier<N>,
    difference: &mut Condition<N>,
    u_max: u32,
    rng: &mut dyn RandomSource,
) {
    let u_max = u_max as i64;
    let mut no_spec = parent.specified_unchanging_attributes().len() as i64;
    let mut no_spec_new = difference.specificity() as i64;

    if no_spec >= u_max {
        while no_spec >= u_max {
            let generalized = parent.generalize_unchanging_condition_attribute(rng);
            debug_assert!(generalized);
            no_spec -= 1;
        }
        while no_spec + no_spec_new > u_max {
            if rng.gen_unit() < 0.5 {
                difference.generalize_specific_attribute_randomly(rng);
                no_spec_new -= 1;
            } else if parent.generalize_unchanging_condition_attribute(rng) {
                no_spec -= 1;
            }
        }
    } else {
        while no_spec + no_spec_new > u_max {
            difference.generalize_specific_attribute_randomly(rng);
            no_spec_new -= 1;
        }
    }
}

fn generalize_over_specification_butz<const N: usize>(
    child: &mut Classifier<N>,
    difference: &mut Condition<N>,
    u_max: u32,
    rng: &mut dyn RandomSource,
) {
    let u_max = u_max as i64;
    let mut spec = child.condition.specificity() as i64;
    let mut spec_new = difference.specificity() as i64;

    if spec >= u_max {
        child.condition.generalize_specific_attribute_randomly(rng);
        spec -= 1;
        while spec + spec_new > u_max {
            if rng.gen_unit() < 0.5 {
                child.condition.generalize_specific_attribute_randomly(rng);
                spec -= 1;
            } else {
                difference.generalize_specific_attribute_randomly(rng);
                spec_new -= 1;
            }
        }
    } else {
        while spec + spec_new > u_max {
            difference.generalize_specific_attribute_randomly(rng);
            spec_new -= 1;
        }
    }
}

pub fn unexpected_case<const N: usize>(
    classifier: &mut Classifier<N>,
    p0: &Perception<N>,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
) -> Option<Classifier<N>> {
    classifier.decrease_quality(config.beta);
    classifier.set_mark(p0);

    if !classifier.effect.is_specializable(p0, p1) {
        return None;
    }

    let mut child = Classifier::copy_from(classifier, time);
    child.specialize(p0, p1, !config.do_pee);
    if child.q < 0.5 {
        child.q = 0.5;
    }
    Some(child)
}

pub fn apply_alp<const N: usize>(
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    p0: &Perception<N>,
    action: usize,
    p1: &Perception<N>,
    time: u64,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    let original_action_set = action_set.clone();
    let mut new_list: Vec<Classifier<N>> = Vec::new();
    let mut was_expected_case = false;
    let mut victims: Vec<ClassifierRef> = Vec::new();

    for &reference in &original_action_set {
        let anticipates_correctly;
        let child;
        {
            let classifier = population.get_mut(reference);
            classifier.increase_experience();
            classifier.update_application_average(time, config.beta);
            anticipates_correctly = classifier.does_anticipate_correctly(p0, p1);
            child = if anticipates_correctly {
                expected_case(classifier, p0, time, config, rng)
            } else {
                unexpected_case(classifier, p0, p1, time, config)
            };
        }

        if anticipates_correctly {
            was_expected_case = true;
        } else if population.get(reference).is_inadequate(config.theta_i) {
            victims.push(reference);
        }

        if let Some(mut new_classifier) = child {
            new_classifier.tga = time;
            add_classifier(new_classifier, population, &original_action_set, &mut new_list, config);
        }
    }

    if !was_expected_case {
        let covering = cover(p0, action, p1, time, config);
        add_classifier(covering, population, &original_action_set, &mut new_list, config);
    }

    for new_classifier in new_list {
        let matches_next = new_classifier.does_match(p1);
        let reference = population.insert(new_classifier);
        action_set.push(reference);
        if matches_next {
            match_set.push(reference);
        }
    }

    if !victims.is_empty() {
        victims.sort_unstable();
        victims.dedup();
        remap_after_removal(match_set, &victims);
        remap_after_removal(action_set, &victims);
        population.remove_many(&victims);
    }
}
