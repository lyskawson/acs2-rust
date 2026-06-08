use crate::classifier::Classifier;
use crate::config::Configuration;
use crate::perception::Perception;
use crate::population::{remap_after_removal, ClassifierRef, Population};
use crate::rng::RandomSource;
use crate::subsumption::does_subsume;

pub fn should_apply<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    time: u64,
    theta_ga: u32,
) -> bool {
    let mut overall_time = 0u64;
    let mut overall_num = 0u32;
    for &reference in action_set {
        let classifier = population.get(reference);
        overall_time += classifier.tga * classifier.num as u64;
        overall_num += classifier.num;
    }

    if overall_num == 0 {
        return false;
    }

    time as f64 - overall_time as f64 / overall_num as f64 > theta_ga as f64
}

fn set_timestamps<const N: usize>(
    population: &mut Population<N>,
    action_set: &[ClassifierRef],
    time: u64,
) {
    for &reference in action_set {
        population.get_mut(reference).tga = time;
    }
}

fn weighted_choice(
    action_set: &[ClassifierRef],
    weights: &[f64],
    rng: &mut dyn RandomSource,
) -> ClassifierRef {
    let total: f64 = weights.iter().sum();
    let pick = rng.gen_unit() * total;
    let mut accumulated = 0.0;
    for (index, &weight) in weights.iter().enumerate() {
        accumulated += weight;
        if accumulated > pick {
            return action_set[index];
        }
    }
    *action_set.last().unwrap()
}

pub fn roulette_wheel_selection<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    rng: &mut dyn RandomSource,
) -> (ClassifierRef, ClassifierRef) {
    let weights: Vec<f64> = action_set
        .iter()
        .map(|&reference| {
            let classifier = population.get(reference);
            classifier.q.powi(3) * classifier.num as f64
        })
        .collect();

    let first = weighted_choice(action_set, &weights, rng);
    let second = weighted_choice(action_set, &weights, rng);
    (first, second)
}

pub fn generalizing_mutation<const N: usize>(
    classifier: &mut Classifier<N>,
    mu: f64,
    rng: &mut dyn RandomSource,
) {
    for index in 0..N {
        if !classifier.condition.get(index).is_wildcard() && rng.gen_bool(mu) {
            classifier.condition.generalize(index);
        }
    }
}

pub fn two_point_crossover<const N: usize>(
    first: &mut Classifier<N>,
    second: &mut Classifier<N>,
    rng: &mut dyn RandomSource,
) {
    let left_pick = rng.gen_range(N + 1);
    let mut right_pick = rng.gen_range(N);
    if right_pick >= left_pick {
        right_pick += 1;
    }
    let left = left_pick.min(right_pick);
    let right = left_pick.max(right_pick);

    for index in left..right {
        let from_first = first.condition.get(index);
        let from_second = second.condition.get(index);
        first.condition.set(index, from_second);
        second.condition.set(index, from_first);
    }
}

fn is_preferred_to_delete<const N: usize>(
    marked_for_deletion: &Classifier<N>,
    examined: &Classifier<N>,
) -> bool {
    if examined.q - marked_for_deletion.q < -0.1 {
        return true;
    }

    if (examined.q - marked_for_deletion.q).abs() <= 0.1 {
        if examined.is_marked() && !marked_for_deletion.is_marked() {
            return true;
        } else if examined.is_marked() || !marked_for_deletion.is_marked() {
            if examined.tav > marked_for_deletion.tav {
                return true;
            }
        }
    }

    false
}

fn action_set_numerosity<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
) -> u32 {
    action_set
        .iter()
        .map(|&reference| population.get(reference).num)
        .sum()
}

fn delete_classifiers<const N: usize>(
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    insize: usize,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    while insize as u32 + action_set_numerosity(population, action_set) > config.theta_as {
        let mut victim: Option<ClassifierRef> = None;
        while victim.is_none() {
            for &reference in action_set.iter() {
                for _ in 0..population.get(reference).num {
                    if rng.gen_bool(0.3) {
                        victim = Some(match victim {
                            None => reference,
                            Some(current) => {
                                if is_preferred_to_delete(population.get(current), population.get(reference)) {
                                    reference
                                } else {
                                    current
                                }
                            }
                        });
                    }
                }
            }
        }

        let chosen = victim.unwrap();
        if population.get(chosen).num > 1 {
            population.get_mut(chosen).num -= 1;
        } else {
            let removed = [chosen];
            remap_after_removal(match_set, &removed);
            remap_after_removal(action_set, &removed);
            population.remove_many(&removed);
        }
    }
}

fn find_old_classifier<const N: usize>(
    population: &Population<N>,
    action_set: &[ClassifierRef],
    child: &Classifier<N>,
    config: &Configuration,
) -> Option<ClassifierRef> {
    if config.do_subsumption {
        let mut subsumer: Option<ClassifierRef> = None;
        for &reference in action_set {
            if does_subsume(population.get(reference), child, config.theta_exp, config.theta_r) {
                let replaces = match subsumer {
                    None => true,
                    Some(current) => {
                        population.get(reference).is_more_general(population.get(current))
                    }
                };
                if replaces {
                    subsumer = Some(reference);
                }
            }
        }
        if subsumer.is_some() {
            return subsumer;
        }
    }

    action_set
        .iter()
        .copied()
        .find(|&reference| *population.get(reference) == *child)
}

fn add_classifier<const N: usize>(
    child: Classifier<N>,
    state: &Perception<N>,
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    config: &Configuration,
) {
    match find_old_classifier(population, action_set, &child, config) {
        None => {
            let matches_state = child.does_match(state);
            let reference = population.insert(child);
            action_set.push(reference);
            if matches_state {
                match_set.push(reference);
            }
        }
        Some(reference) => {
            if !population.get(reference).is_marked() {
                population.get_mut(reference).num += 1;
            }
        }
    }
}

pub fn apply_ga<const N: usize>(
    time: u64,
    population: &mut Population<N>,
    match_set: &mut Vec<ClassifierRef>,
    action_set: &mut Vec<ClassifierRef>,
    state: &Perception<N>,
    config: &Configuration,
    rng: &mut dyn RandomSource,
) {
    if !should_apply(population, action_set, time, config.theta_ga) {
        return;
    }

    set_timestamps(population, action_set, time);

    let (first_parent, second_parent) = roulette_wheel_selection(population, action_set, rng);
    let mut first_child = Classifier::copy_from(population.get(first_parent), time);
    let mut second_child = Classifier::copy_from(population.get(second_parent), time);

    generalizing_mutation(&mut first_child, config.mu, rng);
    generalizing_mutation(&mut second_child, config.mu, rng);

    if rng.gen_bool(config.chi) && first_child.effect == second_child.effect {
        two_point_crossover(&mut first_child, &mut second_child, rng);
        let mean_q = (first_child.q + second_child.q) / 2.0;
        let mean_r = (first_child.r + second_child.r) / 2.0;
        first_child.q = mean_q;
        second_child.q = mean_q;
        first_child.r = mean_r;
        second_child.r = mean_r;
    }

    first_child.q /= 2.0;
    second_child.q /= 2.0;

    let mut children: Vec<Classifier<N>> = Vec::new();
    if first_child.condition.specificity() > 0 {
        children.push(first_child);
    }
    if second_child.condition.specificity() > 0
        && !children.iter().any(|existing| *existing == second_child)
    {
        children.push(second_child);
    }

    delete_classifiers(population, match_set, action_set, children.len(), config, rng);

    for child in children {
        add_classifier(child, state, population, match_set, action_set, config);
    }
}
