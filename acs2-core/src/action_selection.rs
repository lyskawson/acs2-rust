use crate::population::{ClassifierRef, Population};
use crate::rng::{shuffle, RandomSource};

pub trait ActionSelector<const N: usize> {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize;
}

pub struct EpsilonGreedy {
    pub number_of_possible_actions: usize,
    pub epsilon: f64,
}

pub struct BestAction {
    pub number_of_possible_actions: usize,
}

pub struct RandomAction {
    pub number_of_possible_actions: usize,
}

fn best_change_anticipating_action<const N: usize>(
    population: &Population<N>,
    match_set: &[ClassifierRef],
    number_of_possible_actions: usize,
    rng: &mut dyn RandomSource,
) -> usize {
    let mut anticipating: Vec<ClassifierRef> = match_set
        .iter()
        .copied()
        .filter(|&reference| population.get(reference).does_anticipate_change())
        .collect();

    if anticipating.is_empty() {
        return rng.gen_range(number_of_possible_actions);
    }

    shuffle(&mut anticipating, rng);

    let best = anticipating
        .iter()
        .copied()
        .max_by(|&left, &right| {
            let left_score = population.get(left).fitness() * population.get(left).num as f64;
            let right_score = population.get(right).fitness() * population.get(right).num as f64;
            left_score.partial_cmp(&right_score).unwrap()
        })
        .unwrap();

    population.get(best).action.unwrap()
}

impl<const N: usize> ActionSelector<N> for EpsilonGreedy {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        if rng.gen_bool(self.epsilon) {
            rng.gen_range(self.number_of_possible_actions)
        } else {
            best_change_anticipating_action(
                population,
                match_set,
                self.number_of_possible_actions,
                rng,
            )
        }
    }
}

impl<const N: usize> ActionSelector<N> for BestAction {
    fn select(
        &self,
        population: &Population<N>,
        match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        best_change_anticipating_action(
            population,
            match_set,
            self.number_of_possible_actions,
            rng,
        )
    }
}

impl<const N: usize> ActionSelector<N> for RandomAction {
    fn select(
        &self,
        _population: &Population<N>,
        _match_set: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        rng.gen_range(self.number_of_possible_actions)
    }
}
