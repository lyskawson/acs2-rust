use std::mem::size_of;

use acs2_core::ports::goal::Goal;
use acs2_core::representation::population::{ClassifierRef, Population};
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::{shuffle, RandomSource};
use acs2_trajectory::SampleOrigin;

mod learning;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalValue {
    pub r: f64,
    pub ir: f64,
    pub updated: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RewardUpdates {
    pub original: u64,
    pub relabeled: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PerGoalRewards<const G: usize> {
    goals: Vec<Goal<G>>,
    entries: Vec<Vec<GoalValue>>,
    updates: RewardUpdates,
}

impl<const G: usize> PerGoalRewards<G> {
    pub fn new(goals: &[Goal<G>], config: &Configuration, general_classifiers: usize) -> Self {
        assert!(!config.do_ga, "per-goal reward agents require GA disabled");
        let mut ordered = goals.to_vec();
        ordered.sort_unstable();
        ordered.dedup();
        assert!(
            !ordered.is_empty(),
            "per-goal rewards require a finite goal set"
        );
        assert_eq!(
            ordered.len(),
            goals.len(),
            "per-goal reward goals must be unique"
        );
        let mut result = Self {
            goals: ordered,
            entries: Vec::new(),
            updates: RewardUpdates::default(),
        };
        result.initialize_general(general_classifiers, config);
        result
    }

    pub fn goals(&self) -> &[Goal<G>] {
        &self.goals
    }

    pub fn entry(&self, classifier: ClassifierRef, goal: &Goal<G>) -> &GoalValue {
        &self.entries[classifier][self.goal_index(goal)]
    }

    pub fn entry_mut(&mut self, classifier: ClassifierRef, goal: &Goal<G>) -> &mut GoalValue {
        let index = self.goal_index(goal);
        &mut self.entries[classifier][index]
    }

    pub fn existing_pairs(&self) -> usize {
        self.entries.len() * self.goals.len()
    }

    pub fn updated_entries(&self) -> usize {
        self.entries
            .iter()
            .flatten()
            .filter(|entry| entry.updated)
            .count()
    }

    pub fn updates(&self) -> RewardUpdates {
        self.updates
    }

    pub fn logical_bytes(&self) -> usize {
        self.goals.len() * size_of::<Goal<G>>()
            + self.entries.len() * size_of::<Vec<GoalValue>>()
            + self.existing_pairs() * size_of::<GoalValue>()
    }

    pub(crate) fn initialize_general(&mut self, classifiers: usize, config: &Configuration) {
        assert_eq!(self.updates, RewardUpdates::default());
        self.entries = vec![self.start_values(config.initial_r, config.initial_ir); classifiers];
    }

    fn start_values(&self, r: f64, ir: f64) -> Vec<GoalValue> {
        vec![
            GoalValue {
                r,
                ir,
                updated: false
            };
            self.goals.len()
        ]
    }

    fn goal_index(&self, goal: &Goal<G>) -> usize {
        self.goals
            .binary_search(goal)
            .expect("unregistered per-goal reward goal")
    }

    pub fn bootstrap<const M: usize>(
        &self,
        population: &Population<M>,
        matched: &[ClassifierRef],
        goal: &Goal<G>,
    ) -> f64 {
        let index = self.goal_index(goal);
        matched
            .iter()
            .copied()
            .filter(|&reference| population.get(reference).does_anticipate_change())
            .map(|reference| population.get(reference).q * self.entries[reference][index].r)
            .fold(0.0, f64::max)
    }

    pub fn best_action<const M: usize>(
        &self,
        population: &Population<M>,
        matched: &[ClassifierRef],
        goal: &Goal<G>,
        actions: usize,
        rng: &mut dyn RandomSource,
    ) -> usize {
        let index = self.goal_index(goal);
        let mut anticipating: Vec<_> = matched
            .iter()
            .copied()
            .filter(|&reference| population.get(reference).does_anticipate_change())
            .collect();
        if anticipating.is_empty() {
            return rng.gen_range(actions);
        }
        shuffle(&mut anticipating, rng);
        let score = |reference| {
            let classifier = population.get(reference);
            classifier.q * self.entries[reference][index].r * f64::from(classifier.num)
        };
        let best = anticipating
            .into_iter()
            .max_by(|&left, &right| score(left).partial_cmp(&score(right)).unwrap())
            .unwrap();
        population.get(best).action.unwrap()
    }

    pub(crate) fn learn<const S: usize, const M: usize>(
        &mut self,
        population: &mut Population<M>,
        config: &Configuration,
        drawn: &acs2_trajectory::DrawnSample<G, M>,
        time: u64,
        rng: &mut dyn RandomSource,
    ) {
        assert!(!config.do_ga, "per-goal reward agents require GA disabled");
        assert_eq!(self.entries.len(), population.len());
        let sample = &drawn.scored.sample;
        assert_eq!(
            acs2_core::ports::goal::GoalLayout::<S, G, M>::split(&sample.state).1,
            drawn.goal
        );
        assert_eq!(
            acs2_core::ports::goal::GoalLayout::<S, G, M>::split(&sample.next_state).1,
            drawn.goal
        );
        let goal = self.goal_index(&drawn.goal);
        let matched = population.form_match_set(&sample.state);
        let mut sets = learning::AlpSets {
            action_set: population.form_action_set(&matched, sample.action),
            next_matched: population.form_match_set(&sample.next_state),
        };
        learning::apply_alp(self, population, &mut sets, sample, time, config, rng);
        let next_matched = population.form_match_set(&sample.next_state);
        let bootstrap = if sample.done {
            0.0
        } else {
            self.bootstrap(population, &next_matched, &drawn.goal)
        };
        let discounted_reward = sample.reward + config.gamma * bootstrap;
        for reference in sets.action_set {
            let entry = &mut self.entries[reference][goal];
            entry.r += config.beta * (discounted_reward - entry.r);
            entry.ir += config.beta * (sample.reward - entry.ir);
            entry.updated = true;
            match drawn.provenance.origin {
                SampleOrigin::Original => self.updates.original += 1,
                SampleOrigin::Relabeled => self.updates.relabeled += 1,
            }
        }
    }
}
