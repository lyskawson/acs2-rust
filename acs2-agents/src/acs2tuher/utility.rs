use std::collections::BTreeMap;

use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::{ClassifierRef, Population};
use acs2_core::representation::symbol::Symbol;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{GoalEvaluator, StoredEpisode};

use crate::per_goal_reward::{bootstrap_with, GoalValue, PerGoalRewards};

use super::{Cost, Measure};

pub(super) fn sample_candidates<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    observed_goals: &[Goal<G>],
    count: usize,
    evaluator: &impl GoalEvaluator<S, G>,
    rng: &mut impl RandomSource,
) -> Vec<Goal<G>> {
    let mut candidates = observed_goals
        .iter()
        .copied()
        .filter(|goal| {
            *goal != episode.start().desired && !evaluator.is_reached(episode.achieved(0), goal)
        })
        .collect::<Vec<_>>();
    let count = count.min(candidates.len());
    for index in 0..count {
        let remaining = candidates.len() - index;
        if remaining > 1 {
            candidates.swap(index, index + rng.gen_range(remaining));
        }
    }
    candidates.truncate(count);
    candidates
}

pub(super) fn first_reach<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    goal: &Goal<G>,
    evaluator: &impl GoalEvaluator<S, G>,
) -> Option<usize> {
    (1..=episode.len()).find(|&state| evaluator.is_reached(episode.achieved(state), goal))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Score {
    pub utility: f64,
    pub post_update_level: Option<f64>,
}

pub(super) struct ObservationMatches {
    sets: Vec<Vec<ClassifierRef>>,
    order: Vec<usize>,
}

impl ObservationMatches {
    pub fn new<const S: usize, const G: usize, const M: usize>(
        episode: &StoredEpisode<S, G>,
        states: usize,
        population: &Population<M>,
        cost: &mut Cost,
    ) -> Self {
        let mut observations = Vec::new();
        let mut sets = Vec::new();
        let mut order = Vec::with_capacity(states);
        let any_goal = Goal::<G>::new([Symbol::Wildcard; G]);
        for state in 0..states {
            let observation = episode.observation(state);
            let index = observations.iter().position(|&seen| seen == observation);
            order.push(index.unwrap_or_else(|| {
                let mask = GoalLayout::<S, G, M>::join(observation, &any_goal);
                sets.push(
                    population
                        .classifiers()
                        .iter()
                        .enumerate()
                        .filter_map(|(reference, classifier)| {
                            classifier.does_match(&mask).then_some(reference)
                        })
                        .collect(),
                );
                cost.observation_match_sets += 1;
                observations.push(observation);
                observations.len() - 1
            }));
        }
        Self { sets, order }
    }

    pub fn for_goal<const S: usize, const G: usize, const M: usize>(
        &self,
        population: &Population<M>,
        goal: &Goal<G>,
        cost: &mut Cost,
    ) -> Vec<Vec<ClassifierRef>> {
        let mask = GoalLayout::<S, G, M>::join(&Perception::new([Symbol::Wildcard; S]), goal);
        cost.match_sets += self.sets.len() as u64;
        self.sets
            .iter()
            .map(|set| {
                set.iter()
                    .copied()
                    .filter(|&reference| population.get(reference).does_match(&mask))
                    .collect()
            })
            .collect()
    }
}

pub(super) struct Scoring<'a, const G: usize, const M: usize> {
    pub population: &'a Population<M>,
    pub rewards: &'a PerGoalRewards<G>,
    pub config: &'a Configuration,
}

impl<const G: usize, const M: usize> Scoring<'_, G, M> {
    pub fn score<const S: usize>(
        &self,
        measure: Measure,
        episode: &StoredEpisode<S, G>,
        (goal, prefix): (Goal<G>, usize),
        observations: &ObservationMatches,
        evaluator: &impl GoalEvaluator<S, G>,
        cost: &mut Cost,
    ) -> Score {
        let sets = observations.for_goal::<S, G, M>(self.population, &goal, cost);
        let matched = |state: usize| sets[observations.order[state]].as_slice();
        let index = self.rewards.goal_index(&goal);
        let live = |reference| self.rewards.value_at(reference, index);
        let utility = match measure {
            Measure::RandomGoal => unreachable!("the random arm does not score candidates"),
            Measure::MeanValue => {
                let total = (0..prefix)
                    .map(|step| {
                        let action = Some(episode.steps()[step].action);
                        let action_set = matched(step)
                            .iter()
                            .copied()
                            .filter(|&reference| self.population.get(reference).action == action);
                        bootstrap_with(self.population, action_set, |reference| live(reference).r)
                    })
                    .sum::<f64>();
                cost.value_reads += prefix as u64;
                Score {
                    utility: total / prefix as f64,
                    post_update_level: None,
                }
            }
            Measure::TrialUpdate => {
                let mut copied: BTreeMap<ClassifierRef, GoalValue> = BTreeMap::new();
                let mut value = |state, copied: &BTreeMap<_, _>| {
                    cost.value_reads += 1;
                    bootstrap_with(
                        self.population,
                        matched(state).iter().copied(),
                        |reference| {
                            copied
                                .get(&reference)
                                .copied()
                                .unwrap_or_else(|| live(reference))
                                .r
                        },
                    )
                };
                let before = (0..=episode.len())
                    .map(|state| value(state, &copied))
                    .collect::<Vec<_>>();
                for step in (0..prefix).rev() {
                    let stored = &episode.steps()[step];
                    let outcome = evaluator.outcome(&stored.step, &goal);
                    let bootstrap = if outcome.terminated {
                        0.0
                    } else {
                        value(step + 1, &copied)
                    };
                    let discounted = outcome.reward + self.config.gamma * bootstrap;
                    for reference in self
                        .population
                        .form_action_set(matched(step), stored.action)
                    {
                        copied
                            .entry(reference)
                            .or_insert_with(|| live(reference))
                            .update(self.config.beta, outcome.reward, discounted);
                        cost.trial_reward_entries += 1;
                    }
                }
                let mut delta = 0.0;
                let mut level = 0.0;
                for (state, previous) in before.into_iter().enumerate() {
                    let after = value(state, &copied);
                    delta += after - previous;
                    level += after;
                }
                cost.trial_value_updates += prefix as u64;
                let states = (episode.len() + 1) as f64;
                Score {
                    utility: delta / states,
                    post_update_level: Some(level / states),
                }
            }
        };
        cost.candidates_scored += 1;
        utility
    }
}
