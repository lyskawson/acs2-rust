use std::collections::BTreeMap;

use acs2_core::goal::Goal;

use crate::relabel::{build_sample, GoalEvaluator, ObjectiveCost, SampleError};
use crate::sampler::SamplerConfiguration;
use crate::selection::{distribution_with, source_goals, GoalFacts, GoalStrategy};
use crate::store::StoredEpisode;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpectedComposition {
    pub admissible_share: f64,
    pub no_admissible_goal: f64,
    pub original: f64,
    pub relabeled: f64,
    pub fallback: f64,
    pub already_reached: f64,
    pub after_counterfactual_end: f64,
    pub done: f64,
    pub outside_candidates: f64,
    pub mean_reward: f64,
    pub mean_reward_evaluations: f64,
    pub mean_reach_evaluations: f64,
}

impl ExpectedComposition {
    pub fn add_weighted(&mut self, other: Self, weight: f64) {
        self.admissible_share += other.admissible_share * weight;
        self.no_admissible_goal += other.no_admissible_goal * weight;
        self.original += other.original * weight;
        self.relabeled += other.relabeled * weight;
        self.fallback += other.fallback * weight;
        self.already_reached += other.already_reached * weight;
        self.after_counterfactual_end += other.after_counterfactual_end * weight;
        self.done += other.done * weight;
        self.outside_candidates += other.outside_candidates * weight;
        self.mean_reward += other.mean_reward * weight;
        self.mean_reward_evaluations += other.mean_reward_evaluations * weight;
        self.mean_reach_evaluations += other.mean_reach_evaluations * weight;
    }
    pub fn mean_objective_evaluations(self) -> f64 {
        self.mean_reward_evaluations + self.mean_reach_evaluations
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EpisodeComposition {
    pub transitions: usize,
    pub expected: ExpectedComposition,
    pub analysis_cost: ObjectiveCost,
}

struct GoalHistory {
    reached: Vec<bool>,
    first: Option<usize>,
}

impl GoalHistory {
    fn facts(&self, transition: usize) -> GoalFacts {
        GoalFacts {
            already_reached: self.reached[transition],
            after_counterfactual_end: self.first.is_some_and(|state| state <= transition),
        }
    }
    fn draw_queries(&self, transition: usize) -> u64 {
        if self.reached[transition] {
            1
        } else {
            1 + self
                .first
                .filter(|&state| state < transition)
                .map_or(transition, |state| state + 1) as u64
        }
    }
}

pub fn episode_composition<const S: usize, const G: usize, const M: usize>(
    episode: &StoredEpisode<S, G>,
    candidates: &[Goal<G>],
    configuration: SamplerConfiguration,
    evaluator: &impl GoalEvaluator<S, G>,
) -> Result<EpisodeComposition, SampleError> {
    configuration.validate();
    let mut histories = BTreeMap::new();
    let mut goals = source_goals(episode, 0, configuration.selection.strategy, candidates);
    if configuration.selection.candidate_filter {
        goals.retain(|goal, _| candidates.contains(goal));
    }
    goals.insert(episode.start().desired, 1);
    for goal in goals.into_keys() {
        histories.entry(goal).or_insert_with(|| {
            let reached: Vec<bool> = (0..episode.len())
                .map(|state| evaluator.is_reached(episode.achieved(state), &goal))
                .collect();
            let first = reached.iter().position(|&value| value);
            GoalHistory { reached, first }
        });
    }
    let mut result = EpisodeComposition {
        transitions: episode.len(),
        expected: ExpectedComposition::default(),
        analysis_cost: ObjectiveCost {
            reward_evaluations: 0,
            reach_evaluations: (histories.len() * episode.len()) as u64,
        },
    };
    let attempt = if configuration.selection.strategy == GoalStrategy::Original {
        0.0
    } else {
        configuration.relabeled_proportion
    };
    for transition in 0..episode.len() {
        let distribution = distribution_with(
            episode,
            transition,
            configuration.selection,
            candidates,
            |goal, cost| {
                let history = &histories[goal];
                cost.reach_evaluations += history.draw_queries(transition);
                history.facts(transition)
            },
        );
        let empty = distribution.admitted_count == 0;
        let fallback = if empty { attempt } else { 0.0 };
        let mut expected = ExpectedComposition {
            admissible_share: distribution.admissible_share(),
            no_admissible_goal: f64::from(empty),
            original: 1.0 - attempt + fallback,
            relabeled: attempt - fallback,
            fallback,
            mean_reach_evaluations: attempt * distribution.cost.reach_evaluations as f64,
            ..ExpectedComposition::default()
        };
        let mut accumulate = |goal: &Goal<G>,
                              facts: GoalFacts,
                              probability: f64|
         -> Result<(), SampleError> {
            if probability == 0.0 {
                return Ok(());
            }
            let scored = build_sample::<S, G, M>(
                episode,
                transition,
                goal,
                evaluator,
                configuration.truncation,
            )?;
            result.analysis_cost.add(scored.cost);
            expected.already_reached += probability * f64::from(facts.already_reached);
            expected.after_counterfactual_end +=
                probability * f64::from(facts.after_counterfactual_end);
            expected.done += probability * f64::from(scored.sample.done);
            expected.outside_candidates += probability * f64::from(!candidates.contains(goal));
            expected.mean_reward += probability * scored.sample.reward;
            expected.mean_reward_evaluations += probability * scored.cost.reward_evaluations as f64;
            expected.mean_reach_evaluations += probability * scored.cost.reach_evaluations as f64;
            Ok(())
        };
        for choice in &distribution.choices {
            accumulate(&choice.goal, choice.facts, attempt * choice.probability)?;
        }
        let original = episode.start().desired;
        let history = &histories[&original];
        let original_probability = 1.0 - attempt + fallback;
        accumulate(&original, history.facts(transition), original_probability)?;
        expected.mean_reach_evaluations +=
            original_probability * history.draw_queries(transition) as f64;
        result
            .expected
            .add_weighted(expected, 1.0 / episode.len() as f64);
    }
    Ok(result)
}
