use std::collections::{BTreeMap, BTreeSet};

use acs2_core::goal::Goal;

use crate::cost::{FactQueries, ObjectiveCost};
use crate::relabel::{build_sample, GoalEvaluator, SampleError};
use crate::sampler::{RouteMeans, SampleOrigin, SamplerConfiguration};
use crate::selection::{distribution_from_source, source_goals, GoalFacts, GoalStrategy};
use crate::store::StoredEpisode;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteComposition {
    pub share: f64,
    pub already_reached: f64,
    pub after_counterfactual_end: f64,
    pub done: f64,
    pub outside_candidates: f64,
    pub reward: f64,
}

impl RouteComposition {
    fn add_draw(
        &mut self,
        probability: f64,
        facts: GoalFacts,
        done: bool,
        outside_candidates: bool,
        reward: f64,
    ) {
        self.already_reached += probability * f64::from(facts.already_reached);
        self.after_counterfactual_end += probability * f64::from(facts.after_counterfactual_end);
        self.done += probability * f64::from(done);
        self.outside_candidates += probability * f64::from(outside_candidates);
        self.reward += probability * reward;
    }
    pub fn add_weighted(&mut self, other: Self, weight: f64) {
        self.share += other.share * weight;
        self.already_reached += other.already_reached * weight;
        self.after_counterfactual_end += other.after_counterfactual_end * weight;
        self.done += other.done * weight;
        self.outside_candidates += other.outside_candidates * weight;
        self.reward += other.reward * weight;
    }
    pub fn given_route(self) -> Option<RouteMeans> {
        (self.share > 0.0).then(|| RouteMeans {
            already_reached: self.already_reached / self.share,
            after_counterfactual_end: self.after_counterfactual_end / self.share,
            done: self.done / self.share,
            outside_candidates: self.outside_candidates / self.share,
            reward: self.reward / self.share,
        })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ExpectedComposition {
    pub admissible_share: f64,
    pub no_admissible_goal: f64,
    pub original: RouteComposition,
    pub relabeled: RouteComposition,
    pub fallback: f64,
    pub already_reached: f64,
    pub after_counterfactual_end: f64,
    pub done: f64,
    pub outside_candidates: f64,
    pub mean_reward: f64,
    pub mean_reward_evaluations: f64,
    pub mean_reach_evaluations: f64,
    pub mean_scoring_evaluations: f64,
    pub mean_selection_evaluations: f64,
    pub mean_provenance_evaluations: f64,
}

impl ExpectedComposition {
    pub fn add_weighted(&mut self, other: Self, weight: f64) {
        self.admissible_share += other.admissible_share * weight;
        self.no_admissible_goal += other.no_admissible_goal * weight;
        self.original.add_weighted(other.original, weight);
        self.relabeled.add_weighted(other.relabeled, weight);
        self.fallback += other.fallback * weight;
        self.already_reached += other.already_reached * weight;
        self.after_counterfactual_end += other.after_counterfactual_end * weight;
        self.done += other.done * weight;
        self.outside_candidates += other.outside_candidates * weight;
        self.mean_reward += other.mean_reward * weight;
        self.mean_reward_evaluations += other.mean_reward_evaluations * weight;
        self.mean_reach_evaluations += other.mean_reach_evaluations * weight;
        self.mean_scoring_evaluations += other.mean_scoring_evaluations * weight;
        self.mean_selection_evaluations += other.mean_selection_evaluations * weight;
        self.mean_provenance_evaluations += other.mean_provenance_evaluations * weight;
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
    fn queries(&self, transition: usize) -> FactQueries {
        FactQueries {
            current_state: 1,
            earlier_states: if self.reached[transition] {
                0
            } else {
                self.first
                    .filter(|&state| state < transition)
                    .map_or(transition, |state| state + 1) as u64
            },
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
    let mut source = source_goals(episode, 0, configuration.selection.strategy, candidates);
    let candidates: BTreeSet<_> = candidates.iter().copied().collect();
    let mut goals = source.clone();
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
        let distribution =
            distribution_from_source(&source, configuration.selection, &candidates, |goal| {
                let history = &histories[goal];
                (history.facts(transition), history.queries(transition))
            });
        let empty = distribution.admitted_count == 0;
        let fallback = if empty { attempt } else { 0.0 };
        let original_probability = 1.0 - attempt + fallback;
        let mut expected = ExpectedComposition {
            admissible_share: distribution.admissible_share(),
            no_admissible_goal: f64::from(empty),
            original: RouteComposition {
                share: original_probability,
                ..RouteComposition::default()
            },
            relabeled: RouteComposition {
                share: attempt - fallback,
                ..RouteComposition::default()
            },
            fallback,
            mean_reach_evaluations: attempt * distribution.cost.total().reach_evaluations as f64,
            mean_selection_evaluations: attempt * distribution.cost.selection.total() as f64,
            mean_provenance_evaluations: attempt * distribution.cost.provenance.total() as f64,
            ..ExpectedComposition::default()
        };
        let mut accumulate = |goal: &Goal<G>,
                              facts: GoalFacts,
                              probability: f64,
                              origin: SampleOrigin|
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
            let outside_candidates = !candidates.contains(goal);
            expected.already_reached += probability * f64::from(facts.already_reached);
            expected.after_counterfactual_end +=
                probability * f64::from(facts.after_counterfactual_end);
            expected.done += probability * f64::from(scored.sample.done);
            expected.outside_candidates += probability * f64::from(outside_candidates);
            expected.mean_reward += probability * scored.sample.reward;
            expected.mean_reward_evaluations += probability * scored.cost.reward_evaluations as f64;
            expected.mean_reach_evaluations += probability * scored.cost.reach_evaluations as f64;
            expected.mean_scoring_evaluations += probability * scored.cost.total() as f64;
            let route = match origin {
                SampleOrigin::Original => &mut expected.original,
                SampleOrigin::Relabeled => &mut expected.relabeled,
            };
            route.add_draw(
                probability,
                facts,
                scored.sample.done,
                outside_candidates,
                scored.sample.reward,
            );
            Ok(())
        };
        for choice in &distribution.choices {
            accumulate(
                &choice.goal,
                choice.facts,
                attempt * choice.probability,
                SampleOrigin::Relabeled,
            )?;
        }
        let original = episode.start().desired;
        let history = &histories[&original];
        accumulate(
            &original,
            history.facts(transition),
            original_probability,
            SampleOrigin::Original,
        )?;
        let original_queries = history.queries(transition).total() as f64;
        expected.mean_reach_evaluations += original_probability * original_queries;
        expected.mean_provenance_evaluations += original_probability * original_queries;
        result
            .expected
            .add_weighted(expected, 1.0 / episode.len() as f64);
        if configuration.selection.strategy == GoalStrategy::Future {
            let visited = episode.achieved(transition + 1);
            let weight = source.get_mut(visited).expect("future state weight");
            *weight -= 1;
            if *weight == 0 {
                source.remove(visited);
            }
        }
    }
    Ok(result)
}
