use std::collections::BTreeMap;

use acs2_core::goal::Goal;
use acs2_core::rng::RandomSource;

use crate::relabel::{GoalEvaluator, ObjectiveCost};
use crate::store::StoredEpisode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GoalStrategy {
    Original,
    Final,
    Future,
    Episode,
    UniformReal,
}

impl GoalStrategy {
    pub fn index(self) -> usize {
        match self {
            Self::Original => 0,
            Self::Final => 1,
            Self::Future => 2,
            Self::Episode => 3,
            Self::UniformReal => 4,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Final => "final",
            Self::Future => "future",
            Self::Episode => "episode",
            Self::UniformReal => "uniform_real",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admissibility {
    EveryTransition,
    FromNonGoalState,
    CounterfactualEpisode,
}

impl Admissibility {
    pub fn admits(self, facts: GoalFacts) -> bool {
        match self {
            Self::EveryTransition => true,
            Self::FromNonGoalState => !facts.already_reached,
            Self::CounterfactualEpisode => !facts.after_counterfactual_end,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::EveryTransition => "every_transition",
            Self::FromNonGoalState => "from_non_goal_state",
            Self::CounterfactualEpisode => "counterfactual_episode",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub strategy: GoalStrategy,
    pub admissibility: Admissibility,
    pub candidate_filter: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GoalFacts {
    pub already_reached: bool,
    pub after_counterfactual_end: bool,
}

pub fn goal_facts<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    transition: usize,
    goal: &Goal<G>,
    evaluator: &impl GoalEvaluator<S, G>,
    cost: &mut ObjectiveCost,
) -> GoalFacts {
    assert!(transition < episode.len());
    cost.reach_evaluations += 1;
    let already_reached = evaluator.is_reached(episode.achieved(transition), goal);
    let after_counterfactual_end = already_reached
        || (0..transition).any(|state| {
            cost.reach_evaluations += 1;
            evaluator.is_reached(episode.achieved(state), goal)
        });
    GoalFacts {
        already_reached,
        after_counterfactual_end,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GoalProbability<const G: usize> {
    pub goal: Goal<G>,
    pub weight: usize,
    pub probability: f64,
    pub facts: GoalFacts,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GoalDistribution<const G: usize> {
    pub choices: Vec<GoalProbability<G>>,
    pub source_count: usize,
    pub admitted_count: usize,
    pub cost: ObjectiveCost,
}

impl<const G: usize> GoalDistribution<G> {
    pub fn draw(&self, rng: &mut dyn RandomSource) -> Option<GoalProbability<G>> {
        if self.admitted_count == 0 {
            return None;
        }
        let mut index = rng.gen_range(self.admitted_count);
        for choice in &self.choices {
            if index < choice.weight {
                return Some(*choice);
            }
            index -= choice.weight;
        }
        unreachable!("admissible goal weights")
    }
    pub fn admissible_share(&self) -> f64 {
        if self.source_count == 0 {
            0.0
        } else {
            self.admitted_count as f64 / self.source_count as f64
        }
    }
}

pub(crate) fn source_goals<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    transition: usize,
    strategy: GoalStrategy,
    candidates: &[Goal<G>],
) -> BTreeMap<Goal<G>, usize> {
    let goals: Vec<Goal<G>> = match strategy {
        GoalStrategy::Original => vec![episode.start().desired],
        GoalStrategy::Final => vec![*episode.achieved(episode.len())],
        GoalStrategy::Future => (transition + 1..=episode.len())
            .map(|state| *episode.achieved(state))
            .collect(),
        GoalStrategy::Episode => (1..=episode.len())
            .map(|state| *episode.achieved(state))
            .collect(),
        GoalStrategy::UniformReal => candidates.to_vec(),
    };
    let mut weights = BTreeMap::new();
    for goal in goals {
        *weights.entry(goal).or_insert(0) += 1;
    }
    weights
}

pub(crate) fn distribution_with<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    transition: usize,
    selection: Selection,
    candidates: &[Goal<G>],
    mut facts_for: impl FnMut(&Goal<G>, &mut ObjectiveCost) -> GoalFacts,
) -> GoalDistribution<G> {
    assert!(transition < episode.len());
    let source = source_goals(episode, transition, selection.strategy, candidates);
    let mut result = GoalDistribution {
        source_count: source.values().sum(),
        admitted_count: 0,
        choices: Vec::new(),
        cost: ObjectiveCost::default(),
    };
    for (goal, weight) in source {
        if selection.candidate_filter && !candidates.contains(&goal) {
            continue;
        }
        let facts = facts_for(&goal, &mut result.cost);
        if selection.admissibility.admits(facts) {
            result.admitted_count += weight;
            result.choices.push(GoalProbability {
                goal,
                weight,
                probability: 0.0,
                facts,
            });
        }
    }
    for choice in &mut result.choices {
        choice.probability = choice.weight as f64 / result.admitted_count as f64;
    }
    result
}

pub fn goal_distribution<const S: usize, const G: usize>(
    episode: &StoredEpisode<S, G>,
    transition: usize,
    selection: Selection,
    candidates: &[Goal<G>],
    evaluator: &impl GoalEvaluator<S, G>,
) -> GoalDistribution<G> {
    distribution_with(episode, transition, selection, candidates, |goal, cost| {
        goal_facts(episode, transition, goal, evaluator, cost)
    })
}
