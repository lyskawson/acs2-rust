use acs2_core::acs2er::ReplaySample;
use acs2_core::goal::{Goal, GoalLayout, GoalObjective, GoalOutcome, GoalStep};
use acs2_core::trial::TruncationMode;

use crate::cost::{CostByPurpose, ObjectiveCost};
use crate::selection::{goal_facts, Admissibility};
use crate::store::{EpisodeId, StoredEpisode};

pub trait GoalEvaluator<const S: usize, const G: usize> {
    fn outcome(&self, step: &GoalStep<S, G>, desired: &Goal<G>) -> GoalOutcome;
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool;
}

pub struct ObjectiveEvaluator<'a, O: ?Sized>(pub &'a O);

impl<O: GoalObjective<G> + ?Sized, const S: usize, const G: usize> GoalEvaluator<S, G>
    for ObjectiveEvaluator<'_, O>
{
    fn outcome(&self, step: &GoalStep<S, G>, desired: &Goal<G>) -> GoalOutcome {
        step.outcome(self.0, desired)
    }
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool {
        self.0.is_reached(achieved, desired)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleError {
    EmptyStore,
    NegativeReward,
    NonFiniteReward,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredSample<const M: usize> {
    pub sample: ReplaySample<M>,
    pub outcome: GoalOutcome,
    pub cost: ObjectiveCost,
}

pub(crate) fn scoring_cost<const S: usize, const G: usize>(step: &GoalStep<S, G>) -> ObjectiveCost {
    ObjectiveCost {
        reward_evaluations: 1,
        reach_evaluations: u64::from(!step.terminal_state),
    }
}

pub fn build_sample<const S: usize, const G: usize, const M: usize>(
    episode: &StoredEpisode<S, G>,
    transition: usize,
    goal: &Goal<G>,
    evaluator: &impl GoalEvaluator<S, G>,
    truncation: TruncationMode,
) -> Result<ScoredSample<M>, SampleError> {
    let raw = &episode.steps()[transition];
    let outcome = evaluator.outcome(&raw.step, goal);
    if !outcome.reward.is_finite() {
        return Err(SampleError::NonFiniteReward);
    }
    if outcome.reward < 0.0 {
        return Err(SampleError::NegativeReward);
    }
    Ok(ScoredSample {
        sample: ReplaySample {
            state: GoalLayout::<S, G, M>::join(episode.observation(transition), goal),
            action: raw.action,
            reward: outcome.reward,
            next_state: GoalLayout::<S, G, M>::join(&raw.step.observation, goal),
            done: truncation.is_terminal(outcome.terminated, outcome.truncated),
        },
        outcome,
        cost: scoring_cost(&raw.step),
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EpisodeEnd {
    Terminated,
    Truncated,
    Cut,
    AlreadyReachedAtStart,
    NoAdmissibleTransition,
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexedSample<const M: usize> {
    pub transition: usize,
    pub scored: ScoredSample<M>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RelabeledEpisode<const M: usize> {
    pub id: EpisodeId,
    pub samples: Vec<IndexedSample<M>>,
    pub end: EpisodeEnd,
    pub cost: CostByPurpose,
}

pub fn relabel_episode<const S: usize, const G: usize, const M: usize>(
    episode: &StoredEpisode<S, G>,
    goal: &Goal<G>,
    rule: Admissibility,
    evaluator: &impl GoalEvaluator<S, G>,
    truncation: TruncationMode,
) -> Result<RelabeledEpisode<M>, SampleError> {
    let mut result = RelabeledEpisode {
        id: episode.id(),
        samples: Vec::new(),
        end: EpisodeEnd::NoAdmissibleTransition,
        cost: CostByPurpose::default(),
    };
    for transition in 0..episode.len() {
        let (facts, queries) = goal_facts(episode, transition, goal, evaluator);
        result.cost.add(rule.attribute(queries));
        if !rule.admits(facts) {
            if transition == 0 && rule == Admissibility::CounterfactualEpisode {
                result.end = EpisodeEnd::AlreadyReachedAtStart;
            }
            continue;
        }
        let scored = build_sample(episode, transition, goal, evaluator, truncation)?;
        result.cost.scoring.add(scored.cost);
        result.end = if scored.outcome.terminated {
            EpisodeEnd::Terminated
        } else if scored.outcome.truncated {
            EpisodeEnd::Truncated
        } else {
            EpisodeEnd::Cut
        };
        let terminated = scored.outcome.terminated;
        result.samples.push(IndexedSample { transition, scored });
        if rule == Admissibility::CounterfactualEpisode && terminated {
            break;
        }
    }
    Ok(result)
}
