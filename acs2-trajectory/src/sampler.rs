use acs2_core::goal::Goal;
use acs2_core::rng::RandomSource;
use acs2_core::trial::TruncationMode;

use crate::relabel::{build_sample, GoalEvaluator, ObjectiveCost, SampleError, ScoredSample};
use crate::selection::{goal_distribution, goal_facts, GoalStrategy, Selection};
use crate::store::{EpisodeId, TrajectoryStore};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplerConfiguration {
    pub selection: Selection,
    pub relabeled_proportion: f64,
    pub truncation: TruncationMode,
}

impl SamplerConfiguration {
    pub fn validate(self) {
        assert!(
            self.relabeled_proportion.is_finite()
                && (0.0..=1.0).contains(&self.relabeled_proportion),
            "relabeled proportion must be in [0, 1]"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleOrigin {
    Original,
    Relabeled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Provenance {
    pub origin: SampleOrigin,
    pub strategy: GoalStrategy,
    pub requested_strategy: GoalStrategy,
    pub already_reached: bool,
    pub after_counterfactual_end: bool,
    pub done: bool,
    pub outside_candidates: bool,
    pub fallback: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrawnSample<const G: usize, const M: usize> {
    pub episode: EpisodeId,
    pub transition: usize,
    pub goal: Goal<G>,
    pub scored: ScoredSample<M>,
    pub provenance: Provenance,
    pub cost: ObjectiveCost,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayCounters {
    pub draws: u64,
    pub failed_draws: u64,
    pub original: u64,
    pub relabeled: u64,
    pub fallbacks: u64,
    pub already_reached: u64,
    pub after_counterfactual_end: u64,
    pub done: u64,
    pub outside_candidates: u64,
    pub strategies: [u64; 5],
    pub cost: ObjectiveCost,
}

impl ReplayCounters {
    fn record<const G: usize, const M: usize>(&mut self, drawn: &DrawnSample<G, M>) {
        let p = drawn.provenance;
        self.draws += 1;
        self.original += u64::from(p.origin == SampleOrigin::Original);
        self.relabeled += u64::from(p.origin == SampleOrigin::Relabeled);
        self.fallbacks += u64::from(p.fallback);
        self.already_reached += u64::from(p.already_reached);
        self.after_counterfactual_end += u64::from(p.after_counterfactual_end);
        self.done += u64::from(p.done);
        self.outside_candidates += u64::from(p.outside_candidates);
        self.strategies[p.strategy.index()] += 1;
        self.cost.add(drawn.cost);
    }
}

pub struct Sampler<R> {
    rng: R,
    configuration: SamplerConfiguration,
    counters: ReplayCounters,
}

impl<R: RandomSource> Sampler<R> {
    pub fn new(configuration: SamplerConfiguration, rng: R) -> Self {
        configuration.validate();
        Self {
            rng,
            configuration,
            counters: ReplayCounters::default(),
        }
    }
    pub fn configuration(&self) -> SamplerConfiguration {
        self.configuration
    }
    pub fn counters(&self) -> &ReplayCounters {
        &self.counters
    }
    pub fn random_source(&self) -> &R {
        &self.rng
    }

    pub fn draw<const S: usize, const G: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        count: usize,
        evaluator: &impl GoalEvaluator<S, G>,
    ) -> Result<Vec<DrawnSample<G, M>>, SampleError> {
        if count > 0 && store.is_empty() {
            return Err(SampleError::EmptyStore);
        }
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            let (episode, transition) = store.transition(self.rng.gen_range(store.len()));
            let config = self.configuration;
            let attempt = config.selection.strategy != GoalStrategy::Original
                && (config.relabeled_proportion == 1.0
                    || (config.relabeled_proportion > 0.0
                        && self.rng.gen_bool(config.relabeled_proportion)));
            let mut cost = ObjectiveCost::default();
            let selected = if attempt {
                let distribution = goal_distribution(
                    episode,
                    transition,
                    config.selection,
                    store.candidates(),
                    evaluator,
                );
                cost.add(distribution.cost);
                distribution.draw(&mut self.rng)
            } else {
                None
            };
            let (goal, facts, origin, strategy) = if let Some(choice) = selected {
                (
                    choice.goal,
                    choice.facts,
                    SampleOrigin::Relabeled,
                    config.selection.strategy,
                )
            } else {
                let goal = episode.start().desired;
                let facts = goal_facts(episode, transition, &goal, evaluator, &mut cost);
                (goal, facts, SampleOrigin::Original, GoalStrategy::Original)
            };
            let scored =
                match build_sample(episode, transition, &goal, evaluator, config.truncation) {
                    Ok(scored) => scored,
                    Err(error) => {
                        cost.add(ObjectiveCost {
                            reward_evaluations: 1,
                            reach_evaluations: u64::from(
                                !episode.steps()[transition].step.terminal_state,
                            ),
                        });
                        self.counters.cost.add(cost);
                        self.counters.failed_draws += 1;
                        return Err(error);
                    }
                };
            cost.add(scored.cost);
            let drawn = DrawnSample {
                episode: episode.id(),
                transition,
                goal,
                provenance: Provenance {
                    origin,
                    strategy,
                    requested_strategy: config.selection.strategy,
                    already_reached: facts.already_reached,
                    after_counterfactual_end: facts.after_counterfactual_end,
                    done: scored.sample.done,
                    outside_candidates: !store.candidates().contains(&goal),
                    fallback: attempt && selected.is_none(),
                },
                scored,
                cost,
            };
            self.counters.record(&drawn);
            result.push(drawn);
        }
        Ok(result)
    }
}
