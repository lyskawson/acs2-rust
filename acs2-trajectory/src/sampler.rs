use acs2_core::goal::Goal;
use acs2_core::rng::RandomSource;
use acs2_core::trial::TruncationMode;

use crate::cost::CostByPurpose;
use crate::relabel::{build_sample, scoring_cost, GoalEvaluator, SampleError, ScoredSample};
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
    pub cost: CostByPurpose,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteMeans {
    pub already_reached: f64,
    pub after_counterfactual_end: f64,
    pub done: f64,
    pub outside_candidates: f64,
    pub reward: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RouteCounters {
    pub draws: u64,
    pub already_reached: u64,
    pub after_counterfactual_end: u64,
    pub done: u64,
    pub outside_candidates: u64,
    pub reward: f64,
}

impl RouteCounters {
    fn record(&mut self, provenance: Provenance, reward: f64) {
        self.draws += 1;
        self.already_reached += u64::from(provenance.already_reached);
        self.after_counterfactual_end += u64::from(provenance.after_counterfactual_end);
        self.done += u64::from(provenance.done);
        self.outside_candidates += u64::from(provenance.outside_candidates);
        self.reward += reward;
    }
    pub fn add(&mut self, other: Self) {
        self.draws += other.draws;
        self.already_reached += other.already_reached;
        self.after_counterfactual_end += other.after_counterfactual_end;
        self.done += other.done;
        self.outside_candidates += other.outside_candidates;
        self.reward += other.reward;
    }
    pub fn given_route(&self) -> Option<RouteMeans> {
        let draws = self.draws as f64;
        (self.draws > 0).then(|| RouteMeans {
            already_reached: self.already_reached as f64 / draws,
            after_counterfactual_end: self.after_counterfactual_end as f64 / draws,
            done: self.done as f64 / draws,
            outside_candidates: self.outside_candidates as f64 / draws,
            reward: self.reward / draws,
        })
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ReplayCounters {
    pub failed_draws: u64,
    pub fallbacks: u64,
    pub original: RouteCounters,
    pub relabeled: RouteCounters,
    pub strategies: [u64; 5],
    pub cost: CostByPurpose,
}

impl ReplayCounters {
    pub fn pooled(&self) -> RouteCounters {
        let mut pooled = self.original;
        pooled.add(self.relabeled);
        pooled
    }
    fn record<const G: usize, const M: usize>(&mut self, drawn: &DrawnSample<G, M>) {
        let provenance = drawn.provenance;
        let route = match provenance.origin {
            SampleOrigin::Original => &mut self.original,
            SampleOrigin::Relabeled => &mut self.relabeled,
        };
        route.record(provenance, drawn.scored.sample.reward);
        self.fallbacks += u64::from(provenance.fallback);
        self.strategies[provenance.strategy.index()] += 1;
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
        let mut performed = CostByPurpose::default();
        for _ in 0..count {
            match self.draw_one(store, evaluator) {
                Ok(drawn) => {
                    performed.add(drawn.cost);
                    result.push(drawn);
                }
                Err((error, cost)) => {
                    performed.add(cost);
                    self.counters.cost.add(performed);
                    self.counters.failed_draws += 1;
                    return Err(error);
                }
            }
        }
        for drawn in &result {
            self.counters.record(drawn);
        }
        Ok(result)
    }

    fn draw_one<const S: usize, const G: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        evaluator: &impl GoalEvaluator<S, G>,
    ) -> Result<DrawnSample<G, M>, (SampleError, CostByPurpose)> {
        let (episode, transition) = store.transition(self.rng.gen_range(store.len()));
        let config = self.configuration;
        let attempt = config.selection.strategy != GoalStrategy::Original
            && (config.relabeled_proportion == 1.0
                || (config.relabeled_proportion > 0.0
                    && self.rng.gen_bool(config.relabeled_proportion)));
        let mut cost = CostByPurpose::default();
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
            let (facts, queries) = goal_facts(episode, transition, &goal, evaluator);
            cost.add(queries.as_provenance());
            (goal, facts, SampleOrigin::Original, GoalStrategy::Original)
        };
        let scored = match build_sample(episode, transition, &goal, evaluator, config.truncation) {
            Ok(scored) => scored,
            Err(error) => {
                cost.scoring
                    .add(scoring_cost(&episode.steps()[transition].step));
                return Err((error, cost));
            }
        };
        cost.scoring.add(scored.cost);
        Ok(DrawnSample {
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
        })
    }
}
