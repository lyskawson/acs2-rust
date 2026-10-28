use acs2_core::learning::action_selection::{ActionSelector, BestAction, EpsilonGreedy};
use acs2_core::learning::replay::{replay_learning_step, ReplaySample};
use acs2_core::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::ports::goal::GoalLayout;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_trajectory::{
    DrawnSample, GoalEvaluator, GoalStrategy, ReplayCounters, SampleOrigin, Sampler,
    SamplerConfiguration, Selection, TrajectoryStore,
};

use crate::acs2her::HindsightConfiguration;
use crate::episodic_replay::{
    values_learning_step, CopyUpdate, EpisodicConfiguration, ReplayEnvironment, UpdateCounts,
};
use priority::{acceptance, action_value, population_variance, PriorityCounts, MAX_TEMPERATURE};

pub mod priority;

#[cfg(test)]
mod pyalcs_differential;
#[cfg(test)]
mod tests;

pub const EXTRA_HEAD_STREAM: u64 = 8;
pub const ACCEPTANCE_STREAM: u64 = 12;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    PrioritizedTwo { temperature: u32 },
    PrioritizedFour { temperature: u32 },
    UniformTwo,
    UniformFour,
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Self::PrioritizedTwo { .. } => "acs2vcp_k2",
            Self::PrioritizedFour { .. } => "acs2vcp_k4",
            Self::UniformTwo => "acs2vcp_k2_uniform",
            Self::UniformFour => "acs2vcp_k4_uniform",
        }
    }

    pub fn parameters(self) -> ConsistencyConfiguration {
        let (heads, temperature) = match self {
            Self::PrioritizedTwo { temperature } => (2, temperature),
            Self::PrioritizedFour { temperature } => (4, temperature),
            Self::UniformTwo => (2, 0),
            Self::UniformFour => (4, 0),
        };
        if matches!(
            self,
            Self::PrioritizedTwo { .. } | Self::PrioritizedFour { .. }
        ) {
            assert!((1..=MAX_TEMPERATURE).contains(&temperature));
        }
        ConsistencyConfiguration {
            heads,
            temperature,
            hindsight: crate::acs2her::Preset::Acs2Her { proportion: 2.0 / 3.0 }.configuration(),
            reward_on_reach: 1000.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConsistencyConfiguration {
    pub heads: usize,
    pub temperature: u32,
    pub hindsight: HindsightConfiguration,
    pub reward_on_reach: f64,
}

impl ConsistencyConfiguration {
    pub fn variance_maximum(self) -> f64 {
        self.reward_on_reach * self.reward_on_reach / 4.0
    }
}

pub struct Head<const M: usize, R> {
    population: Population<M>,
    rng: R,
    counts: UpdateCounts,
    episodes: u64,
}

impl<const M: usize, R: RandomSource> Head<M, R> {
    pub fn population(&self) -> &Population<M> {
        &self.population
    }
    pub fn random_source(&self) -> &R {
        &self.rng
    }
    pub fn counts(&self) -> UpdateCounts {
        self.counts
    }
    pub fn episodes(&self) -> u64 {
        self.episodes
    }
}

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P, A> {
    config: Configuration,
    replay: EpisodicConfiguration,
    parameters: ConsistencyConfiguration,
    heads: Vec<Head<M, R>>,
    store: TrajectoryStore<S, G>,
    sampler: Sampler<P>,
    acceptance_rng: A,
    accepted: ReplayCounters,
    priority: PriorityCounts,
    episodes: u64,
}

impl<const S: usize, const G: usize, const M: usize>
    Agent<S, G, M, ChaChaRandomSource, ChaChaRandomSource, ChaChaRandomSource>
{
    pub fn seeded(
        config: Configuration,
        replay: EpisodicConfiguration,
        parameters: ConsistencyConfiguration,
        seed: u64,
        agent_stream: u64,
    ) -> Self {
        let sources = (0..parameters.heads)
            .map(|head| {
                ChaChaRandomSource::from_seed_and_stream(
                    seed,
                    if head == 0 {
                        agent_stream
                    } else {
                        EXTRA_HEAD_STREAM + head as u64 - 1
                    },
                )
            })
            .collect();
        Self::new(
            config,
            replay,
            parameters,
            sources,
            ChaChaRandomSource::from_seed_and_stream(seed, acs2_trajectory::SAMPLER_STREAM),
            ChaChaRandomSource::from_seed_and_stream(seed, ACCEPTANCE_STREAM),
        )
    }
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
    > Agent<S, G, M, R, P, A>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        parameters: ConsistencyConfiguration,
        sources: Vec<R>,
        sampler_rng: P,
        acceptance_rng: A,
    ) -> Self {
        assert!((1..=4).contains(&parameters.heads) && sources.len() == parameters.heads);
        assert!(parameters.temperature <= MAX_TEMPERATURE);
        assert!(
            parameters.reward_on_reach.is_finite()
                && parameters.reward_on_reach >= 1.0
                && parameters.reward_on_reach <= 1000.0
        );
        assert!((0.0..=1.0).contains(&config.beta) && (0.0..=1.0).contains(&config.gamma));
        assert!((0.0..=1.0).contains(&config.initial_q));
        assert!((0.0..=parameters.reward_on_reach).contains(&config.initial_r));
        assert!(replay.capacity > 0 && replay.warmup > 0 && replay.warmup <= replay.capacity);
        assert!(replay.updates_per_step > 0);
        Self {
            config,
            replay,
            parameters,
            heads: sources
                .into_iter()
                .map(|rng| Head {
                    population: Population::new(),
                    rng,
                    counts: UpdateCounts::default(),
                    episodes: 0,
                })
                .collect(),
            store: TrajectoryStore::new(replay.capacity),
            sampler: Sampler::new(
                SamplerConfiguration {
                    selection: Selection {
                        strategy: GoalStrategy::Future,
                        admissibility: parameters.hindsight.admissibility,
                        candidate_filter: false,
                    },
                    relabeled_proportion: parameters.hindsight.relabeled_proportion,
                    truncation: replay.truncation,
                },
                sampler_rng,
            ),
            acceptance_rng,
            accepted: ReplayCounters::default(),
            priority: PriorityCounts::default(),
            episodes: 0,
        }
    }

    pub fn with_populations(mut self, populations: Vec<Population<M>>) -> Self {
        assert_eq!(self.episodes, 0);
        assert_eq!(populations.len(), self.heads.len());
        for (head, population) in self.heads.iter_mut().zip(populations) {
            head.population = population;
        }
        self
    }

    pub fn heads(&self) -> &[Head<M, R>] {
        &self.heads
    }
    pub fn config(&self) -> &Configuration {
        &self.config
    }
    pub fn parameters(&self) -> ConsistencyConfiguration {
        self.parameters
    }
    pub fn replay_configuration(&self) -> EpisodicConfiguration {
        self.replay
    }
    pub fn store(&self) -> &TrajectoryStore<S, G> {
        &self.store
    }
    pub fn sampler(&self) -> &Sampler<P> {
        &self.sampler
    }
    pub fn accepted_counters(&self) -> &ReplayCounters {
        &self.accepted
    }
    pub fn priority_counts(&self) -> &PriorityCounts {
        &self.priority
    }
    pub fn playing_head(&self) -> usize {
        self.episodes as usize % self.heads.len()
    }

    pub fn counts(&self) -> UpdateCounts {
        let mut result = UpdateCounts::default();
        for head in &self.heads {
            result.full += head.counts.full;
            result.values_only += head.counts.values_only;
            result.environment_steps += head.counts.environment_steps;
            result.skipped_warmup_steps += head.counts.skipped_warmup_steps;
        }
        result
    }

    pub fn snapshot(&self) -> String {
        let heads: Vec<_> = self
            .heads
            .iter()
            .map(|head| {
                (
                    head.population.classifiers(),
                    head.rng.capture_state(),
                    head.counts,
                    head.episodes,
                )
            })
            .collect();
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.config,
            self.replay,
            self.parameters,
            heads,
            self.store,
            self.sampler.configuration(),
            self.sampler.random_source().capture_state(),
            self.sampler.counters(),
            self.acceptance_rng.capture_state(),
            self.accepted,
            self.priority,
            self.episodes
        )
    }

    pub fn explore_action(&mut self, state: &Perception<M>) -> usize {
        let index = self.playing_head();
        let head = &mut self.heads[index];
        let matched = head.population.form_match_set(state);
        EpsilonGreedy {
            number_of_possible_actions: self.config.number_of_possible_actions,
            epsilon: self.config.epsilon,
        }
        .select(&head.population, &matched, &mut head.rng)
    }

    pub fn greedy_action(
        &self,
        index: usize,
        state: &Perception<M>,
        rng: &mut dyn RandomSource,
    ) -> (usize, f64) {
        let population = &self.heads[index].population;
        let matched = population.form_match_set(state);
        let action = BestAction {
            number_of_possible_actions: self.config.number_of_possible_actions,
        }
        .select(population, &matched, rng);
        let action_set = population.form_action_set(&matched, action);
        (
            action,
            MaxFitnessBootstrap.estimate(population, &action_set),
        )
    }

    pub fn variance(&self, sample: &ReplaySample<M>) -> f64 {
        if self.heads.len() == 1 {
            return 0.0;
        }
        let values: Vec<_> = self
            .heads
            .iter()
            .map(|head| {
                let value = action_value(&head.population, sample);
                assert!(
                    (0.0..=self.parameters.reward_on_reach).contains(&value),
                    "VCP value exceeds the registered scale"
                );
                value
            })
            .collect();
        population_variance(&values)
    }

    pub fn draw_batch(
        &mut self,
        count: usize,
        evaluator: &impl GoalEvaluator<S, G>,
    ) -> Vec<DrawnSample<G, M>> {
        self.draw_batch_observed(count, evaluator, &mut |_, _| {})
    }

    pub fn draw_batch_observed(
        &mut self,
        count: usize,
        evaluator: &impl GoalEvaluator<S, G>,
        observe: &mut impl FnMut(f64, bool),
    ) -> Vec<DrawnSample<G, M>> {
        let mut draws = Vec::with_capacity(count);
        let maximum = self.parameters.variance_maximum();
        let mut since_acceptance = 0u64;
        while draws.len() < count {
            assert!(since_acceptance < 1_000_000, "VCP cannot draw: one million consecutive rejected proposals; abort without a fallback");
            since_acceptance += 1;
            let mut proposal = self
                .sampler
                .draw(&self.store, 1, evaluator)
                .expect("VCP proposal failed");
            let drawn = proposal.pop().expect("one proposal");
            assert!(
                drawn.scored.sample.reward == 0.0
                    || (drawn.scored.sample.done
                        && drawn.scored.sample.reward <= self.parameters.reward_on_reach),
                "VCP requires sparse terminal rewards within its registered scale"
            );
            let measure_variance = self.parameters.temperature > 0;
            let variance = if measure_variance {
                self.variance(&drawn.scored.sample)
            } else {
                0.0
            };
            let probability = acceptance(variance, maximum, self.parameters.temperature);
            self.priority.proposals += 1;
            self.priority.value_queries += if !measure_variance || self.heads.len() == 1 {
                0
            } else {
                self.heads.len() as u64
            };
            self.priority.acceptance_sum += probability;
            if measure_variance {
                self.priority.variance.record(variance, maximum);
                let base = (maximum - variance + priority::EPSILON) / (maximum + priority::EPSILON);
                let mut weight = 1.0;
                for sum in &mut self.priority.candidate_acceptance_sums {
                    weight *= base;
                    *sum += weight;
                }
            }
            let accepted = if probability == 1.0 {
                true
            } else {
                self.priority.acceptance_coins += 1;
                self.acceptance_rng.gen_bool(probability)
            };
            if measure_variance {
                observe(variance, accepted);
            }
            if accepted {
                since_acceptance = 0;
                self.priority.accepted += 1;
                if measure_variance {
                    self.priority.accepted_variance.record(variance, maximum);
                }
                record_accepted(&mut self.accepted, &drawn);
                draws.push(drawn);
            }
        }
        self.priority.batch_peak_bytes = self
            .priority
            .batch_peak_bytes
            .max(draws.len() * std::mem::size_of::<DrawnSample<G, M>>());
        draws
    }

    pub fn learn_draw(&mut self, head: usize, drawn: &DrawnSample<G, M>, time: u64) {
        let head = &mut self.heads[head];
        if drawn.provenance.origin == SampleOrigin::Relabeled
            && self.parameters.hindsight.copy_update == CopyUpdate::ValuesOnly
        {
            values_learning_step(&mut head.population, &self.config, &drawn.scored.sample);
            head.counts.values_only += 1;
        } else {
            replay_learning_step(
                &mut head.population,
                &self.config,
                &MaxFitnessBootstrap,
                &drawn.scored.sample,
                time,
                &mut head.rng,
            );
            head.counts.full += 1;
        }
    }

    pub fn train_episode(&mut self, env: &mut impl ReplayEnvironment<S, G, M>, time: u64) {
        self.train_episode_observed(env, time, &mut |_, _| {});
    }

    pub fn train_episode_observed(
        &mut self,
        env: &mut impl ReplayEnvironment<S, G, M>,
        time: u64,
        observe: &mut impl FnMut(f64, bool),
    ) {
        let index = self.playing_head();
        let mut state = env.reset();
        let start = env.episode_start();
        assert_eq!(
            state,
            GoalLayout::<S, G, M>::join(&start.observation, &start.desired)
        );
        let mut episode = self.store.begin_episode(start);
        loop {
            let action = self.explore_action(&state);
            let outcome = env.step(action);
            let step = env.last_step();
            assert_eq!(
                outcome.observation,
                GoalLayout::<S, G, M>::join(&step.observation, &start.desired)
            );
            episode.push(action, step);
            state = outcome.observation;
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
        let steps = episode.steps().len();
        self.store
            .insert(episode)
            .expect("complete episode must fit the trajectory capacity");
        self.heads[index].counts.environment_steps += steps as u64;
        self.heads[index].episodes += 1;
        self.episodes += 1;
        if self.store.len() < self.replay.warmup {
            self.heads[index].counts.skipped_warmup_steps += steps as u64;
            return;
        }
        let count = steps
            .checked_mul(self.replay.updates_per_step)
            .expect("VCP draw count overflow");
        let draws = self.draw_batch_observed(count, env, observe);
        for drawn in &draws {
            self.learn_draw(index, drawn, time + steps as u64);
        }
    }
}

fn record_accepted<const G: usize, const M: usize>(
    counters: &mut ReplayCounters,
    drawn: &DrawnSample<G, M>,
) {
    let provenance = drawn.provenance;
    let route = match provenance.origin {
        SampleOrigin::Original => &mut counters.original,
        SampleOrigin::Relabeled => &mut counters.relabeled,
    };
    route.draws += 1;
    route.already_reached += u64::from(provenance.already_reached);
    route.after_counterfactual_end += u64::from(provenance.after_counterfactual_end);
    route.done += u64::from(provenance.done);
    route.outside_candidates += u64::from(provenance.outside_candidates);
    route.reward += drawn.scored.sample.reward;
    counters.fallbacks += u64::from(provenance.fallback);
    counters.strategies[provenance.strategy.index()] += 1;
    counters.cost.add(drawn.cost);
}
