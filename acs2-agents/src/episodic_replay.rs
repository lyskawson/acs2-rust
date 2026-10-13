use acs2_core::learning::replay::{replay_learning_step, ReplaySample};
use acs2_core::trial::TruncationMode;
use acs2_core::learning::action_selection::{ActionSelector, BestAction, EpsilonGreedy};
use acs2_core::learning::rl::{
    apply_reinforcement_learning, BootstrapEstimator, MaxFitnessBootstrap,
};
use acs2_core::ports::environment::Environment;
use acs2_core::ports::goal::{GoalLayout, GoalStart, GoalStep};
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{
    DrawnSample, GoalEvaluator, SampleOrigin, Sampler, SamplerConfiguration, TrajectoryStore,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EpisodicConfiguration {
    pub capacity: usize,
    pub warmup: usize,
    pub updates_per_step: usize,
    pub truncation: TruncationMode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CopyUpdate {
    Full,
    ValuesOnly,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct UpdateCounts {
    pub full: u64,
    pub values_only: u64,
    pub skipped_warmup_steps: u64,
    pub environment_steps: u64,
}

pub trait ReplayEnvironment<const S: usize, const G: usize, const M: usize>:
    Environment<M> + GoalEvaluator<S, G>
{
    fn episode_start(&self) -> GoalStart<S, G>;
    fn last_step(&self) -> GoalStep<S, G>;
}

pub struct EpisodicReplay<const S: usize, const G: usize, const M: usize, R, P> {
    population: Population<M>,
    config: Configuration,
    replay: EpisodicConfiguration,
    copy_update: CopyUpdate,
    store: TrajectoryStore<S, G>,
    rng: R,
    sampler: Sampler<P>,
    counts: UpdateCounts,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    EpisodicReplay<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        sampler: SamplerConfiguration,
        copy_update: CopyUpdate,
        rng: R,
        sampler_rng: P,
    ) -> Self {
        assert!(replay.capacity > 0 && replay.warmup > 0 && replay.warmup <= replay.capacity);
        assert!(replay.updates_per_step > 0);
        assert_eq!(
            sampler.truncation, replay.truncation,
            "sampler and agent truncation must agree"
        );
        Self {
            population: Population::new(),
            config,
            replay,
            copy_update,
            store: TrajectoryStore::new(replay.capacity),
            rng,
            sampler: Sampler::new(sampler, sampler_rng),
            counts: UpdateCounts::default(),
        }
    }

    pub fn with_population(mut self, population: Population<M>) -> Self {
        assert_eq!(self.counts.environment_steps, 0);
        self.population = population;
        self
    }

    pub fn population(&self) -> &Population<M> {
        &self.population
    }

    pub fn config(&self) -> &Configuration {
        &self.config
    }

    pub fn replay_configuration(&self) -> EpisodicConfiguration {
        self.replay
    }

    pub fn copy_update(&self) -> CopyUpdate {
        self.copy_update
    }

    pub fn store(&self) -> &TrajectoryStore<S, G> {
        &self.store
    }

    pub fn sampler(&self) -> &Sampler<P> {
        &self.sampler
    }

    pub fn counts(&self) -> UpdateCounts {
        self.counts
    }

    pub fn snapshot(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            self.population.classifiers(),
            self.config,
            self.replay,
            self.copy_update,
            self.store,
            self.rng.capture_state(),
            self.sampler.configuration(),
            self.sampler.random_source().capture_state(),
            self.sampler.counters(),
            self.counts
        )
    }

    pub fn explore_action(&mut self, state: &Perception<M>) -> usize {
        let matched = self.population.form_match_set(state);
        EpsilonGreedy {
            number_of_possible_actions: self.config.number_of_possible_actions,
            epsilon: self.config.epsilon,
        }
        .select(&self.population, &matched, &mut self.rng)
    }

    pub fn greedy_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        let matched = self.population.form_match_set(state);
        let action = BestAction {
            number_of_possible_actions: self.config.number_of_possible_actions,
        }
        .select(&self.population, &matched, rng);
        let action_set = self.population.form_action_set(&matched, action);
        (
            action,
            MaxFitnessBootstrap.estimate(&self.population, &action_set),
        )
    }

    pub fn train_episode(&mut self, env: &mut impl ReplayEnvironment<S, G, M>, time: u64) {
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
        self.counts.environment_steps += steps as u64;
        if self.store.len() < self.replay.warmup {
            self.counts.skipped_warmup_steps += steps as u64;
            return;
        }
        let count = steps
            .checked_mul(self.replay.updates_per_step)
            .expect("replay draw count");
        let draws = self
            .sampler
            .draw(&self.store, count, env)
            .expect("valid task replay objective");
        for drawn in draws {
            self.learn_draw(&drawn, time + steps as u64);
        }
    }

    pub fn learn_draw(&mut self, drawn: &DrawnSample<G, M>, time: u64) {
        if drawn.provenance.origin == SampleOrigin::Relabeled
            && self.copy_update == CopyUpdate::ValuesOnly
        {
            values_learning_step(&mut self.population, &self.config, &drawn.scored.sample);
            self.counts.values_only += 1;
        } else {
            replay_learning_step(
                &mut self.population,
                &self.config,
                &MaxFitnessBootstrap,
                &drawn.scored.sample,
                time,
                &mut self.rng,
            );
            self.counts.full += 1;
        }
    }
}

pub fn values_learning_step<const M: usize>(
    population: &mut Population<M>,
    config: &Configuration,
    sample: &ReplaySample<M>,
) {
    let matched = population.form_match_set(&sample.state);
    let action_set = population.form_action_set(&matched, sample.action);
    let next_matched = population.form_match_set(&sample.next_state);
    let bootstrap = if sample.done {
        0.0
    } else {
        MaxFitnessBootstrap.estimate(population, &next_matched)
    };
    apply_reinforcement_learning(
        population,
        &action_set,
        sample.reward,
        bootstrap,
        config.beta,
        config.gamma,
    );
}

pub trait EpisodicAgent<const S: usize, const G: usize, const M: usize> {
    type PolicyRandom: RandomSource;
    type SamplerRandom: RandomSource;
    fn name(&self) -> &'static str;
    fn learner(&self) -> &EpisodicReplay<S, G, M, Self::PolicyRandom, Self::SamplerRandom>;
    fn learner_mut(
        &mut self,
    ) -> &mut EpisodicReplay<S, G, M, Self::PolicyRandom, Self::SamplerRandom>;
}
