use crate::acs2er::Acs2ErAgent;
use crate::acs2er::replay::ReplayConfiguration;
use acs2_core::trial::{LearningAgent, TrialMetrics, TruncationMode};
use acs2_core::checkpoint::{AgentState, Checkpointed};
use acs2_core::learning::{action_selection::ActionSelector, rl::BootstrapEstimator};
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::representation::{
    perception::Perception,
    population::{ClassifierRef, Population},
};
use acs2_core::support::{
    config::Configuration,
    rng::{RandomSource, RngState},
};
use acs2_trajectory::ReplayCounters;
use std::{cell::RefCell, collections::VecDeque, mem::size_of, rc::Rc};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagnosticConfiguration {
    pub capacity: usize,
    pub warmup: usize,
}

impl DiagnosticConfiguration {
    pub fn replay(self) -> ReplayConfiguration {
        ReplayConfiguration {
            buffer_size: self.capacity,
            min_samples: self.warmup,
            samples_number: 1,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct StoredOutcome {
    reward: f64,
    done: bool,
}

#[derive(Debug, Default)]
struct Trace {
    outcomes: VecDeque<StoredOutcome>,
    pending: bool,
    counters: ReplayCounters,
}

struct ObservedRandom<R> {
    inner: R,
    trace: Rc<RefCell<Trace>>,
}

impl<R: RandomSource> RandomSource for ObservedRandom<R> {
    fn gen_bool(&mut self, probability: f64) -> bool {
        self.inner.gen_bool(probability)
    }
    fn gen_unit(&mut self) -> f64 {
        self.inner.gen_unit()
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        let chosen = self.inner.gen_range(bound);
        let mut trace = self.trace.borrow_mut();
        if trace.pending {
            assert_eq!(
                bound,
                trace.outcomes.len(),
                "first post-transition draw is core replay selection"
            );
            let sample = trace.outcomes[chosen];
            trace.counters.original.draws += 1;
            trace.counters.original.done += u64::from(sample.done);
            trace.counters.original.reward += sample.reward;
            trace.counters.strategies[0] += 1;
            trace.pending = false;
        }
        chosen
    }
    fn capture_state(&self) -> Option<RngState> {
        self.inner.capture_state()
    }
}

struct RecordingEnvironment<'a, E> {
    inner: &'a mut E,
    trace: Rc<RefCell<Trace>>,
    capacity: usize,
    truncation: TruncationMode,
}

struct ObservedSelector<'a, S> {
    inner: &'a S,
    trace: Rc<RefCell<Trace>>,
}

impl<S: ActionSelector<M>, const M: usize> ActionSelector<M> for ObservedSelector<'_, S> {
    fn select(
        &self,
        population: &Population<M>,
        matched: &[ClassifierRef],
        rng: &mut dyn RandomSource,
    ) -> usize {
        self.trace.borrow_mut().pending = false;
        self.inner.select(population, matched, rng)
    }
}

impl<E: Environment<M>, const M: usize> Environment<M> for RecordingEnvironment<'_, E> {
    fn reset(&mut self) -> Perception<M> {
        self.inner.reset()
    }
    fn step(&mut self, action: usize) -> StepOutcome<M> {
        let outcome = self.inner.step(action);
        let mut trace = self.trace.borrow_mut();
        if trace.outcomes.len() == self.capacity {
            trace.outcomes.pop_front();
        }
        trace.outcomes.push_back(StoredOutcome {
            reward: outcome.reward,
            done: self
                .truncation
                .is_terminal(outcome.terminated, outcome.truncated),
        });
        trace.pending = true;
        outcome
    }
}

pub struct Agent<const M: usize, R: RandomSource> {
    core: Acs2ErAgent<M, ObservedRandom<R>>,
    trace: Rc<RefCell<Trace>>,
    parameters: DiagnosticConfiguration,
    truncation: TruncationMode,
}

impl<const M: usize, R: RandomSource> Agent<M, R> {
    pub fn new(
        config: Configuration,
        parameters: DiagnosticConfiguration,
        truncation: TruncationMode,
        rng: R,
    ) -> Self {
        let trace = Rc::new(RefCell::new(Trace::default()));
        Self {
            core: Acs2ErAgent::new(
                config,
                parameters.replay(),
                ObservedRandom {
                    inner: rng,
                    trace: Rc::clone(&trace),
                },
            )
            .with_truncation_mode(truncation),
            trace,
            parameters,
            truncation,
        }
    }
    pub fn counters(&self) -> ReplayCounters {
        self.trace.borrow().counters.clone()
    }
    pub fn captured_state(&self) -> AgentState<M> {
        self.core.capture()
    }
    pub fn snapshot(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}|{:?}",
            self.core.capture(),
            self.trace.borrow(),
            self.parameters,
            self.truncation
        )
    }
    pub fn observer_logical_bytes(&self) -> usize {
        size_of::<Trace>() + self.trace.borrow().outcomes.len() * size_of::<StoredOutcome>()
    }
}

impl<const M: usize, R: RandomSource> LearningAgent<M> for Agent<M, R> {
    fn population(&self) -> &Population<M> {
        self.core.population()
    }
    fn config(&self) -> &Configuration {
        self.core.config()
    }
    fn run_explore_trial<E, S, B>(
        &mut self,
        env: &mut E,
        selector: &S,
        bootstrap: &B,
        time: u64,
    ) -> TrialMetrics
    where
        E: Environment<M>,
        S: ActionSelector<M>,
        B: BootstrapEstimator<M>,
    {
        let mut recording = RecordingEnvironment {
            inner: env,
            trace: Rc::clone(&self.trace),
            capacity: self.parameters.capacity,
            truncation: self.truncation,
        };
        let observed = ObservedSelector {
            inner: selector,
            trace: Rc::clone(&self.trace),
        };
        self.core
            .run_explore_trial(&mut recording, &observed, bootstrap, time)
    }
    fn run_exploit_trial<E, B>(&mut self, env: &mut E, bootstrap: &B, time: u64) -> TrialMetrics
    where
        E: Environment<M>,
        B: BootstrapEstimator<M>,
    {
        self.trace.borrow_mut().pending = false;
        self.core.run_exploit_trial(env, bootstrap, time)
    }
}
