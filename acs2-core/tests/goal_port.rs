use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::config::Configuration;
use acs2_core::environment::Environment;
use acs2_core::goal::{
    ExactMatch, Goal, GoalConditioned, GoalEnvironment, GoalLayout, GoalObjective, GoalStart,
    GoalStep,
};
use acs2_core::perception::Perception;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;

const LENGTH: u8 = 6;
const TIME_LIMIT: u32 = 12;
const MOVE_LEFT: usize = 0;
const MOVE_RIGHT: usize = 1;
const REWARD_ON_REACH: f64 = 1000.0;

fn position_symbol(position: u8) -> Symbol {
    Symbol::Token(b'0' + position)
}

fn position_of(goal: &Goal<1>) -> u8 {
    match goal.symbols[0] {
        Symbol::Token(value) => value - b'0',
        Symbol::Wildcard => panic!("a corridor goal is a position"),
    }
}

struct ReachOrPass;

impl GoalObjective<1> for ReachOrPass {
    fn reward(&self, achieved: &Goal<1>, desired: &Goal<1>) -> f64 {
        if self.is_reached(achieved, desired) {
            100.0 * f64::from(position_of(achieved))
        } else {
            0.0
        }
    }

    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        position_of(achieved) >= position_of(desired)
    }
}

struct Corridor<O> {
    rng: ChaChaRandomSource,
    objective: O,
    position: u8,
    elapsed: u32,
}

impl Corridor<ExactMatch> {
    fn new(seed: u64) -> Self {
        Self::with_objective(
            seed,
            ExactMatch {
                reward_on_reach: REWARD_ON_REACH,
            },
        )
    }
}

impl<O> Corridor<O> {
    fn with_objective(seed: u64, objective: O) -> Self {
        Self {
            rng: ChaChaRandomSource::from_seed_and_stream(seed, 2),
            objective,
            position: 0,
            elapsed: 0,
        }
    }

    fn observe(&self) -> (Perception<1>, Goal<1>) {
        let symbol = position_symbol(self.position);
        (Perception::new([symbol]), Goal::new([symbol]))
    }

    fn start(&mut self, desired: Goal<1>) -> GoalStart<1, 1> {
        let offset = 1 + self.rng.gen_range(LENGTH as usize - 1) as u8;
        self.position = (position_of(&desired) + offset) % LENGTH;
        self.elapsed = 0;
        let (observation, achieved) = self.observe();
        GoalStart {
            observation,
            achieved,
            desired,
        }
    }
}

impl<O: GoalObjective<1>> GoalEnvironment<1, 1> for Corridor<O> {
    type Objective = O;

    fn objective(&self) -> &O {
        &self.objective
    }

    fn reset(&mut self) -> GoalStart<1, 1> {
        let goal_position = self.rng.gen_range(LENGTH as usize) as u8;
        self.start(Goal::new([position_symbol(goal_position)]))
    }

    fn reset_with_goal(&mut self, desired: Goal<1>) -> GoalStart<1, 1> {
        self.start(desired)
    }

    fn step(&mut self, action: usize) -> GoalStep<1, 1> {
        self.position = match action {
            MOVE_LEFT => self.position.saturating_sub(1),
            MOVE_RIGHT => (self.position + 1).min(LENGTH - 1),
            other => panic!("a corridor has no action {other}"),
        };
        self.elapsed += 1;
        let (observation, achieved) = self.observe();
        GoalStep {
            observation,
            achieved,
            terminal_state: false,
            time_limit_reached: self.elapsed >= TIME_LIMIT,
        }
    }
}

fn corridor_config() -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        ..Configuration::default_protocol()
    }
}

fn population_signature(seed: u64, trials: u32) -> Vec<String> {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::new(seed));
    let mut agent = Agent::<2, _>::new(
        corridor_config(),
        ChaChaRandomSource::from_seed_and_stream(seed, 1),
    );
    let selector = EpsilonGreedy {
        number_of_possible_actions: 2,
        epsilon: 0.8,
    };
    let mut time = 0u64;
    for _ in 0..trials {
        let metrics = agent.run_explore_trial(&mut env, &selector, &MaxFitnessBootstrap, time);
        assert!(metrics.steps >= 1 && metrics.steps <= TIME_LIMIT);
        time += metrics.steps as u64;
    }
    agent
        .population()
        .classifiers()
        .iter()
        .map(|classifier| {
            format!(
                "{:?}|{:?}|{:?}|{:016x}|{:016x}|{}",
                classifier.condition.symbols,
                classifier.action,
                classifier.effect.symbols,
                classifier.q.to_bits(),
                classifier.r.to_bits(),
                classifier.num,
            )
        })
        .collect()
}

#[test]
fn every_observation_carries_the_desired_goal_of_its_episode() {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::new(3));
    for _ in 0..20 {
        let first = env.reset();
        let desired = env.desired().expect("a reset sets the desired goal");
        assert_eq!(GoalLayout::<1, 1, 2>::split(&first).1, desired);
        for action in [MOVE_RIGHT, MOVE_RIGHT, MOVE_LEFT, MOVE_RIGHT] {
            let outcome = env.step(action);
            let (state, carried) = GoalLayout::<1, 1, 2>::split(&outcome.observation);
            assert_eq!(carried, desired);
            let achieved = Goal::new(state.symbols);
            let objective = env.environment().objective();
            assert_eq!(outcome.reward, objective.reward(&achieved, &desired));
            assert_eq!(outcome.terminated, objective.is_reached(&achieved, &desired));
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
    }
}

#[test]
fn the_time_limit_truncates_an_episode_that_never_reaches_its_goal() {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::new(5));
    let far_end = Goal::new([position_symbol(LENGTH - 1)]);
    env.reset_with_goal(far_end);
    assert_eq!(env.desired(), Some(far_end));
    for step in 1..=TIME_LIMIT {
        let outcome = env.step(MOVE_LEFT);
        assert_eq!(outcome.reward, 0.0);
        assert!(!outcome.terminated);
        assert_eq!(outcome.truncated, step == TIME_LIMIT);
    }
}

#[test]
#[should_panic(expected = "must be reset before it is stepped")]
fn stepping_before_a_reset_is_refused() {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::new(1));
    env.step(MOVE_RIGHT);
}

#[test]
#[should_panic(expected = "must be reset before it is stepped")]
fn stepping_after_the_episode_ended_is_refused() {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::with_objective(8, ReachOrPass));
    env.reset_with_goal(Goal::new([position_symbol(0)]));
    let outcome = env.step(MOVE_RIGHT);
    assert!(outcome.terminated);
    assert_eq!(env.desired(), None);
    env.step(MOVE_RIGHT);
}

#[test]
fn the_adapter_gives_the_objective_the_achieved_goal_before_the_desired_one() {
    let mut env = GoalConditioned::<_, 1, 1, 2>::new(Corridor::with_objective(8, ReachOrPass));
    env.reset_with_goal(Goal::new([position_symbol(0)]));
    let outcome = env.step(MOVE_RIGHT);
    let (state, carried) = GoalLayout::<1, 1, 2>::split(&outcome.observation);
    let achieved = Goal::new(state.symbols);
    assert_eq!(carried, Goal::new([position_symbol(0)]));
    assert!(position_of(&achieved) >= 2);
    assert!(outcome.terminated);
    assert_eq!(outcome.reward, 100.0 * f64::from(position_of(&achieved)));
}

#[test]
fn the_unchanged_acs2_agent_learns_a_goal_task_through_the_adapter() {
    let signature = population_signature(42, 300);
    assert!(!signature.is_empty());
    assert_eq!(signature, population_signature(42, 300));
    assert_ne!(signature, population_signature(43, 300));
}
