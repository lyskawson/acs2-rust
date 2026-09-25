use core::ops::Range;

use crate::environment::{Environment, StepOutcome};
use crate::perception::Perception;
use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct Goal<const G: usize> {
    pub symbols: [Symbol; G],
}

impl<const G: usize> Goal<G> {
    pub fn new(symbols: [Symbol; G]) -> Self {
        Self { symbols }
    }
}

pub trait GoalObjective<const G: usize> {
    fn reward(&self, achieved: &Goal<G>, desired: &Goal<G>) -> f64;
    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool;
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ExactMatch {
    pub reward_on_reach: f64,
}

impl<const G: usize> GoalObjective<G> for ExactMatch {
    fn reward(&self, achieved: &Goal<G>, desired: &Goal<G>) -> f64 {
        if achieved == desired {
            self.reward_on_reach
        } else {
            0.0
        }
    }

    fn is_reached(&self, achieved: &Goal<G>, desired: &Goal<G>) -> bool {
        achieved == desired
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GoalStart<const S: usize, const G: usize> {
    pub observation: Perception<S>,
    pub achieved: Goal<G>,
    pub desired: Goal<G>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GoalStep<const S: usize, const G: usize> {
    pub observation: Perception<S>,
    pub achieved: Goal<G>,
    pub terminal_state: bool,
    pub time_limit_reached: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct GoalOutcome {
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
}

impl<const S: usize, const G: usize> GoalStep<S, G> {
    pub fn outcome<O>(&self, objective: &O, desired: &Goal<G>) -> GoalOutcome
    where
        O: GoalObjective<G> + ?Sized,
    {
        let terminated = self.terminal_state || objective.is_reached(&self.achieved, desired);
        GoalOutcome {
            reward: objective.reward(&self.achieved, desired),
            terminated,
            truncated: !terminated && self.time_limit_reached,
        }
    }
}

pub trait GoalEnvironment<const S: usize, const G: usize> {
    type Objective: GoalObjective<G>;

    fn objective(&self) -> &Self::Objective;
    fn reset(&mut self) -> GoalStart<S, G>;
    fn reset_with_goal(&mut self, desired: Goal<G>) -> GoalStart<S, G>;
    fn step(&mut self, action: usize) -> GoalStep<S, G>;
}

pub struct GoalLayout<const S: usize, const G: usize, const M: usize>;

impl<const S: usize, const G: usize, const M: usize> GoalLayout<S, G, M> {
    const CONSISTENT: () = assert!(S + G == M, "a goal layout requires M = S + G");

    pub fn join(state: &Perception<S>, goal: &Goal<G>) -> Perception<M> {
        let () = Self::CONSISTENT;
        Perception::new(core::array::from_fn(|index| {
            if index < S {
                state.symbols[index]
            } else {
                goal.symbols[index - S]
            }
        }))
    }

    pub fn split(joined: &Perception<M>) -> (Perception<S>, Goal<G>) {
        let () = Self::CONSISTENT;
        (
            Perception::new(core::array::from_fn(|index| joined.symbols[index])),
            Goal::new(core::array::from_fn(|index| joined.symbols[S + index])),
        )
    }

    pub fn goal_positions() -> Range<usize> {
        let () = Self::CONSISTENT;
        S..M
    }
}

pub struct GoalConditioned<E, const S: usize, const G: usize, const M: usize> {
    environment: E,
    desired: Option<Goal<G>>,
}

impl<E, const S: usize, const G: usize, const M: usize> GoalConditioned<E, S, G, M>
where
    E: GoalEnvironment<S, G>,
{
    pub fn new(environment: E) -> Self {
        let _ = GoalLayout::<S, G, M>::goal_positions();
        Self {
            environment,
            desired: None,
        }
    }

    pub fn environment(&self) -> &E {
        &self.environment
    }

    pub fn environment_mut(&mut self) -> &mut E {
        &mut self.environment
    }

    pub fn into_environment(self) -> E {
        self.environment
    }

    pub fn desired(&self) -> Option<Goal<G>> {
        self.desired
    }

    pub fn reset_with_goal(&mut self, desired: Goal<G>) -> Perception<M> {
        let start = self.environment.reset_with_goal(desired);
        self.begin(start)
    }

    fn begin(&mut self, start: GoalStart<S, G>) -> Perception<M> {
        self.desired = Some(start.desired);
        GoalLayout::<S, G, M>::join(&start.observation, &start.desired)
    }
}

impl<E, const S: usize, const G: usize, const M: usize> Environment<M> for GoalConditioned<E, S, G, M>
where
    E: GoalEnvironment<S, G>,
{
    fn reset(&mut self) -> Perception<M> {
        let start = self.environment.reset();
        self.begin(start)
    }

    fn step(&mut self, action: usize) -> StepOutcome<M> {
        let desired = self
            .desired
            .expect("a goal-conditioned environment must be reset before it is stepped");
        let step = self.environment.step(action);
        let outcome = step.outcome(self.environment.objective(), &desired);
        StepOutcome {
            observation: GoalLayout::<S, G, M>::join(&step.observation, &desired),
            reward: outcome.reward,
            terminated: outcome.terminated,
            truncated: outcome.truncated,
            info: (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token(value: u8) -> Symbol {
        Symbol::Token(b'0' + value)
    }

    fn goal(first: u8, second: u8) -> Goal<2> {
        Goal::new([token(first), token(second)])
    }

    fn step(achieved: Goal<2>, terminal_state: bool, time_limit_reached: bool) -> GoalStep<3, 2> {
        GoalStep {
            observation: Perception::new([token(1), token(2), token(3)]),
            achieved,
            terminal_state,
            time_limit_reached,
        }
    }

    const OBJECTIVE: ExactMatch = ExactMatch {
        reward_on_reach: 1000.0,
    };

    #[test]
    fn a_joined_perception_splits_back_into_its_state_and_goal() {
        let state = Perception::new([token(1), token(0), token(9)]);
        let desired = goal(4, 2);

        let joined = GoalLayout::<3, 2, 5>::join(&state, &desired);

        assert_eq!(
            joined.symbols,
            [token(1), token(0), token(9), token(4), token(2)]
        );
        assert_eq!(GoalLayout::<3, 2, 5>::split(&joined), (state, desired));
        assert_eq!(GoalLayout::<3, 2, 5>::goal_positions(), 3..5);
    }

    #[test]
    fn reaching_the_desired_goal_terminates_and_pays() {
        let outcome = step(goal(4, 2), false, false).outcome(&OBJECTIVE, &goal(4, 2));
        assert_eq!(
            outcome,
            GoalOutcome {
                reward: 1000.0,
                terminated: true,
                truncated: false
            }
        );
    }

    #[test]
    fn the_same_step_is_a_miss_under_another_goal() {
        let reached = step(goal(4, 2), false, false);
        let outcome = reached.outcome(&OBJECTIVE, &goal(4, 3));
        assert_eq!(
            outcome,
            GoalOutcome {
                reward: 0.0,
                terminated: false,
                truncated: false
            }
        );
    }

    #[test]
    fn a_goal_reached_on_the_time_limit_terminates_rather_than_truncates() {
        let on_limit = step(goal(4, 2), false, true);

        assert_eq!(
            on_limit.outcome(&OBJECTIVE, &goal(4, 2)),
            GoalOutcome {
                reward: 1000.0,
                terminated: true,
                truncated: false
            }
        );
        assert_eq!(
            on_limit.outcome(&OBJECTIVE, &goal(0, 0)),
            GoalOutcome {
                reward: 0.0,
                terminated: false,
                truncated: true
            }
        );
    }

    #[test]
    fn a_terminal_state_ends_the_episode_under_every_goal() {
        let fallen = step(goal(1, 1), true, false);

        for desired in [goal(1, 1), goal(3, 3)] {
            let outcome = fallen.outcome(&OBJECTIVE, &desired);
            assert!(outcome.terminated);
            assert!(!outcome.truncated);
            assert_eq!(outcome.reward, OBJECTIVE.reward(&goal(1, 1), &desired));
        }
    }
}
