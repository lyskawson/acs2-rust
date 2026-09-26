use acs2_core::goal::{ExactMatch, Goal, GoalEnvironment, GoalStart, GoalStep};
use acs2_core::knowledge::Transition;
use acs2_core::perception::Perception;
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;

use super::SPARSE_GOAL_REWARD;

pub type Position = (usize, usize);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandEyeState {
    pub gripper: Position,
    pub block: Position,
    pub held: bool,
}

impl HandEyeState {
    fn validate<const SIDE: usize>(&self) {
        for position in [self.gripper, self.block] {
            assert!(
                position.0 < SIDE && position.1 < SIDE,
                "position must be inside the grid"
            );
        }
        assert!(
            !self.held || self.gripper == self.block,
            "held block must be at the gripper"
        );
    }

    pub fn after_action<const SIDE: usize>(mut self, action: usize) -> Self {
        self.validate::<SIDE>();
        match action {
            0 => self.gripper.1 = self.gripper.1.saturating_sub(1),
            1 => self.gripper.0 = (self.gripper.0 + 1).min(SIDE - 1),
            2 => self.gripper.1 = (self.gripper.1 + 1).min(SIDE - 1),
            3 => self.gripper.0 = self.gripper.0.saturating_sub(1),
            4 => {
                if self.gripper == self.block {
                    self.held = true;
                }
            }
            5 => self.held = false,
            _ => panic!("hand-eye action must be in 0..6"),
        }
        if self.held {
            self.block = self.gripper;
        }
        self
    }

    pub fn perception<const SIDE: usize, const S: usize>(&self) -> Perception<S> {
        let () = HandEye::<SIDE, S>::LAYOUT;
        self.validate::<SIDE>();
        let mut symbols = [Symbol::Token(b'w'); S];
        symbols[self.block.1 * SIDE + self.block.0] = Symbol::Token(b'b');
        if !self.held {
            symbols[self.gripper.1 * SIDE + self.gripper.0] = Symbol::Token(b'g');
        }
        symbols[S - 1] = Symbol::Token(if self.held {
            b'2'
        } else if self.gripper == self.block {
            b'1'
        } else {
            b'0'
        });
        Perception::new(symbols)
    }
}

pub type HandEye3 = HandEye<3, 10>;
pub type HandEye4 = HandEye<4, 17>;
pub type HandEye5 = HandEye<5, 26>;

pub struct HandEye<const SIDE: usize, const S: usize> {
    state: HandEyeState,
    goals: Vec<Goal<2>>,
    desired: Option<Goal<2>>,
    step_cap: u32,
    elapsed_steps: u32,
    rng: Box<dyn RandomSource>,
}

impl<const SIDE: usize, const S: usize> HandEye<SIDE, S> {
    const LAYOUT: () = assert!(
        SIDE >= 2 && SIDE <= 256 && S == SIDE * SIDE + 1,
        "hand-eye requires S = SIDE * SIDE + 1 and SIDE in 2..=256"
    );

    pub fn new(step_cap: u32, rng: Box<dyn RandomSource>) -> Self {
        let () = Self::LAYOUT;
        assert!(step_cap > 0, "step cap must be positive");
        Self {
            state: HandEyeState {
                gripper: (0, 0),
                block: (0, 0),
                held: false,
            },
            goals: (0..SIDE * SIDE)
                .map(|index| position_goal((index % SIDE, index / SIDE)))
                .collect(),
            desired: None,
            step_cap,
            elapsed_steps: 0,
            rng,
        }
    }

    pub fn state(&self) -> HandEyeState {
        self.state
    }

    pub fn step_cap(&self) -> u32 {
        self.step_cap
    }

    pub fn goal_pool(&self) -> &[Goal<2>] {
        &self.goals
    }

    pub fn distance_to_goal(&self, start: HandEyeState, desired: &Goal<2>) -> Option<u32> {
        if !self.goals.contains(desired) || !valid_state::<SIDE>(start) {
            return None;
        }
        let target = goal_position(desired);
        if start.block == target {
            return Some(0);
        }
        let transport = manhattan(start.block, target);
        Some(if start.held {
            transport
        } else {
            manhattan(start.gripper, start.block) + 1 + transport
        })
    }

    pub fn reset_at(&mut self, state: HandEyeState, desired: Goal<2>) -> GoalStart<S, 2> {
        state.validate::<SIDE>();
        assert!(
            self.goals.contains(&desired),
            "desired goal must belong to the grid"
        );
        assert_ne!(
            position_goal(state.block),
            desired,
            "start block must differ from the goal"
        );
        self.state = state;
        self.desired = Some(desired);
        self.elapsed_steps = 0;
        GoalStart {
            observation: state.perception::<SIDE, S>(),
            achieved: position_goal(state.block),
            desired,
        }
    }

    pub fn states(&self) -> impl Iterator<Item = HandEyeState> {
        (0..SIDE * SIDE)
            .flat_map(|gripper| {
                (0..SIDE * SIDE).map(move |block| HandEyeState {
                    gripper: (gripper % SIDE, gripper / SIDE),
                    block: (block % SIDE, block / SIDE),
                    held: false,
                })
            })
            .chain((0..SIDE * SIDE).map(|block| HandEyeState {
                gripper: (block % SIDE, block / SIDE),
                block: (block % SIDE, block / SIDE),
                held: true,
            }))
    }

    pub fn knowledge_transitions(&self) -> Vec<Transition<S>> {
        let mut transitions = Vec::new();
        for state in self.states() {
            let p0 = state.perception::<SIDE, S>();
            for action in 0..6 {
                let p1 = state.after_action::<SIDE>(action).perception::<SIDE, S>();
                if p0 != p1 {
                    transitions.push(Transition::new(p0, action, p1));
                }
            }
        }
        transitions
    }

    fn random_state(&mut self) -> HandEyeState {
        let block = (self.rng.gen_range(SIDE), self.rng.gen_range(SIDE));
        let held = self.rng.gen_bool(0.5);
        let gripper = if held {
            block
        } else {
            (self.rng.gen_range(SIDE), self.rng.gen_range(SIDE))
        };
        HandEyeState {
            gripper,
            block,
            held,
        }
    }
}

impl<const SIDE: usize, const S: usize> GoalEnvironment<S, 2> for HandEye<SIDE, S> {
    type Objective = ExactMatch;

    fn objective(&self) -> &Self::Objective {
        &SPARSE_GOAL_REWARD
    }

    fn reset(&mut self) -> GoalStart<S, 2> {
        let state = self.random_state();
        let excluded = state.block.1 * SIDE + state.block.0;
        let chosen = self.rng.gen_range(self.goals.len() - 1);
        self.reset_at(state, self.goals[chosen + usize::from(chosen >= excluded)])
    }

    fn reset_with_goal(&mut self, desired: Goal<2>) -> GoalStart<S, 2> {
        assert!(
            self.goals.contains(&desired),
            "desired goal must belong to the grid"
        );
        loop {
            let state = self.random_state();
            if position_goal(state.block) != desired {
                return self.reset_at(state, desired);
            }
        }
    }

    fn step(&mut self, action: usize) -> GoalStep<S, 2> {
        let desired = self.desired.expect("hand-eye must be reset before a step");
        self.state = self.state.after_action::<SIDE>(action);
        self.elapsed_steps += 1;
        let step = GoalStep {
            observation: self.state.perception::<SIDE, S>(),
            achieved: position_goal(self.state.block),
            terminal_state: false,
            time_limit_reached: self.elapsed_steps >= self.step_cap,
        };
        let result = step.outcome(self.objective(), &desired);
        if result.terminated || result.truncated {
            self.desired = None;
        }
        step
    }
}

pub fn position_goal(position: Position) -> Goal<2> {
    Goal::new([
        Symbol::Token(u8::try_from(position.0).expect("x must fit a symbol")),
        Symbol::Token(u8::try_from(position.1).expect("y must fit a symbol")),
    ])
}

fn goal_position(goal: &Goal<2>) -> Position {
    match goal.symbols {
        [Symbol::Token(x), Symbol::Token(y)] => (usize::from(x), usize::from(y)),
        _ => panic!("position goal must contain coordinates"),
    }
}

fn valid_state<const SIDE: usize>(state: HandEyeState) -> bool {
    [state.gripper, state.block]
        .iter()
        .all(|&(x, y)| x < SIDE && y < SIDE)
        && (!state.held || state.gripper == state.block)
}

fn manhattan(first: Position, second: Position) -> u32 {
    (first.0.abs_diff(second.0) + first.1.abs_diff(second.1)) as u32
}
