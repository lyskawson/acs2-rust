use std::collections::VecDeque;

use acs2_core::goal::{ExactMatch, Goal, GoalEnvironment, GoalStart, GoalStep};
use acs2_core::knowledge::Transition;
use acs2_core::perception::Perception;
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;

use super::SPARSE_GOAL_REWARD;

pub const STANDS: [(usize, usize); 4] = [(0, 0), (0, 4), (4, 0), (4, 3)];

const EAST_OPEN: [[bool; 4]; 5] = [
    [true, false, true, true],
    [true, false, true, true],
    [true, true, true, true],
    [false, true, false, true],
    [false, true, false, true],
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TaxiState {
    pub row: usize,
    pub col: usize,
    pub passenger: usize,
}

impl TaxiState {
    fn valid(&self) -> bool {
        self.row < 5 && self.col < 5 && self.passenger < 5
    }

    pub fn after_action(mut self, action: usize) -> Self {
        assert!(self.valid(), "taxi state must be inside the map");
        match action {
            0 => self.row = (self.row + 1).min(4),
            1 => self.row = self.row.saturating_sub(1),
            2 => {
                if self.col < 4 && EAST_OPEN[self.row][self.col] {
                    self.col += 1;
                }
            }
            3 => {
                if self.col > 0 && EAST_OPEN[self.row][self.col - 1] {
                    self.col -= 1;
                }
            }
            4 => {
                if self.passenger < 4 && (self.row, self.col) == STANDS[self.passenger] {
                    self.passenger = 4;
                }
            }
            5 => {
                if self.passenger == 4 {
                    if let Some(stand) = STANDS
                        .iter()
                        .position(|&position| position == (self.row, self.col))
                    {
                        self.passenger = stand;
                    }
                }
            }
            _ => panic!("taxi action must be in 0..6"),
        }
        self
    }

    pub fn perception(&self) -> Perception<3> {
        assert!(self.valid(), "taxi state must be inside the map");
        Perception::new([
            Symbol::Token(self.row as u8),
            Symbol::Token(self.col as u8),
            Symbol::Token(self.passenger as u8),
        ])
    }

    pub fn achieved(&self) -> Goal<1> {
        assert!(self.valid(), "taxi state must be inside the map");
        passenger_goal(self.passenger)
    }

    fn index(&self) -> usize {
        (self.row * 5 + self.col) * 5 + self.passenger
    }
}

pub struct Taxi {
    state: TaxiState,
    goals: [Goal<1>; 4],
    desired: Option<Goal<1>>,
    step_cap: u32,
    elapsed_steps: u32,
    rng: Box<dyn RandomSource>,
}

impl Taxi {
    pub fn new(step_cap: u32, rng: Box<dyn RandomSource>) -> Self {
        assert!(step_cap > 0, "step cap must be positive");
        Self {
            state: TaxiState {
                row: 0,
                col: 0,
                passenger: 0,
            },
            goals: core::array::from_fn(passenger_goal),
            desired: None,
            step_cap,
            elapsed_steps: 0,
            rng,
        }
    }

    pub fn state(&self) -> TaxiState {
        self.state
    }

    pub fn step_cap(&self) -> u32 {
        self.step_cap
    }

    pub fn goal_pool(&self) -> &[Goal<1>] {
        &self.goals
    }

    pub fn states(&self) -> impl Iterator<Item = TaxiState> {
        (0..125).map(|index| TaxiState {
            row: index / 25,
            col: (index / 5) % 5,
            passenger: index % 5,
        })
    }

    pub fn distance_to_goal(&self, start: TaxiState, desired: &Goal<1>) -> Option<u32> {
        if !start.valid() || !self.goals.contains(desired) {
            return None;
        }
        let mut distances = [None; 125];
        distances[start.index()] = Some(0);
        let mut queue = VecDeque::from([start]);
        while let Some(state) = queue.pop_front() {
            let distance = distances[state.index()].unwrap();
            if state.achieved() == *desired {
                return Some(distance);
            }
            for action in 0..6 {
                let next = state.after_action(action);
                if distances[next.index()].is_none() {
                    distances[next.index()] = Some(distance + 1);
                    queue.push_back(next);
                }
            }
        }
        None
    }

    pub fn reset_at(&mut self, state: TaxiState, desired: Goal<1>) -> GoalStart<3, 1> {
        assert!(state.valid(), "taxi state must be inside the map");
        assert!(
            self.goals.contains(&desired),
            "desired goal must be a stand"
        );
        assert_ne!(
            state.achieved(),
            desired,
            "start passenger must differ from the goal"
        );
        self.state = state;
        self.desired = Some(desired);
        self.elapsed_steps = 0;
        GoalStart {
            observation: state.perception(),
            achieved: state.achieved(),
            desired,
        }
    }

    pub fn knowledge_transitions(&self) -> Vec<Transition<3>> {
        let mut transitions = Vec::new();
        for state in self.states() {
            for action in 0..6 {
                let next = state.after_action(action);
                if state != next {
                    transitions.push(Transition::new(
                        state.perception(),
                        action,
                        next.perception(),
                    ));
                }
            }
        }
        transitions
    }
}

impl GoalEnvironment<3, 1> for Taxi {
    type Objective = ExactMatch;

    fn objective(&self) -> &Self::Objective {
        &SPARSE_GOAL_REWARD
    }

    fn reset(&mut self) -> GoalStart<3, 1> {
        let desired = self.goals[self.rng.gen_range(4)];
        self.reset_with_goal(desired)
    }

    fn reset_with_goal(&mut self, desired: Goal<1>) -> GoalStart<3, 1> {
        let excluded = self
            .goals
            .iter()
            .position(|goal| *goal == desired)
            .expect("desired goal must be a stand");
        let chosen = self.rng.gen_range(3);
        let state = TaxiState {
            row: self.rng.gen_range(5),
            col: self.rng.gen_range(5),
            passenger: chosen + usize::from(chosen >= excluded),
        };
        self.reset_at(state, desired)
    }

    fn step(&mut self, action: usize) -> GoalStep<3, 1> {
        let desired = self.desired.expect("taxi must be reset before a step");
        self.state = self.state.after_action(action);
        self.elapsed_steps += 1;
        let step = GoalStep {
            observation: self.state.perception(),
            achieved: self.state.achieved(),
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

pub fn passenger_goal(passenger: usize) -> Goal<1> {
    assert!(
        passenger < 5,
        "passenger location must be a stand or in the taxi"
    );
    Goal::new([Symbol::Token(passenger as u8)])
}
