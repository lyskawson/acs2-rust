use acs2_core::goal::{ExactMatch, Goal, GoalEnvironment, GoalStart, GoalStep};
use acs2_core::knowledge::Transition;
use acs2_core::perception::Perception;
use acs2_core::rng::RandomSource;
use acs2_core::symbol::Symbol;

use super::SPARSE_GOAL_REWARD;

pub struct BitFlipping<const N: usize> {
    state: Perception<N>,
    desired: Option<Goal<N>>,
    step_cap: u32,
    elapsed_steps: u32,
    rng: Box<dyn RandomSource>,
}

impl<const N: usize> BitFlipping<N> {
    pub fn new(rng: Box<dyn RandomSource>) -> Self {
        Self::with_step_cap(
            u32::try_from(N).expect("bit count must fit the step cap"),
            rng,
        )
    }

    pub fn with_step_cap(step_cap: u32, rng: Box<dyn RandomSource>) -> Self {
        assert!(N > 0, "bit flipping requires at least one bit");
        assert!(step_cap > 0, "step cap must be positive");
        Self {
            state: Perception::new([Symbol::Token(b'0'); N]),
            desired: None,
            step_cap,
            elapsed_steps: 0,
            rng,
        }
    }

    pub fn state(&self) -> Perception<N> {
        self.state
    }

    pub fn step_cap(&self) -> u32 {
        self.step_cap
    }

    pub fn goal_pool(&self) -> BitGoals<N> {
        BitGoals::new()
    }

    pub fn distance_to_goal(&self, start: &Perception<N>, desired: &Goal<N>) -> Option<u32> {
        if !valid_bits(&start.symbols) || !valid_bits(&desired.symbols) {
            return None;
        }
        u32::try_from(
            start
                .symbols
                .iter()
                .zip(desired.symbols)
                .filter(|(a, b)| **a != *b)
                .count(),
        )
        .ok()
    }

    pub fn reset_at(&mut self, state: Perception<N>, desired: Goal<N>) -> GoalStart<N, N> {
        assert!(
            valid_bits(&state.symbols) && valid_bits(&desired.symbols),
            "state and goal must be binary"
        );
        assert_ne!(
            state.symbols, desired.symbols,
            "start must differ from the goal"
        );
        self.state = state;
        self.desired = Some(desired);
        self.elapsed_steps = 0;
        GoalStart {
            observation: state,
            achieved: Goal::new(state.symbols),
            desired,
        }
    }

    pub fn knowledge_transitions(&self) -> impl Iterator<Item = Transition<N>> {
        self.goal_pool().flat_map(|goal| {
            (0..N).map(move |action| {
                let p0 = Perception::new(goal.symbols);
                let mut p1 = p0;
                p1.symbols[action] = flipped(p1.symbols[action]);
                Transition::new(p0, action, p1)
            })
        })
    }

    fn random_bits(&mut self) -> [Symbol; N] {
        core::array::from_fn(|_| Symbol::Token(if self.rng.gen_bool(0.5) { b'1' } else { b'0' }))
    }
}

impl<const N: usize> GoalEnvironment<N, N> for BitFlipping<N> {
    type Objective = ExactMatch;

    fn objective(&self) -> &Self::Objective {
        &SPARSE_GOAL_REWARD
    }

    fn reset(&mut self) -> GoalStart<N, N> {
        let desired = Goal::new(self.random_bits());
        self.reset_with_goal(desired)
    }

    fn reset_with_goal(&mut self, desired: Goal<N>) -> GoalStart<N, N> {
        assert!(valid_bits(&desired.symbols), "desired goal must be binary");
        loop {
            let state = self.random_bits();
            if state != desired.symbols {
                return self.reset_at(Perception::new(state), desired);
            }
        }
    }

    fn step(&mut self, action: usize) -> GoalStep<N, N> {
        let desired = self
            .desired
            .expect("bit flipping must be reset before a step");
        self.state.symbols[action] = flipped(self.state.symbols[action]);
        self.elapsed_steps += 1;
        let step = GoalStep {
            observation: self.state,
            achieved: Goal::new(self.state.symbols),
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

pub struct BitGoals<const N: usize> {
    next: Option<[Symbol; N]>,
}

impl<const N: usize> BitGoals<N> {
    fn new() -> Self {
        Self {
            next: Some([Symbol::Token(b'0'); N]),
        }
    }
}

impl<const N: usize> Iterator for BitGoals<N> {
    type Item = Goal<N>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next?;
        let mut successor = current;
        let mut overflow = true;
        for symbol in successor.iter_mut().rev() {
            *symbol = flipped(*symbol);
            if *symbol == Symbol::Token(b'1') {
                overflow = false;
                break;
            }
        }
        self.next = if overflow { None } else { Some(successor) };
        Some(Goal::new(current))
    }
}

fn valid_bits(symbols: &[Symbol]) -> bool {
    symbols
        .iter()
        .all(|symbol| matches!(symbol, Symbol::Token(b'0' | b'1')))
}

fn flipped(symbol: Symbol) -> Symbol {
    match symbol {
        Symbol::Token(b'0') => Symbol::Token(b'1'),
        Symbol::Token(b'1') => Symbol::Token(b'0'),
        _ => panic!("only a binary symbol can be flipped"),
    }
}
