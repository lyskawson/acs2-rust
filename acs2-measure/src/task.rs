use std::collections::BTreeSet;
use std::marker::PhantomData;

use acs2_core::goal::{Goal, GoalEnvironment, GoalStart};
use acs2_core::perception::Perception;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::{position_goal, HandEye, HandEyeState};
use acs2_envs::goal::maze::{GoalMaze, MazeGoalEncoding};
use acs2_envs::goal::taxi::{passenger_goal, Taxi, TaxiState};
use acs2_envs::maze::geometries::MazeGeometry;
use acs2_envs::maze::topology::Cell;

pub type Pair<T, const G: usize> = (T, Goal<G>, f64);

pub trait Task<const S: usize, const G: usize, const M: usize> {
    type Env: GoalEnvironment<S, G>;
    type State: Copy + PartialEq + std::fmt::Debug;

    fn name(&self) -> &str;
    fn actions(&self) -> usize;
    fn cap(&self) -> u32;
    fn encoding(&self) -> &'static str;
    fn pool_label(&self) -> String;
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env;
    fn states(&self) -> Vec<Self::State>;
    fn next(&self, state: Self::State, action: usize) -> Self::State;
    fn achieved(&self, state: Self::State) -> Goal<G>;
    fn distance(&self, state: Self::State, goal: &Goal<G>) -> Option<u32>;
    fn pairs(&self) -> Vec<Pair<Self::State, G>>;
    fn reset_at(&self, env: &mut Self::Env, state: Self::State, goal: Goal<G>) -> GoalStart<S, G>;
    fn training_goal(&self, _rng: &mut ChaChaRandomSource) -> Option<Goal<G>> {
        None
    }
    fn sampled_evaluation(&self) -> bool {
        false
    }
    fn analytical_random_success(&self, _pairs: &[Pair<Self::State, G>]) -> Option<f64> {
        None
    }
    fn analytical_reachability(&self) -> Option<(f64, f64, f64)> {
        None
    }
}

pub struct MazeTask<E, const G: usize> {
    pub name: String,
    pub geometry: &'static MazeGeometry,
    pub pool: Vec<Cell>,
    pub cap: u32,
    pub template: GoalMaze<E, G>,
    pub encoding: &'static str,
    marker: PhantomData<E>,
}

impl<E: MazeGoalEncoding<G>, const G: usize> MazeTask<E, G> {
    pub fn new(
        name: String,
        geometry: &'static MazeGeometry,
        pool: Vec<Cell>,
        cap: u32,
        encoding: &'static str,
    ) -> Self {
        let template = GoalMaze::multi_goal(
            geometry,
            pool.clone(),
            cap,
            Box::new(ChaChaRandomSource::from_seed(0)),
        )
        .expect("valid maze pool");
        Self {
            name,
            geometry,
            pool,
            cap,
            template,
            encoding,
            marker: PhantomData,
        }
    }
}

impl<E: MazeGoalEncoding<G>, const G: usize, const M: usize> Task<8, G, M> for MazeTask<E, G> {
    type Env = GoalMaze<E, G>;
    type State = Cell;

    fn name(&self) -> &str {
        &self.name
    }
    fn actions(&self) -> usize {
        8
    }
    fn cap(&self) -> u32 {
        self.cap
    }
    fn encoding(&self) -> &'static str {
        self.encoding
    }
    fn pool_label(&self) -> String {
        format!("{:?}", self.pool)
    }
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env {
        GoalMaze::multi_goal(self.geometry, self.pool.clone(), self.cap, Box::new(rng))
            .expect("valid maze pool")
    }
    fn states(&self) -> Vec<Cell> {
        self.template.topology().walkable_cells().to_vec()
    }
    fn next(&self, state: Cell, action: usize) -> Cell {
        self.template.topology().next_cell(state, action)
    }
    fn achieved(&self, state: Cell) -> Goal<G> {
        self.template.goal_at(state)
    }
    fn distance(&self, state: Cell, goal: &Goal<G>) -> Option<u32> {
        self.template.distance_to_goal(state, goal)
    }
    fn pairs(&self) -> Vec<Pair<Cell, G>> {
        let cells = self.template.topology().walkable_cells().to_vec();
        let weight = 1.0 / (self.pool.len() * (cells.len() - 1)) as f64;
        self.pool
            .iter()
            .flat_map(|&goal| {
                cells
                    .iter()
                    .filter(move |&&start| start != goal)
                    .map(move |&start| (start, self.template.goal_at(goal), weight))
            })
            .collect()
    }
    fn reset_at(&self, env: &mut Self::Env, state: Cell, goal: Goal<G>) -> GoalStart<8, G> {
        env.reset_at(state, goal)
    }
}

pub struct HandEyeTask<const SIDE: usize, const S: usize> {
    pub name: String,
    pub cap: u32,
    pub pool: Vec<Goal<2>>,
    pub template: HandEye<SIDE, S>,
}

impl<const SIDE: usize, const S: usize> HandEyeTask<SIDE, S> {
    pub fn new(name: String, cap: u32, pool: Vec<Goal<2>>) -> Self {
        let template = HandEye::new(cap, Box::new(ChaChaRandomSource::from_seed(0)));
        assert!(
            !pool.is_empty() && pool.iter().all(|goal| template.goal_pool().contains(goal)),
            "pool must list real hand-eye goals"
        );
        assert_eq!(
            pool.iter().copied().collect::<BTreeSet<_>>().len(),
            pool.len(),
            "pool goals must be unique"
        );
        Self {
            name,
            cap,
            pool,
            template,
        }
    }
}

impl<const SIDE: usize, const S: usize, const M: usize> Task<S, 2, M> for HandEyeTask<SIDE, S> {
    type Env = HandEye<SIDE, S>;
    type State = HandEyeState;

    fn name(&self) -> &str {
        &self.name
    }
    fn actions(&self) -> usize {
        6
    }
    fn cap(&self) -> u32 {
        self.cap
    }
    fn encoding(&self) -> &'static str {
        "coordinates"
    }
    fn pool_label(&self) -> String {
        format!("{:?}", self.pool)
    }
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env {
        HandEye::new(self.cap, Box::new(rng))
    }
    fn states(&self) -> Vec<Self::State> {
        self.template.states().collect()
    }
    fn next(&self, state: Self::State, action: usize) -> Self::State {
        state.after_action::<SIDE>(action)
    }
    fn achieved(&self, state: Self::State) -> Goal<2> {
        position_goal(state.block)
    }
    fn distance(&self, state: Self::State, goal: &Goal<2>) -> Option<u32> {
        self.template.distance_to_goal(state, goal)
    }
    fn pairs(&self) -> Vec<Pair<Self::State, 2>> {
        let cells = SIDE * SIDE;
        let mut pairs = Vec::new();
        for block_index in 0..cells {
            let block = (block_index % SIDE, block_index / SIDE);
            for held in [true, false] {
                let grippers: Vec<_> = if held {
                    vec![block]
                } else {
                    (0..cells)
                        .map(|index| (index % SIDE, index / SIDE))
                        .collect()
                };
                let state_weight = 0.5 / cells as f64 / grippers.len() as f64;
                for gripper in grippers {
                    let state = HandEyeState {
                        gripper,
                        block,
                        held,
                    };
                    for &goal in &self.pool {
                        if goal != position_goal(block) {
                            pairs.push((
                                state,
                                goal,
                                state_weight / self.pool.len() as f64 * cells as f64
                                    / (cells - 1) as f64,
                            ));
                        }
                    }
                }
            }
        }
        pairs
    }
    fn reset_at(&self, env: &mut Self::Env, state: Self::State, goal: Goal<2>) -> GoalStart<S, 2> {
        env.reset_at(state, goal)
    }
    fn training_goal(&self, rng: &mut ChaChaRandomSource) -> Option<Goal<2>> {
        if self.pool.len() == SIDE * SIDE {
            None
        } else {
            Some(self.pool[rng.gen_range(self.pool.len())])
        }
    }
}

pub struct TaxiTask {
    pub cap: u32,
    pub pool: Vec<Goal<1>>,
    pub template: Taxi,
}

impl TaxiTask {
    pub fn new(cap: u32, pool: Vec<Goal<1>>) -> Self {
        let template = Taxi::new(cap, Box::new(ChaChaRandomSource::from_seed(0)));
        assert!(
            !pool.is_empty() && pool.iter().all(|goal| template.goal_pool().contains(goal)),
            "pool must list real taxi goals"
        );
        assert_eq!(
            pool.iter().copied().collect::<BTreeSet<_>>().len(),
            pool.len(),
            "pool goals must be unique"
        );
        Self {
            cap,
            pool,
            template,
        }
    }
}

impl Task<3, 1, 4> for TaxiTask {
    type Env = Taxi;
    type State = TaxiState;
    fn name(&self) -> &str {
        "taxi"
    }
    fn actions(&self) -> usize {
        6
    }
    fn cap(&self) -> u32 {
        self.cap
    }
    fn encoding(&self) -> &'static str {
        "stand"
    }
    fn pool_label(&self) -> String {
        format!("{:?}", self.pool)
    }
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env {
        Taxi::new(self.cap, Box::new(rng))
    }
    fn states(&self) -> Vec<Self::State> {
        self.template.states().collect()
    }
    fn next(&self, state: Self::State, action: usize) -> Self::State {
        state.after_action(action)
    }
    fn achieved(&self, state: Self::State) -> Goal<1> {
        state.achieved()
    }
    fn distance(&self, state: Self::State, goal: &Goal<1>) -> Option<u32> {
        self.template.distance_to_goal(state, goal)
    }
    fn pairs(&self) -> Vec<Pair<Self::State, 1>> {
        let weight = 1.0 / (self.pool.len() * 3 * 25) as f64;
        self.pool
            .iter()
            .flat_map(|&goal| {
                self.template
                    .states()
                    .filter(move |state| state.passenger < 4 && state.achieved() != goal)
                    .map(move |state| (state, goal, weight))
            })
            .collect()
    }
    fn reset_at(&self, env: &mut Self::Env, state: Self::State, goal: Goal<1>) -> GoalStart<3, 1> {
        env.reset_at(state, goal)
    }
    fn training_goal(&self, rng: &mut ChaChaRandomSource) -> Option<Goal<1>> {
        if self.pool.len() == 4 {
            None
        } else {
            Some(self.pool[rng.gen_range(self.pool.len())])
        }
    }
}

pub struct BitTask<const N: usize> {
    pub cap: u32,
    pub pool: Vec<Goal<N>>,
    pub template: BitFlipping<N>,
}

impl<const N: usize> BitTask<N> {
    pub fn new(cap: u32, pool: Vec<Goal<N>>) -> Self {
        assert!(
            !pool.is_empty()
                && pool.iter().all(|goal| goal
                    .symbols
                    .iter()
                    .all(|symbol| matches!(symbol, Symbol::Token(b'0' | b'1')))),
            "pool must list binary goals"
        );
        assert_eq!(
            pool.iter().copied().collect::<BTreeSet<_>>().len(),
            pool.len(),
            "pool goals must be unique"
        );
        Self {
            cap,
            pool,
            template: BitFlipping::with_step_cap(cap, Box::new(ChaChaRandomSource::from_seed(0))),
        }
    }
}

impl<const N: usize, const M: usize> Task<N, N, M> for BitTask<N> {
    type Env = BitFlipping<N>;
    type State = Perception<N>;
    fn name(&self) -> &str {
        "bitflipping"
    }
    fn actions(&self) -> usize {
        N
    }
    fn cap(&self) -> u32 {
        self.cap
    }
    fn encoding(&self) -> &'static str {
        "bits"
    }
    fn pool_label(&self) -> String {
        format!("{:?}", self.pool)
    }
    fn environment(&self, rng: ChaChaRandomSource) -> Self::Env {
        BitFlipping::with_step_cap(self.cap, Box::new(rng))
    }
    fn states(&self) -> Vec<Self::State> {
        self.template
            .goal_pool()
            .map(|goal| Perception::new(goal.symbols))
            .collect()
    }
    fn next(&self, mut state: Self::State, action: usize) -> Self::State {
        state.symbols[action] = match state.symbols[action] {
            Symbol::Token(b'0') => Symbol::Token(b'1'),
            _ => Symbol::Token(b'0'),
        };
        state
    }
    fn achieved(&self, state: Self::State) -> Goal<N> {
        Goal::new(state.symbols)
    }
    fn distance(&self, state: Self::State, goal: &Goal<N>) -> Option<u32> {
        self.template.distance_to_goal(&state, goal)
    }
    fn pairs(&self) -> Vec<Pair<Self::State, N>> {
        let total = self.pool.len() * ((1usize << N) - 1);
        if total > 20_000 {
            let mut rng = ChaChaRandomSource::from_seed_and_stream(0, 5);
            return (0..8_192)
                .map(|_| {
                    let goal = self.pool[rng.gen_range(self.pool.len())];
                    let excluded = goal.symbols.iter().fold(0usize, |value, symbol| {
                        (value << 1) | usize::from(*symbol == Symbol::Token(b'1'))
                    });
                    let picked = rng.gen_range((1usize << N) - 1);
                    let index = picked + usize::from(picked >= excluded);
                    let state = Perception::new(core::array::from_fn(|bit| {
                        Symbol::Token(if (index >> (N - bit - 1)) & 1 == 1 {
                            b'1'
                        } else {
                            b'0'
                        })
                    }));
                    (state, goal, 1.0 / 8_192.0)
                })
                .collect();
        }
        let states: Vec<_> = self
            .template
            .goal_pool()
            .map(|goal| Perception::new(goal.symbols))
            .collect();
        let weight = 1.0 / (self.pool.len() * (states.len() - 1)) as f64;
        self.pool
            .iter()
            .flat_map(|&goal| {
                states
                    .iter()
                    .filter(move |&&state| state.symbols != goal.symbols)
                    .map(move |&state| (state, goal, weight))
            })
            .collect()
    }
    fn reset_at(&self, env: &mut Self::Env, state: Self::State, goal: Goal<N>) -> GoalStart<N, N> {
        env.reset_at(state, goal)
    }
    fn training_goal(&self, rng: &mut ChaChaRandomSource) -> Option<Goal<N>> {
        if self.pool.len() == (1usize << N) {
            None
        } else {
            Some(self.pool[rng.gen_range(self.pool.len())])
        }
    }
    fn sampled_evaluation(&self) -> bool {
        self.pool.len() * ((1usize << N) - 1) > 20_000
    }
    fn analytical_random_success(&self, _pairs: &[Pair<Self::State, N>]) -> Option<f64> {
        let mut current = vec![0.0; N + 1];
        current[0] = 1.0;
        for _ in 0..self.cap {
            let mut next = vec![0.0; N + 1];
            next[0] = 1.0;
            for (distance, probability) in next.iter_mut().enumerate().skip(1) {
                *probability = (distance as f64 * current[distance - 1]
                    + if distance < N {
                        (N - distance) as f64 * current[distance + 1]
                    } else {
                        0.0
                    })
                    / N as f64;
            }
            current = next;
        }
        let mut combinations = 1u64;
        let mut total = 0.0;
        for (distance, &probability) in current.iter().enumerate().skip(1) {
            combinations = combinations * (N - distance + 1) as u64 / distance as u64;
            total += combinations as f64 * probability;
        }
        Some(total / ((1usize << N) - 1) as f64)
    }
    fn analytical_reachability(&self) -> Option<(f64, f64, f64)> {
        let mut combinations = 1u64;
        let mut within = 0u64;
        let mut after = 0u64;
        for distance in 1..=N {
            combinations = combinations * (N - distance + 1) as u64 / distance as u64;
            if distance as u32 <= self.cap {
                within += combinations;
            } else {
                after += combinations;
            }
        }
        let denominator = ((1usize << N) - 1) as f64;
        Some((within as f64 / denominator, after as f64 / denominator, 0.0))
    }
}

pub fn parse_cells(input: &str) -> Vec<Cell> {
    input
        .split(',')
        .map(|part| {
            let (row, col) = part
                .split_once(':')
                .expect("maze and hand-eye goals use row:col");
            (row.parse().expect("row"), col.parse().expect("column"))
        })
        .collect()
}

pub fn bit_goal<const N: usize>(bits: &str) -> Goal<N> {
    assert_eq!(bits.len(), N, "goal bit count");
    Goal::new(core::array::from_fn(|index| match bits.as_bytes()[index] {
        b'0' | b'1' => Symbol::Token(bits.as_bytes()[index]),
        _ => panic!("binary goal"),
    }))
}

pub fn taxi_goal(index: usize) -> Goal<1> {
    passenger_goal(index)
}
