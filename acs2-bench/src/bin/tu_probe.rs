use std::collections::{BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use acs2_core::acs2er::{replay_learning_step, ReplayMemory, ReplaySample};
use acs2_core::action_selection::{ActionSelector, BestAction, EpsilonGreedy, RandomAction};
use acs2_core::agent::Agent;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::environment::{Environment, StepOutcome};
use acs2_core::perception::Perception;
use acs2_core::population::Population;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::LearningAgent;
use acs2_envs::maze_data::{geometry_by_id, MazeGeometry};

const STATE_LEN: usize = 8;
const ACTIONS: usize = 8;
const OFFSETS: [(isize, isize); ACTIONS] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];
const WALL: u8 = 1;
const DIGIT_ZERO: u8 = b'0';
const GOAL_REWARD: f64 = 1000.0;
const EXPLORE_EPSILON: f64 = 0.8;
const EVALUATED_TRAJECTORIES: usize = 100;
const SUCCESS_PAIRS: usize = 150;
const EVALUATION_SEED: u64 = 0x5455_5052_4f42_4500;
const AGENT_STREAM: u64 = 1;
const ENVIRONMENT_STREAM: u64 = 2;
const PAIR_STREAM: u64 = 3;
const TIE_STREAM: u64 = 4;
const POOL_STREAM: u64 = 5;

type Cell = (usize, usize);

fn seeded_stream(seed: u64, stream_id: u64) -> ChaChaRandomSource {
    let mut state = ChaChaRandomSource::from_seed(seed)
        .capture_state()
        .expect("chacha exposes its state");
    state.stream = stream_id;
    ChaChaRandomSource::from_state(&state)
}

fn offset(cell: Cell, direction: usize) -> Cell {
    let (delta_row, delta_col) = OFFSETS[direction];
    (
        (cell.0 as isize + delta_row) as usize,
        (cell.1 as isize + delta_col) as usize,
    )
}

struct Grid {
    matrix: Vec<Vec<u8>>,
    cells: Vec<Cell>,
    max_steps: u32,
}

impl Grid {
    fn from_geometry(geometry: &MazeGeometry) -> Self {
        let matrix: Vec<Vec<u8>> = geometry.matrix.iter().map(|row| row.to_vec()).collect();
        let mut cells = Vec::new();
        for (row, line) in matrix.iter().enumerate() {
            for (col, &code) in line.iter().enumerate() {
                if code != WALL {
                    cells.push((row, col));
                }
            }
        }
        Self {
            matrix,
            cells,
            max_steps: geometry.max_episode_steps,
        }
    }

    fn perception(&self, cell: Cell) -> [Symbol; STATE_LEN] {
        core::array::from_fn(|index| {
            let (row, col) = offset(cell, index);
            Symbol::Token(DIGIT_ZERO + self.matrix[row][col])
        })
    }

    fn next_cell(&self, cell: Cell, action: usize) -> Cell {
        let target = offset(cell, action);
        if self.matrix[target.0][target.1] == WALL {
            cell
        } else {
            target
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GoalEncoding {
    Perception,
    Coordinates,
}

impl GoalEncoding {
    fn parse(value: &str) -> Self {
        match value {
            "perception" => GoalEncoding::Perception,
            "coordinates" => GoalEncoding::Coordinates,
            other => panic!("unknown goal encoding {other} (expected perception or coordinates)"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            GoalEncoding::Perception => "perception",
            GoalEncoding::Coordinates => "coordinates",
        }
    }
}

fn goal_symbols<const G: usize>(grid: &Grid, encoding: GoalEncoding, cell: Cell) -> [Symbol; G] {
    match encoding {
        GoalEncoding::Perception => {
            assert_eq!(G, STATE_LEN, "perception goals are eight symbols wide");
            let perception = grid.perception(cell);
            core::array::from_fn(|index| perception[index])
        }
        GoalEncoding::Coordinates => {
            assert_eq!(G, 2, "coordinate goals are two symbols wide");
            core::array::from_fn(|index| {
                let coordinate = if index == 0 { cell.0 } else { cell.1 };
                Symbol::Token(DIGIT_ZERO + coordinate as u8)
            })
        }
    }
}

struct GoalTask<const G: usize, const M: usize> {
    grid: Grid,
    states: Vec<[Symbol; STATE_LEN]>,
    goals: Vec<[Symbol; G]>,
    distances: Vec<Vec<u32>>,
    goal_pool: Vec<usize>,
}

impl<const G: usize, const M: usize> GoalTask<G, M> {
    const LAYOUT: () = assert!(STATE_LEN + G == M, "a goal task needs M = 8 + G");

    fn new(grid: Grid, encoding: GoalEncoding) -> Self {
        let () = Self::LAYOUT;
        let states: Vec<[Symbol; STATE_LEN]> =
            grid.cells.iter().map(|&cell| grid.perception(cell)).collect();
        let distinct: BTreeSet<[Symbol; STATE_LEN]> = states.iter().copied().collect();
        assert!(
            encoding != GoalEncoding::Perception || distinct.len() == states.len(),
            "perception goals need a maze without perceptual aliasing"
        );
        let goals = grid
            .cells
            .iter()
            .map(|&cell| goal_symbols::<G>(&grid, encoding, cell))
            .collect();
        let goal_pool = (0..grid.cells.len()).collect();
        let mut task = Self {
            grid,
            states,
            goals,
            distances: Vec::new(),
            goal_pool,
        };
        task.distances = (0..task.cell_count())
            .map(|goal| task.distances_to(goal))
            .collect();
        task
    }

    fn restrict_goals_to_reward_cells(&mut self) {
        self.goal_pool = (0..self.cell_count())
            .filter(|&cell| {
                let (row, col) = self.grid.cells[cell];
                self.grid.matrix[row][col] != 0
            })
            .collect();
        assert!(!self.goal_pool.is_empty(), "the maze has no reward cell");
    }

    fn restrict_goals_to_random_subset(&mut self, size: usize) {
        assert!(
            (2..=self.cell_count()).contains(&size),
            "a goal subset needs between 2 and {} goals",
            self.cell_count()
        );
        let mut chooser = seeded_stream(EVALUATION_SEED, POOL_STREAM);
        let mut chosen: Vec<usize> = Vec::with_capacity(size);
        while chosen.len() < size {
            let candidate = chooser.gen_range(self.cell_count());
            if !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        chosen.sort_unstable();
        self.goal_pool = chosen;
    }

    fn draw_goal(&self, rng: &mut dyn RandomSource) -> usize {
        self.goal_pool[rng.gen_range(self.goal_pool.len())]
    }

    fn cell_count(&self) -> usize {
        self.grid.cells.len()
    }

    fn index_of(&self, cell: Cell) -> usize {
        self.grid
            .cells
            .iter()
            .position(|&candidate| candidate == cell)
            .expect("a walkable cell")
    }

    fn next(&self, cell: usize, action: usize) -> usize {
        self.index_of(self.grid.next_cell(self.grid.cells[cell], action))
    }

    fn perception(&self, cell: usize, goal: usize) -> Perception<M> {
        Perception::new(core::array::from_fn(|index| {
            if index < STATE_LEN {
                self.states[cell][index]
            } else {
                self.goals[goal][index - STATE_LEN]
            }
        }))
    }

    fn distances_to(&self, goal: usize) -> Vec<u32> {
        let mut distances = vec![u32::MAX; self.cell_count()];
        let mut frontier = VecDeque::from([goal]);
        distances[goal] = 0;
        while let Some(cell) = frontier.pop_front() {
            for action in 0..ACTIONS {
                let neighbour = self.next(cell, action);
                if distances[neighbour] == u32::MAX {
                    distances[neighbour] = distances[cell] + 1;
                    frontier.push_back(neighbour);
                }
            }
        }
        distances
    }
}

#[derive(Clone)]
struct Trajectory {
    goal: usize,
    cells: Vec<usize>,
    actions: Vec<usize>,
}

impl Trajectory {
    fn reached_goal(&self) -> bool {
        self.cells.last() == Some(&self.goal)
    }
}

struct GoalMazeEnv<'a, const G: usize, const M: usize> {
    task: &'a GoalTask<G, M>,
    rng: ChaChaRandomSource,
    current: Trajectory,
    recent: VecDeque<Trajectory>,
    steps_taken: u64,
}

impl<'a, const G: usize, const M: usize> GoalMazeEnv<'a, G, M> {
    fn new(task: &'a GoalTask<G, M>, rng: ChaChaRandomSource) -> Self {
        Self {
            task,
            rng,
            current: Trajectory {
                goal: 0,
                cells: vec![0],
                actions: Vec::new(),
            },
            recent: VecDeque::with_capacity(EVALUATED_TRAJECTORIES + 1),
            steps_taken: 0,
        }
    }

    fn cell(&self) -> usize {
        *self.current.cells.last().expect("a trajectory starts somewhere")
    }

    fn observation(&self) -> Perception<M> {
        self.task.perception(self.cell(), self.current.goal)
    }

    fn last_trajectory(&self) -> &Trajectory {
        self.recent.back().expect("an episode has finished")
    }
}

impl<'a, const G: usize, const M: usize> Environment<M> for GoalMazeEnv<'a, G, M> {
    fn reset(&mut self) -> Perception<M> {
        let count = self.task.cell_count();
        let goal = self.task.draw_goal(&mut self.rng);
        let start = (goal + 1 + self.rng.gen_range(count - 1)) % count;
        self.current = Trajectory {
            goal,
            cells: vec![start],
            actions: Vec::new(),
        };
        self.observation()
    }

    fn step(&mut self, action: usize) -> StepOutcome<M> {
        let next = self.task.next(self.cell(), action);
        self.current.actions.push(action);
        self.current.cells.push(next);
        self.steps_taken += 1;
        let terminated = next == self.current.goal;
        let truncated =
            !terminated && self.current.actions.len() as u32 >= self.task.grid.max_steps;
        let observation = self.observation();
        if terminated || truncated {
            self.recent.push_back(self.current.clone());
            if self.recent.len() > EVALUATED_TRAJECTORIES {
                self.recent.pop_front();
            }
        }
        StepOutcome {
            observation,
            reward: if terminated { GOAL_REWARD } else { 0.0 },
            terminated,
            truncated,
            info: (),
        }
    }
}

trait Learner<const G: usize, const M: usize> {
    fn train_episode(&mut self, env: &mut GoalMazeEnv<'_, G, M>);
    fn population(&self) -> &Population<M>;
}

struct OnlineLearner<const M: usize> {
    agent: Agent<M, ChaChaRandomSource>,
    selector: EpsilonGreedy,
    time: u64,
}

impl<const M: usize> OnlineLearner<M> {
    fn new(seed: u64) -> Self {
        Self {
            agent: Agent::new(
                Configuration::default_protocol(),
                seeded_stream(seed, AGENT_STREAM),
            ),
            selector: explore_selector(),
            time: 0,
        }
    }
}

impl<const G: usize, const M: usize> Learner<G, M> for OnlineLearner<M> {
    fn train_episode(&mut self, env: &mut GoalMazeEnv<'_, G, M>) {
        let metrics =
            self.agent
                .run_explore_trial(env, &self.selector, &MaxFitnessBootstrap, self.time);
        self.time += metrics.steps as u64;
    }

    fn population(&self) -> &Population<M> {
        self.agent.population()
    }
}

struct HindsightLearner<const M: usize> {
    population: Population<M>,
    config: Configuration,
    rng: ChaChaRandomSource,
    memory: ReplayMemory<M>,
    selector: EpsilonGreedy,
    goals_per_step: usize,
    replays_per_step: usize,
    time: u64,
}

impl<const M: usize> HindsightLearner<M> {
    fn new(seed: u64, options: &Options) -> Self {
        Self {
            population: Population::new(),
            config: Configuration::default_protocol(),
            rng: seeded_stream(seed, AGENT_STREAM),
            memory: ReplayMemory::new(options.her_buffer),
            selector: explore_selector(),
            goals_per_step: options.her_goals,
            replays_per_step: options.her_replays,
            time: 0,
        }
    }

    fn learn_from<const G: usize>(&mut self, task: &GoalTask<G, M>, trajectory: &Trajectory) {
        let steps = trajectory.actions.len();
        let learning_time = self.time + steps as u64;
        for step in 0..steps {
            let truncated = step + 1 == steps && !trajectory.reached_goal();
            self.memory
                .update(relabeled_sample(task, trajectory, step, trajectory.goal, truncated));
            for goal in future_goals(trajectory, step, self.goals_per_step, &mut self.rng) {
                self.memory
                    .update(relabeled_sample(task, trajectory, step, goal, truncated));
            }
            for index in self.memory.sample_indices(self.replays_per_step, &mut self.rng) {
                let sample = self.memory.get(index);
                replay_learning_step(
                    &mut self.population,
                    &self.config,
                    &MaxFitnessBootstrap,
                    &sample,
                    learning_time,
                    &mut self.rng,
                );
            }
        }
        self.time = learning_time;
    }
}

impl<const G: usize, const M: usize> Learner<G, M> for HindsightLearner<M> {
    fn train_episode(&mut self, env: &mut GoalMazeEnv<'_, G, M>) {
        let mut observation = env.reset();
        loop {
            let match_set = self.population.form_match_set(&observation);
            let action = self
                .selector
                .select(&self.population, &match_set, &mut self.rng);
            let outcome = env.step(action);
            if outcome.terminated || outcome.truncated {
                break;
            }
            observation = outcome.observation;
        }
        let task = env.task;
        let trajectory = env.last_trajectory().clone();
        self.learn_from(task, &trajectory);
    }

    fn population(&self) -> &Population<M> {
        &self.population
    }
}

fn explore_selector() -> EpsilonGreedy {
    EpsilonGreedy {
        number_of_possible_actions: ACTIONS,
        epsilon: EXPLORE_EPSILON,
    }
}

fn relabeled_sample<const G: usize, const M: usize>(
    task: &GoalTask<G, M>,
    trajectory: &Trajectory,
    step: usize,
    goal: usize,
    truncated: bool,
) -> ReplaySample<M> {
    let from = trajectory.cells[step];
    let to = trajectory.cells[step + 1];
    let reached = to == goal;
    ReplaySample {
        state: task.perception(from, goal),
        action: trajectory.actions[step],
        reward: if reached { GOAL_REWARD } else { 0.0 },
        next_state: task.perception(to, goal),
        done: reached || truncated,
    }
}

fn future_goals(
    trajectory: &Trajectory,
    step: usize,
    count: usize,
    rng: &mut dyn RandomSource,
) -> Vec<usize> {
    let candidates = &trajectory.cells[step + 1..];
    let wanted = count.min(candidates.len());
    let mut chosen: Vec<usize> = Vec::with_capacity(wanted);
    while chosen.len() < wanted {
        let position = rng.gen_range(candidates.len());
        if !chosen.contains(&position) {
            chosen.push(position);
        }
    }
    chosen.into_iter().map(|position| candidates[position]).collect()
}

fn action_values<const G: usize, const M: usize>(
    population: &Population<M>,
    task: &GoalTask<G, M>,
    cell: usize,
) -> Vec<[f64; ACTIONS]> {
    let state = &task.states[cell];
    let candidates: Vec<(usize, &[Symbol], f64)> = population
        .classifiers()
        .iter()
        .filter_map(|classifier| {
            let action = classifier.action?;
            let symbols = &classifier.condition.symbols;
            let state_matches = (0..STATE_LEN)
                .all(|index| symbols[index].is_wildcard() || symbols[index] == state[index]);
            state_matches.then(|| (action, &symbols[STATE_LEN..], classifier.fitness()))
        })
        .collect();
    task.goal_pool
        .iter()
        .map(|&goal| {
            let target_goal = &task.goals[goal];
            let mut values = [0.0f64; ACTIONS];
            for &(action, goal_part, fitness) in &candidates {
                let goal_matches = goal_part
                    .iter()
                    .zip(target_goal.iter())
                    .all(|(condition, target)| condition.is_wildcard() || condition == target);
                if goal_matches {
                    values[action] = values[action].max(fitness);
                }
            }
            values
        })
        .collect()
}

struct TrajectoryUtility {
    logged: Vec<f64>,
    value: Vec<f64>,
    flat_logged_steps: usize,
    flat_value_steps: usize,
    steps: usize,
}

fn trajectory_utility<const G: usize, const M: usize>(
    population: &Population<M>,
    task: &GoalTask<G, M>,
    trajectory: &Trajectory,
) -> TrajectoryUtility {
    let goal_count = task.goal_pool.len();
    let mut utility = TrajectoryUtility {
        logged: vec![0.0; goal_count],
        value: vec![0.0; goal_count],
        flat_logged_steps: 0,
        flat_value_steps: 0,
        steps: trajectory.actions.len(),
    };
    for (step, &action) in trajectory.actions.iter().enumerate() {
        let values = action_values(population, task, trajectory.cells[step]);
        let logged_here: Vec<f64> = values.iter().map(|per_action| per_action[action]).collect();
        let value_here: Vec<f64> = values
            .iter()
            .map(|per_action| per_action.iter().copied().fold(0.0, f64::max))
            .collect();
        utility.flat_logged_steps += usize::from(all_equal(&logged_here));
        utility.flat_value_steps += usize::from(all_equal(&value_here));
        for goal in 0..goal_count {
            utility.logged[goal] += logged_here[goal];
            utility.value[goal] += value_here[goal];
        }
    }
    utility
}

fn all_equal(values: &[f64]) -> bool {
    values.windows(2).all(|pair| pair[0] == pair[1])
}

fn unique_argmax(values: &[f64]) -> Option<usize> {
    let best = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut winners = values.iter().enumerate().filter(|(_, &value)| value == best);
    let first = winners.next().map(|(index, _)| index);
    if winners.next().is_some() {
        None
    } else {
        first
    }
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&left, &right| values[left].total_cmp(&values[right]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start;
        while end + 1 < order.len() && values[order[end + 1]] == values[order[start]] {
            end += 1;
        }
        let rank = (start + end) as f64 / 2.0;
        for &index in &order[start..=end] {
            ranks[index] = rank;
        }
        start = end + 1;
    }
    ranks
}

fn rank_correlation(left: &[f64], right: &[f64]) -> Option<f64> {
    let left = average_ranks(left);
    let right = average_ranks(right);
    let count = left.len() as f64;
    let left_mean = left.iter().sum::<f64>() / count;
    let right_mean = right.iter().sum::<f64>() / count;
    let mut covariance = 0.0;
    let mut left_spread = 0.0;
    let mut right_spread = 0.0;
    for (a, b) in left.iter().zip(right.iter()) {
        covariance += (a - left_mean) * (b - right_mean);
        left_spread += (a - left_mean).powi(2);
        right_spread += (b - right_mean).powi(2);
    }
    if left_spread == 0.0 || right_spread == 0.0 {
        None
    } else {
        Some(covariance / (left_spread * right_spread).sqrt())
    }
}

#[derive(Default, Clone, Copy)]
struct UtilityReport {
    trajectories: usize,
    flat_logged_steps: f64,
    flat_value_steps: f64,
    unique_logged: f64,
    unique_value: f64,
    hit_logged: f64,
    hit_value: f64,
    hit_baseline: f64,
    rank_correlation: f64,
    constant_value: f64,
    spread_value: f64,
    nearest_value: f64,
    nearest_baseline: f64,
}

fn evaluate_utility<'a, const G: usize, const M: usize>(
    population: &Population<M>,
    task: &GoalTask<G, M>,
    trajectories: impl IntoIterator<Item = &'a Trajectory>,
) -> UtilityReport {
    let pool = &task.goal_pool;
    let goal_count = pool.len();
    let mut nearest_value = 0usize;
    let mut nearest_baseline = 0.0;
    let mut counted = 0usize;
    let mut steps = 0usize;
    let mut flat_logged = 0usize;
    let mut flat_value = 0usize;
    let mut unique_logged = 0usize;
    let mut unique_value = 0usize;
    let mut hit_logged = 0usize;
    let mut hit_value = 0usize;
    let mut baseline = 0.0;
    let mut constant = 0usize;
    let mut correlation_sum = 0.0;
    let mut correlation_count = 0usize;
    let mut spread_sum = 0.0;

    for trajectory in trajectories {
        counted += 1;
        let utility = trajectory_utility(population, task, trajectory);
        let highest = utility.value.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lowest = utility.value.iter().copied().fold(f64::INFINITY, f64::min);
        if highest > 0.0 {
            spread_sum += (highest - lowest) / highest;
        }
        steps += utility.steps;
        flat_logged += utility.flat_logged_steps;
        flat_value += utility.flat_value_steps;
        let visited: BTreeSet<usize> = trajectory.cells[1..].iter().copied().collect();
        baseline +=
            pool.iter().filter(|goal| visited.contains(goal)).count() as f64 / goal_count as f64;
        let proximity: Vec<f64> = pool
            .iter()
            .map(|&goal| {
                let nearest = trajectory
                    .cells
                    .iter()
                    .map(|&cell| task.distances[goal][cell])
                    .min()
                    .expect("a trajectory visits at least one cell");
                -(nearest as f64)
            })
            .collect();
        let closest = proximity.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let nearest_positions: Vec<usize> = (0..goal_count)
            .filter(|&position| proximity[position] == closest)
            .collect();
        nearest_baseline += nearest_positions.len() as f64 / goal_count as f64;
        if let Some(best) = unique_argmax(&utility.logged) {
            unique_logged += 1;
            hit_logged += usize::from(visited.contains(&pool[best]));
        }
        if let Some(best) = unique_argmax(&utility.value) {
            unique_value += 1;
            hit_value += usize::from(visited.contains(&pool[best]));
            nearest_value += usize::from(nearest_positions.contains(&best));
        }
        match rank_correlation(&utility.value, &proximity) {
            Some(correlation) => {
                correlation_sum += correlation;
                correlation_count += 1;
            }
            None => constant += usize::from(all_equal(&utility.value)),
        }
    }

    let share = |numerator: usize, denominator: usize| {
        if denominator == 0 {
            f64::NAN
        } else {
            numerator as f64 / denominator as f64
        }
    };
    UtilityReport {
        trajectories: counted,
        flat_logged_steps: share(flat_logged, steps),
        flat_value_steps: share(flat_value, steps),
        unique_logged: share(unique_logged, counted),
        unique_value: share(unique_value, counted),
        hit_logged: share(hit_logged, unique_logged),
        hit_value: share(hit_value, unique_value),
        hit_baseline: if counted == 0 { f64::NAN } else { baseline / counted as f64 },
        rank_correlation: if correlation_count == 0 {
            f64::NAN
        } else {
            correlation_sum / correlation_count as f64
        },
        constant_value: share(constant, counted),
        spread_value: if counted == 0 { f64::NAN } else { spread_sum / counted as f64 },
        nearest_value: share(nearest_value, unique_value),
        nearest_baseline: if counted == 0 {
            f64::NAN
        } else {
            nearest_baseline / counted as f64
        },
    }
}

fn specified_goal_positions<const M: usize>(classifier: &Classifier<M>) -> usize {
    classifier.condition.symbols[STATE_LEN..]
        .iter()
        .filter(|symbol| !symbol.is_wildcard())
        .count()
}

fn policy_success<const G: usize, const M: usize, S: ActionSelector<M>>(
    population: &Population<M>,
    task: &GoalTask<G, M>,
    selector: &S,
) -> f64 {
    let mut pairs = seeded_stream(EVALUATION_SEED, PAIR_STREAM);
    let mut ties = seeded_stream(EVALUATION_SEED, TIE_STREAM);
    let count = task.cell_count();
    let mut successes = 0usize;
    for _ in 0..SUCCESS_PAIRS {
        let goal = task.draw_goal(&mut pairs);
        let mut cell = (goal + 1 + pairs.gen_range(count - 1)) % count;
        for _ in 0..task.grid.max_steps {
            let perception = task.perception(cell, goal);
            let match_set = population.form_match_set(&perception);
            let action = selector.select(population, &match_set, &mut ties);
            cell = task.next(cell, action);
            if cell == goal {
                successes += 1;
                break;
            }
        }
    }
    successes as f64 / SUCCESS_PAIRS as f64
}

struct Snapshot {
    episodes: usize,
    env_steps: u64,
    population: usize,
    reliable: usize,
    goal_specific: f64,
    goal_specific_reliable: f64,
    goal_positions_mean: f64,
    fully_goal_specific: f64,
    success: f64,
    utility: UtilityReport,
    wall_seconds: f64,
    capped: bool,
}

fn snapshot<const G: usize, const M: usize, L: Learner<G, M>>(
    learner: &L,
    env: &GoalMazeEnv<'_, G, M>,
    task: &GoalTask<G, M>,
    episodes: usize,
    started: Instant,
    capped: bool,
) -> Snapshot {
    let population = learner.population();
    let theta_r = Configuration::default_protocol().theta_r;
    let reliable: Vec<&Classifier<M>> = population
        .classifiers()
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .collect();
    let goal_positions: Vec<usize> = population
        .classifiers()
        .iter()
        .map(specified_goal_positions)
        .collect();
    let specific_all = goal_positions.iter().filter(|&&count| count > 0).count();
    let specified_total: usize = goal_positions.iter().sum();
    let fully_specific = goal_positions.iter().filter(|&&count| count == G).count();
    let specific_reliable = reliable
        .iter()
        .filter(|classifier| specified_goal_positions(classifier) > 0)
        .count();
    let fraction = |numerator: usize, denominator: usize| {
        if denominator == 0 {
            0.0
        } else {
            numerator as f64 / denominator as f64
        }
    };
    let greedy = BestAction {
        number_of_possible_actions: ACTIONS,
    };
    Snapshot {
        episodes,
        env_steps: env.steps_taken,
        population: population.len(),
        reliable: reliable.len(),
        goal_specific: fraction(specific_all, population.len()),
        goal_specific_reliable: fraction(specific_reliable, reliable.len()),
        goal_positions_mean: fraction(specified_total, specific_all),
        fully_goal_specific: fraction(fully_specific, population.len()),
        success: policy_success(population, task, &greedy),
        utility: evaluate_utility(population, task, env.recent.iter()),
        wall_seconds: started.elapsed().as_secs_f64(),
        capped,
    }
}

fn run<const G: usize, const M: usize, L: Learner<G, M>>(
    mut learner: L,
    task: &GoalTask<G, M>,
    seed: u64,
    options: &Options,
) -> Vec<Snapshot> {
    let mut env = GoalMazeEnv::new(task, seeded_stream(seed, ENVIRONMENT_STREAM));
    let started = Instant::now();
    let cap = Duration::from_secs(options.time_cap_secs);
    let mut episodes = 0usize;
    let mut snapshots = Vec::new();
    for &checkpoint in &options.checkpoints {
        let mut capped = false;
        while episodes < checkpoint {
            learner.train_episode(&mut env);
            episodes += 1;
            if started.elapsed() > cap {
                capped = true;
                break;
            }
        }
        snapshots.push(snapshot(&learner, &env, task, episodes, started, capped));
        if capped {
            break;
        }
    }
    snapshots
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Regime {
    Online,
    Hindsight,
}

impl Regime {
    fn parse(value: &str) -> Self {
        match value {
            "acs2" => Regime::Online,
            "her" => Regime::Hindsight,
            other => panic!("unknown regime {other} (expected acs2 or her)"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Regime::Online => "acs2",
            Regime::Hindsight => "her",
        }
    }
}

struct Options {
    maze: String,
    regimes: Vec<Regime>,
    encodings: Vec<GoalEncoding>,
    seeds: Vec<u64>,
    checkpoints: Vec<usize>,
    her_goals: usize,
    her_replays: usize,
    her_buffer: usize,
    time_cap_secs: u64,
    fixed_goal: bool,
    goal_pool_size: Option<usize>,
}

fn parse_list<T>(value: Option<String>, flag: &str, item: impl Fn(&str) -> T) -> Vec<T> {
    value
        .unwrap_or_else(|| panic!("{flag} needs a value"))
        .split(',')
        .map(|part| item(part.trim()))
        .collect()
}

fn parse_number<T: std::str::FromStr>(value: Option<String>, flag: &str) -> T {
    value
        .unwrap_or_else(|| panic!("{flag} needs a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must be a number"))
}

impl Options {
    fn parse() -> Self {
        let mut options = Options {
            maze: "Maze4-v0".to_string(),
            regimes: vec![Regime::Online, Regime::Hindsight],
            encodings: vec![GoalEncoding::Perception, GoalEncoding::Coordinates],
            seeds: vec![42, 43, 44],
            checkpoints: vec![125, 250, 500, 1000, 2000],
            her_goals: 2,
            her_replays: 4,
            her_buffer: 10_000,
            time_cap_secs: 90,
            fixed_goal: false,
            goal_pool_size: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--maze" => options.maze = args.next().expect("--maze needs a value"),
                "--regimes" => options.regimes = parse_list(args.next(), &flag, Regime::parse),
                "--encodings" => {
                    options.encodings = parse_list(args.next(), &flag, GoalEncoding::parse)
                }
                "--seeds" => {
                    options.seeds = parse_list(args.next(), &flag, |part| {
                        part.parse().expect("a seed is an integer")
                    })
                }
                "--checkpoints" => {
                    options.checkpoints = parse_list(args.next(), &flag, |part| {
                        part.parse().expect("a checkpoint is an episode count")
                    })
                }
                "--her-goals" => options.her_goals = parse_number(args.next(), &flag),
                "--her-replays" => options.her_replays = parse_number(args.next(), &flag),
                "--her-buffer" => options.her_buffer = parse_number(args.next(), &flag),
                "--time-cap-secs" => options.time_cap_secs = parse_number(args.next(), &flag),
                "--fixed-goal" => options.fixed_goal = true,
                "--goal-pool-size" => {
                    options.goal_pool_size = Some(parse_number(args.next(), &flag))
                }
                other => panic!("unknown flag {other}"),
            }
        }
        assert!(
            !(options.fixed_goal && options.goal_pool_size.is_some()),
            "--fixed-goal and --goal-pool-size exclude each other"
        );
        assert!(
            options.checkpoints.windows(2).all(|pair| pair[0] < pair[1]),
            "--checkpoints must increase"
        );
        options
    }
}

fn format_snapshot(maze: &str, regime: Regime, encoding: GoalEncoding, seed: u64, snapshot: &Snapshot) -> String {
    let utility = &snapshot.utility;
    format!(
        "tu-probe maze={maze} regime={} goal={} seed={seed} episodes={} env_steps={} pop={} reliable={} goal_spec={:.3} goal_spec_rel={:.3} goal_pos={:.2} goal_full={:.3} success={:.3} flat_q={:.3} flat_v={:.3} unique_q={:.3} unique_v={:.3} hit_q={:.3} hit_v={:.3} hit_base={:.3} rank_corr={:.3} const_v={:.3} spread_v={:.3} nearest_v={:.3} nearest_base={:.3} trajectories={} wall={:.1}s{}",
        regime.label(),
        encoding.label(),
        snapshot.episodes,
        snapshot.env_steps,
        snapshot.population,
        snapshot.reliable,
        snapshot.goal_specific,
        snapshot.goal_specific_reliable,
        snapshot.goal_positions_mean,
        snapshot.fully_goal_specific,
        snapshot.success,
        utility.flat_logged_steps,
        utility.flat_value_steps,
        utility.unique_logged,
        utility.unique_value,
        utility.hit_logged,
        utility.hit_value,
        utility.hit_baseline,
        utility.rank_correlation,
        utility.constant_value,
        utility.spread_value,
        utility.nearest_value,
        utility.nearest_baseline,
        utility.trajectories,
        snapshot.wall_seconds,
        if snapshot.capped { " capped=true" } else { "" },
    )
}

fn run_encoding<const G: usize, const M: usize>(options: &Options, encoding: GoalEncoding) {
    let geometry = geometry_by_id(&options.maze)
        .unwrap_or_else(|| panic!("unknown maze {}", options.maze));
    let mut task = GoalTask::<G, M>::new(Grid::from_geometry(geometry), encoding);
    if options.fixed_goal {
        task.restrict_goals_to_reward_cells();
    }
    if let Some(size) = options.goal_pool_size {
        task.restrict_goals_to_random_subset(size);
    }
    let random = RandomAction {
        number_of_possible_actions: ACTIONS,
    };
    println!(
        "tu-probe maze={} goal={} goals={} goal_pool={} goal_cells={:?} random_policy_success={:.3}",
        options.maze,
        encoding.label(),
        task.cell_count(),
        task.goal_pool.len(),
        task.goal_pool
            .iter()
            .map(|&goal| task.grid.cells[goal])
            .collect::<Vec<_>>(),
        policy_success(&Population::<M>::new(), &task, &random),
    );
    for &regime in &options.regimes {
        for &seed in &options.seeds {
            let snapshots = match regime {
                Regime::Online => run(OnlineLearner::<M>::new(seed), &task, seed, options),
                Regime::Hindsight => {
                    run(HindsightLearner::<M>::new(seed, options), &task, seed, options)
                }
            };
            for snapshot in &snapshots {
                println!("{}", format_snapshot(&options.maze, regime, encoding, seed, snapshot));
            }
        }
    }
}

fn main() {
    let options = Options::parse();
    println!(
        "tu-probe: maze={} regimes={:?} encodings={:?} seeds={:?} checkpoints={:?} her_goals={} her_replays={} her_buffer={} epsilon={} evaluated_trajectories={} success_pairs={} time_cap={}s",
        options.maze,
        options.regimes.iter().map(|regime| regime.label()).collect::<Vec<_>>(),
        options.encodings.iter().map(|encoding| encoding.label()).collect::<Vec<_>>(),
        options.seeds,
        options.checkpoints,
        options.her_goals,
        options.her_replays,
        options.her_buffer,
        EXPLORE_EPSILON,
        EVALUATED_TRAJECTORIES,
        SUCCESS_PAIRS,
        options.time_cap_secs,
    );
    for &encoding in &options.encodings {
        match encoding {
            GoalEncoding::Perception => run_encoding::<8, 16>(&options, encoding),
            GoalEncoding::Coordinates => run_encoding::<2, 10>(&options, encoding),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maze4_task() -> GoalTask<8, 16> {
        let geometry = geometry_by_id("Maze4-v0").expect("Maze4 is registered");
        GoalTask::new(Grid::from_geometry(geometry), GoalEncoding::Perception)
    }

    fn walk(task: &GoalTask<8, 16>, goal: usize, actions: &[usize]) -> Trajectory {
        let mut cells = vec![0];
        for &action in actions {
            let next = task.next(*cells.last().unwrap(), action);
            cells.push(next);
        }
        Trajectory {
            goal,
            cells,
            actions: actions.to_vec(),
        }
    }

    fn goal_agnostic(action: usize, reward: f64) -> Classifier<16> {
        let mut classifier =
            Classifier::general(Some(action), &Configuration::default_protocol());
        classifier.q = 1.0;
        classifier.r = reward;
        classifier
    }

    fn agnostic_population() -> Vec<Classifier<16>> {
        (0..ACTIONS)
            .map(|action| goal_agnostic(action, 100.0 * (action + 1) as f64))
            .collect()
    }

    #[test]
    fn a_goal_agnostic_population_gives_every_goal_the_same_utility() {
        let task = maze4_task();
        let trajectory = walk(&task, 7, &[2, 4, 4, 2, 0, 6]);
        let population = Population::from_classifiers(agnostic_population());

        let report = evaluate_utility(&population, &task, [&trajectory]);

        assert_eq!(report.flat_logged_steps, 1.0);
        assert_eq!(report.flat_value_steps, 1.0);
        assert_eq!(report.unique_value, 0.0);
        assert_eq!(report.constant_value, 1.0);
    }

    #[test]
    fn a_goal_specific_classifier_makes_its_goal_the_unique_argmax() {
        let task = maze4_task();
        let trajectory = walk(&task, 7, &[2, 4, 4, 2, 0, 6]);
        let favoured = 11;
        let mut specific = goal_agnostic(trajectory.actions[0], 900.0);
        for index in 0..STATE_LEN {
            specific.condition.symbols[STATE_LEN + index] = task.goals[favoured][index];
        }
        let mut classifiers = agnostic_population();
        classifiers.push(specific);
        let population = Population::from_classifiers(classifiers);

        let utility = trajectory_utility(&population, &task, &trajectory);

        assert_eq!(unique_argmax(&utility.value), Some(favoured));
        assert_eq!(unique_argmax(&utility.logged), Some(favoured));
        assert_eq!(utility.flat_value_steps, 0);
        assert!(utility.flat_logged_steps < utility.steps);
    }

    #[test]
    fn a_relabeled_transition_terminates_exactly_on_its_goal() {
        let task = maze4_task();
        let trajectory = walk(&task, 7, &[2, 4, 4, 2]);
        let reached = trajectory.cells[1];
        let elsewhere = (reached + 1) % task.cell_count();

        let hit = relabeled_sample(&task, &trajectory, 0, reached, false);
        let miss = relabeled_sample(&task, &trajectory, 0, elsewhere, false);

        assert_eq!(hit.reward, GOAL_REWARD);
        assert!(hit.done);
        assert_eq!(miss.reward, 0.0);
        assert!(!miss.done);
        assert_eq!(
            hit.state.symbols[STATE_LEN..],
            hit.next_state.symbols[STATE_LEN..],
            "a relabeled transition keeps one goal on both sides"
        );
    }
}
