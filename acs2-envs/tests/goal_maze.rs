use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::acs2er::{Acs2ErAgent, ReplayConfiguration};
use acs2_core::alp::cover;
use acs2_core::agent::Agent;
use acs2_core::checkpoint::Checkpointed;
use acs2_core::classifier::Classifier;
use acs2_core::config::Configuration;
use acs2_core::environment::Environment;
use acs2_core::goal::{Goal, GoalConditioned, GoalEnvironment, GoalLayout};
use acs2_core::perception::Perception;
use acs2_core::population::Population;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::symbol::Symbol;
use acs2_core::trial::{LearningAgent, TruncationMode};
use acs2_envs::goal::maze::{CoordinateGoalMaze, GoalMazeError, PerceptionGoalMaze};
use acs2_envs::goal::SPARSE_GOAL_REWARD;
use acs2_envs::maze::geometries::{alcs, geometry_by_id, pyalcs, MazeGeometry};
use acs2_envs::maze::knowledge::{
    goal_transitions, multi_goal_transitions, pyalcs_transitions, MazeKnowledgeSet,
};
use acs2_envs::maze::topology::{Cell, MazeTopology};
use acs2_envs::maze::Maze;
use acs2_envs::roles::{PERFORMANCE_MAZES, RESEARCH_MAZES, VALIDATION_MAZES};
use serde_json::Value;

fn rng(seed: u64) -> Box<dyn RandomSource> {
    Box::new(ChaChaRandomSource::from_seed_and_stream(seed, 1))
}

fn all_geometries() -> impl Iterator<Item = &'static MazeGeometry> {
    pyalcs::GEOMETRIES.iter().chain(alcs::GEOMETRIES)
}

fn fixture(name: &str) -> Value {
    let path = format!("{}/../fixtures/{name}.json", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn cell(value: &Value) -> Cell {
    (
        value[0].as_u64().unwrap() as usize,
        value[1].as_u64().unwrap() as usize,
    )
}

fn perception(value: &Value) -> Perception<8> {
    let bytes: Vec<u8> = if let Some(string) = value.as_str() {
        string.as_bytes().to_vec()
    } else {
        value
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_str().unwrap().as_bytes()[0])
            .collect()
    };
    assert_eq!(bytes.len(), 8);
    Perception::new(core::array::from_fn(|index| Symbol::Token(bytes[index])))
}

#[test]
fn additions_match_exhaustive_pyalcs_probes_without_extending_p9() {
    let data = fixture("goal_maze_probes");
    assert_eq!(data["mazes"].as_array().unwrap().len(), 3);
    assert_eq!(VALIDATION_MAZES.len(), 5);
    assert_eq!(PERFORMANCE_MAZES.len(), 22);
    assert_eq!(RESEARCH_MAZES.len(), 6);
    for entry in data["mazes"].as_array().unwrap() {
        let geometry = geometry_by_id(entry["id"].as_str().unwrap()).unwrap();
        assert_eq!(
            geometry.max_episode_steps,
            entry["max_episode_steps"].as_u64().unwrap() as u32
        );
        let topology = MazeTopology::new(geometry).unwrap();
        for (row, values) in geometry.matrix.iter().enumerate() {
            for (col, &value) in values.iter().enumerate() {
                assert_eq!(u64::from(value), entry["grid"][row][col].as_u64().unwrap());
            }
        }
        for probe in entry["probes"].as_array().unwrap() {
            let start = (
                probe["row"].as_u64().unwrap() as usize,
                probe["col"].as_u64().unwrap() as usize,
            );
            let action = probe["action"].as_u64().unwrap() as usize;
            let mut maze = Maze::from_geometry(geometry, rng(0));
            maze.place_agent(start.0, start.1);
            assert_eq!(maze.perception(), perception(&probe["perception_before"]));
            assert_eq!(topology.perception_at(start), maze.perception());
            let result = maze.step(action);
            assert_eq!(maze.agent_position(), topology.next_cell(start, action));
            assert_eq!(result.observation, perception(&probe["perception_after"]));
            assert_eq!(result.reward, probe["reward"].as_f64().unwrap());
            assert_eq!(result.terminated, probe["done"].as_bool().unwrap());
        }
    }
}

#[test]
fn maze6_removes_only_the_documented_maze7_wall() {
    let mut differences = Vec::new();
    for (row, values) in pyalcs::MAZE7.matrix.iter().enumerate() {
        for (col, &value) in values.iter().enumerate() {
            if value != pyalcs::MAZE6.matrix[row][col] {
                differences.push((row, col, value, pyalcs::MAZE6.matrix[row][col]));
            }
        }
    }
    assert_eq!(differences, [(3, 5, 1, 0)]);
    for (research, comparison) in [(pyalcs::MAZEF3, alcs::MAZEF3), (pyalcs::MAZEB, alcs::MAZEB)] {
        assert_eq!(research.matrix, comparison.matrix);
        assert_eq!(research.max_episode_steps, 50);
        assert_eq!(comparison.max_episode_steps, 200);
    }
}

#[test]
fn knowledge_matches_pyalcs_multisets_for_every_geometry() {
    let data = fixture("maze_knowledge");
    assert_eq!(
        data["mazes"].as_array().unwrap().len(),
        all_geometries().count()
    );
    for entry in data["mazes"].as_array().unwrap() {
        let id = entry["id"].as_str().unwrap();
        let geometry = geometry_by_id(id).unwrap();
        let topology = MazeTopology::new(geometry).unwrap();
        let mut expected = Vec::new();
        for transition in entry["transitions"].as_array().unwrap() {
            let start = cell(&transition[0]);
            let action = transition[1].as_u64().unwrap() as usize;
            let end = cell(&transition[2]);
            assert!(topology.path_cells().contains(&start));
            assert_ne!(start, end);
            assert_eq!(topology.next_cell(start, action), end);
            expected.push((
                perception(&transition[3]).symbols,
                action,
                perception(&transition[4]).symbols,
            ));
        }
        let mut actual: Vec<_> = pyalcs_transitions(geometry)
            .into_iter()
            .map(|t| (t.p0.symbols, t.action, t.p1.symbols))
            .collect();
        actual.sort();
        expected.sort();
        assert_eq!(actual, expected, "{id}");
        if id == "Maze4-v0" {
            assert_eq!(actual.len(), 115);
        }
    }
}

#[test]
fn multi_goal_knowledge_adds_reward_cell_departures_and_goal_wildcards() {
    for geometry in all_geometries() {
        let topology = MazeTopology::new(geometry).unwrap();
        let reward = topology.reward_cell();
        let departures = (0..8)
            .filter(|&action| topology.next_cell(reward, action) != reward)
            .count();
        assert_eq!(
            multi_goal_transitions(geometry).len(),
            pyalcs_transitions(geometry).len() + departures
        );
        let joined = goal_transitions::<2, 10>(geometry, MazeKnowledgeSet::MultiGoalWalkable);
        let plain = multi_goal_transitions(geometry);
        for (full, state) in joined.iter().zip(plain) {
            assert_eq!(
                GoalLayout::<8, 2, 10>::split(&full.p0),
                (state.p0, Goal::new([Symbol::Wildcard; 2]))
            );
            assert_eq!(
                GoalLayout::<8, 2, 10>::split(&full.p1),
                (state.p1, Goal::new([Symbol::Wildcard; 2]))
            );
        }
    }
}

#[test]
fn perception_goal_twins_are_refused_even_when_outside_the_pool() {
    for geometry in all_geometries() {
        let topology = MazeTopology::new(geometry).unwrap();
        for &goal in topology.walkable_cells() {
            let twin = topology.walkable_cells().iter().find(|&&candidate| {
                candidate != goal
                    && topology.perception_at(candidate) == topology.perception_at(goal)
            });
            let result = PerceptionGoalMaze::multi_goal(geometry, vec![goal], 5, rng(0));
            match (twin, result) {
                (Some(&twin), Err(error)) => {
                    assert_eq!(error, GoalMazeError::PerceptionTwin { goal, twin })
                }
                (None, Ok(_)) => {}
                _ => panic!("{} goal {goal:?}", geometry.id),
            }
        }
    }
}

#[test]
fn coordinate_goals_do_not_reward_the_maze_f3_perception_twin() {
    let mut maze =
        CoordinateGoalMaze::multi_goal(&pyalcs::MAZEF3, vec![(1, 4)], 5, rng(0)).unwrap();
    assert_eq!(
        maze.topology().perception_at((1, 4)),
        maze.topology().perception_at((3, 3))
    );
    let desired = maze.goal_at((1, 4));
    maze.reset_at((3, 2), desired);
    let step = maze.step(2);
    assert_eq!(maze.position(), (3, 3));
    assert_eq!(
        step.achieved,
        Goal::new([Symbol::Token(3), Symbol::Token(3)])
    );
    let outcome = step.outcome(maze.objective(), &desired);
    assert_eq!(outcome.reward, 0.0);
    assert!(!outcome.terminated);
    assert!(!outcome.truncated);
}

#[test]
fn every_coordinate_maze_transition_is_scored_by_the_objective() {
    for geometry in all_geometries() {
        let mut maze = CoordinateGoalMaze::all_walkable(geometry, 1, rng(0)).unwrap();
        let cells = maze.topology().walkable_cells().to_vec();
        for &goal in &cells {
            let desired = maze.goal_at(goal);
            for &start in &cells {
                if start == goal {
                    continue;
                }
                for action in 0..8 {
                    maze.reset_at(start, desired);
                    let expected_cell = maze.topology().next_cell(start, action);
                    let step = maze.step(action);
                    assert_eq!(step.achieved, maze.goal_at(expected_cell));
                    assert!(!step.terminal_state);
                    let outcome = step.outcome(&SPARSE_GOAL_REWARD, &desired);
                    assert_eq!(
                        outcome.reward,
                        if expected_cell == goal { 1000.0 } else { 0.0 }
                    );
                    assert_eq!(outcome.terminated, expected_cell == goal);
                    assert_eq!(outcome.truncated, expected_cell != goal);
                }
            }
        }
    }
}

#[test]
fn the_original_reward_cell_is_a_walkable_landmark_in_multi_goal_mode() {
    let mut maze = CoordinateGoalMaze::multi_goal(&pyalcs::MAZE4, vec![(1, 5)], 3, rng(0)).unwrap();
    let desired = maze.goal_pool()[0];
    maze.reset_at((2, 5), desired);
    let enter = maze.step(1);
    assert_eq!(maze.position(), (1, 6));
    assert!(!enter.outcome(maze.objective(), &desired).terminated);
    let exit = maze.step(6);
    assert_eq!(maze.position(), (1, 5));
    assert_eq!(exit.observation.symbols[2], Symbol::Token(b'9'));
    assert_eq!(exit.outcome(maze.objective(), &desired).reward, 1000.0);
}

#[test]
fn truncation_and_limit_step_success_survive_the_adapter() {
    for action in [0, 2] {
        let maze =
            CoordinateGoalMaze::multi_goal(&pyalcs::MAZEF3, vec![(1, 2)], 1, Box::new(IndexRng(0)))
                .unwrap();
        let mut adapter = GoalConditioned::<_, 8, 2, 10>::new(maze);
        let state = adapter.reset_with_goal(Goal::new([Symbol::Token(1), Symbol::Token(2)]));
        let start = adapter.environment().position();
        let desired = adapter.desired().unwrap();
        let expected = adapter.environment().topology().next_cell(start, action);
        let result = adapter.step(action);
        assert_eq!(&state.symbols[8..], &desired.symbols);
        assert_eq!(&result.observation.symbols[8..], &desired.symbols);
        assert_eq!(result.terminated, expected == (1, 2));
        assert_eq!(result.truncated, expected != (1, 2));
        assert_eq!(adapter.desired(), None);
    }
}

#[test]
fn a_step_cap_truncates_at_the_exact_step_and_success_wins() {
    let mut maze =
        CoordinateGoalMaze::multi_goal(&pyalcs::MAZEF3, vec![(1, 2)], 3, rng(0)).unwrap();
    let desired = maze.goal_pool()[0];
    for final_action in [0, 2] {
        maze.reset_at((1, 1), desired);
        for _ in 0..2 {
            let step = maze.step(0);
            assert!(!step.time_limit_reached);
            assert!(!step.outcome(maze.objective(), &desired).terminated);
        }
        let step = maze.step(final_action);
        let outcome = step.outcome(maze.objective(), &desired);
        assert_eq!(outcome.terminated, final_action == 2);
        assert_eq!(outcome.truncated, final_action == 0);
    }
}

#[test]
fn limit_step_goal_success_has_zero_bootstrap_for_every_learning_path() {
    fn check<A: LearningAgent<10>, E: Environment<10>>(agent: &mut A, env: &mut E, exploit: bool) {
        let selector = EpsilonGreedy {
            number_of_possible_actions: 8,
            epsilon: 0.0,
        };
        let metrics = if exploit {
            agent.run_exploit_trial(env, &MaxFitnessBootstrap, 100)
        } else {
            agent.run_explore_trial(env, &selector, &MaxFitnessBootstrap, 100)
        };
        assert_eq!(metrics.steps, 1);
        assert_eq!(metrics.reward, 1_000.0);
        assert_eq!(agent.population().get(0).r, 504.0);
        assert_eq!(agent.population().get(1).r, 40.0);
    }

    for mode in [TruncationMode::Bootstrap, TruncationMode::Pyalcs] {
        for replay in [false, true] {
            for exploit in [false, true] {
                let maze = CoordinateGoalMaze::multi_goal(
                    &pyalcs::MAZEF3,
                    vec![(1, 2)],
                    1,
                    Box::new(IndexRng(0)),
                )
                .unwrap();
                let mut env = GoalConditioned::<_, 8, 2, 10>::new(maze);
                let start = env.reset();
                let outcome = env.step(2);
                assert!(outcome.terminated);
                assert!(!outcome.truncated);
                let config = Configuration {
                    beta: 0.5,
                    gamma: 0.75,
                    ..Configuration::default_protocol()
                };
                let mut acting = cover(&start, 2, &outcome.observation, 0, &config);
                acting.r = 8.0;
                let mut next = Classifier::general(Some(2), &config);
                next.condition.symbols = outcome.observation.symbols;
                next.effect.set(0, Symbol::Token(b'9'));
                next.r = 40.0;
                let population = Population::from_classifiers(vec![acting, next]);
                if replay {
                    let replay_config = ReplayConfiguration {
                        buffer_size: 4,
                        min_samples: 1,
                        samples_number: 1,
                    };
                    let mut agent = Acs2ErAgent::with_population(
                        config,
                        replay_config,
                        ChaChaRandomSource::from_seed(42),
                        population,
                    )
                    .with_truncation_mode(mode);
                    check(&mut agent, &mut env, exploit);
                    if !exploit {
                        assert!(agent.replay_memory().get(0).done);
                    }
                } else {
                    let mut agent = Agent::with_population(
                        config,
                        ChaChaRandomSource::from_seed(42),
                        population,
                    )
                    .with_truncation_mode(mode);
                    check(&mut agent, &mut env, exploit);
                }
            }
        }
    }
}

#[test]
fn single_goal_matches_maze_step_for_step_including_reset_randomness() {
    for geometry in all_geometries() {
        for seed in 42..45 {
            let mut plain = Maze::from_geometry(geometry, rng(seed));
            let goal = CoordinateGoalMaze::single_goal(geometry, rng(seed)).unwrap();
            let mut joined = GoalConditioned::<_, 8, 2, 10>::new(goal);
            let mut actions = ChaChaRandomSource::from_seed_and_stream(seed, 2);
            for _ in 0..20 {
                let start = plain.reset();
                assert_eq!(GoalLayout::<8, 2, 10>::split(&joined.reset()).0, start);
                assert_eq!(joined.environment().position(), plain.agent_position());
                loop {
                    let action = actions.gen_range(8);
                    let expected = plain.step(action);
                    let actual = joined.step(action);
                    assert_eq!(
                        GoalLayout::<8, 2, 10>::split(&actual.observation).0,
                        expected.observation
                    );
                    assert_eq!(actual.reward, expected.reward);
                    assert_eq!(actual.terminated, expected.terminated);
                    assert_eq!(actual.truncated, expected.truncated);
                    if expected.terminated || expected.truncated {
                        break;
                    }
                }
            }
        }
    }
}

fn compare_classifier<const M: usize>(plain: &Classifier<8>, joined: &Classifier<M>) {
    for position in 0..8 {
        assert_eq!(
            plain.condition.get(position),
            joined.condition.get(position)
        );
        assert_eq!(plain.effect.get(position), joined.effect.get(position));
        assert_eq!(
            plain.mark.attributes[position],
            joined.mark.attributes[position]
        );
    }
    for position in 8..M {
        assert!(joined.condition.get(position).is_wildcard());
        assert!(joined.effect.get(position).is_wildcard());
        assert!(joined.mark.attributes[position].len() <= 1);
    }
    assert_eq!(plain.action, joined.action);
    assert_eq!(plain.q.to_bits(), joined.q.to_bits());
    assert_eq!(plain.r.to_bits(), joined.r.to_bits());
    assert_eq!(plain.ir.to_bits(), joined.ir.to_bits());
    assert_eq!(plain.tav.to_bits(), joined.tav.to_bits());
    assert_eq!(
        (plain.num, plain.exp, plain.talp, plain.tga, plain.ee),
        (joined.num, joined.exp, joined.talp, joined.tga, joined.ee)
    );
}

fn learning_equivalence<const G: usize, const M: usize, E: GoalEnvironment<8, G>>(
    geometry: &MazeGeometry,
    seed: u64,
    goal_env: E,
    mode: TruncationMode,
) {
    let config = Configuration::default_protocol();
    let mut plain_agent = Agent::<8, _>::new(config.clone(), ChaChaRandomSource::from_seed(seed))
        .with_truncation_mode(mode);
    let mut goal_agent = Agent::<M, _>::new(config, ChaChaRandomSource::from_seed(seed))
        .with_truncation_mode(mode);
    let mut plain = Maze::from_geometry(geometry, rng(seed));
    let mut joined = GoalConditioned::<_, 8, G, M>::new(goal_env);
    let selector = EpsilonGreedy {
        number_of_possible_actions: 8,
        epsilon: 0.8,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut time = 0;
    for _ in 0..100 {
        let expected = plain_agent.run_explore_trial(&mut plain, &selector, &bootstrap, time);
        let actual = goal_agent.run_explore_trial(&mut joined, &selector, &bootstrap, time);
        assert_eq!(expected.steps, actual.steps);
        assert_eq!(expected.reward.to_bits(), actual.reward.to_bits());
        assert_eq!(plain_agent.capture().rng, goal_agent.capture().rng);
        assert_eq!(
            plain_agent.population().len(),
            goal_agent.population().len()
        );
        for (a, b) in plain_agent
            .population()
            .iter()
            .zip(goal_agent.population().iter())
        {
            compare_classifier(a, b);
        }
        time += u64::from(expected.steps);
    }
}

#[test]
fn a_constant_goal_suffix_preserves_acs2_learning_and_rng_without_ga() {
    for geometry in pyalcs::GEOMETRIES {
        for (seed, mode) in (42..45).flat_map(|seed| {
            [TruncationMode::Bootstrap, TruncationMode::Pyalcs].map(|mode| (seed, mode))
        }) {
            learning_equivalence::<2, 10, _>(
                geometry,
                seed,
                CoordinateGoalMaze::single_goal(geometry, rng(seed)).unwrap(),
                mode,
            );
            if let Ok(goal) = PerceptionGoalMaze::single_goal(geometry, rng(seed)) {
                learning_equivalence::<8, 16, _>(geometry, seed, goal, mode);
            }
        }
    }
}

#[test]
fn seeded_resets_and_moves_reproduce_and_starts_exclude_only_the_goal() {
    let mut first = CoordinateGoalMaze::all_walkable(&pyalcs::MAZE4, 7, rng(42)).unwrap();
    let mut second = CoordinateGoalMaze::all_walkable(&pyalcs::MAZE4, 7, rng(42)).unwrap();
    let mut actions = ChaChaRandomSource::from_seed_and_stream(42, 2);
    let goals = first.goal_pool().to_vec();
    assert_eq!(goals.len(), 27);
    for desired in goals.iter().cycle().take(100) {
        let a = first.reset_with_goal(*desired);
        let b = second.reset_with_goal(*desired);
        assert_eq!(a, b);
        assert_ne!(a.achieved, a.desired);
        loop {
            let action = actions.gen_range(8);
            let a = first.step(action);
            let b = second.step(action);
            assert_eq!(a, b);
            let result = a.outcome(first.objective(), desired);
            if result.terminated || result.truncated {
                break;
            }
        }
    }
}

#[test]
fn shortest_distances_obey_the_maze_graph_and_goal_pool() {
    let data = fixture("maze_knowledge");
    for entry in data["mazes"].as_array().unwrap() {
        let geometry = geometry_by_id(entry["id"].as_str().unwrap()).unwrap();
        let maze = CoordinateGoalMaze::all_walkable(geometry, 5, rng(0)).unwrap();
        let cells: Vec<_> = entry["walkable_cells"]
            .as_array()
            .unwrap()
            .iter()
            .map(cell)
            .collect();
        assert_eq!(maze.goal_cells(), cells);
        for (row, &start) in cells.iter().enumerate() {
            for (col, &goal) in cells.iter().enumerate() {
                let expected = entry["distances"][row][col]
                    .as_u64()
                    .map(|distance| distance as u32);
                assert_eq!(
                    maze.distance_to_goal(start, &maze.goal_at(goal)),
                    expected,
                    "{} {start:?} -> {goal:?}",
                    geometry.id
                );
            }
        }
        assert_eq!(
            maze.topology()
                .shortest_distance((0, 0), maze.goal_cells()[0]),
            None
        );
    }
}

struct IndexRng(usize);

impl RandomSource for IndexRng {
    fn gen_bool(&mut self, _probability: f64) -> bool {
        false
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        assert!(self.0 < bound);
        self.0
    }

    fn gen_unit(&mut self) -> f64 {
        0.0
    }
}

#[test]
fn every_allowed_start_has_one_random_draw_index_for_each_goal() {
    let topology = MazeTopology::new(&pyalcs::MAZE4).unwrap();
    for &goal in topology.walkable_cells() {
        let mut starts = Vec::new();
        for index in 0..topology.walkable_cells().len() - 1 {
            let mut maze = CoordinateGoalMaze::multi_goal(
                &pyalcs::MAZE4,
                vec![goal],
                3,
                Box::new(IndexRng(index)),
            )
            .unwrap();
            maze.reset_with_goal(maze.goal_pool()[0]);
            starts.push(maze.position());
        }
        let expected: Vec<_> = topology
            .walkable_cells()
            .iter()
            .copied()
            .filter(|&cell| cell != goal)
            .collect();
        assert_eq!(starts, expected);
    }
}

#[test]
fn invalid_pools_and_zero_caps_are_refused() {
    for (pool, cap, error) in [
        (vec![], 3, GoalMazeError::EmptyPool),
        (vec![(0, 0)], 3, GoalMazeError::InvalidGoalCell((0, 0))),
        (
            vec![(1, 1), (1, 1)],
            3,
            GoalMazeError::DuplicateGoalCell((1, 1)),
        ),
        (vec![(1, 1)], 0, GoalMazeError::ZeroStepCap),
    ] {
        assert_eq!(
            CoordinateGoalMaze::multi_goal(&pyalcs::MAZE4, pool, cap, rng(0)).err(),
            Some(error)
        );
    }
}

#[test]
fn maze_distances_also_cover_unambiguous_goals_outside_the_real_pool() {
    let maze = CoordinateGoalMaze::multi_goal(&pyalcs::MAZE4, vec![(1, 1)], 3, rng(0)).unwrap();
    assert_eq!(
        maze.distance_to_goal((1, 5), &maze.goal_at((1, 6))),
        Some(1)
    );
    let ambiguous =
        PerceptionGoalMaze::multi_goal(&pyalcs::MAZEF3, vec![(1, 1)], 3, rng(0)).unwrap();
    assert_eq!(
        ambiguous.distance_to_goal((1, 1), &ambiguous.goal_at((1, 4))),
        None
    );
}
