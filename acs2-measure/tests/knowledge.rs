use acs2_core::classifier::Classifier;
use acs2_core::goal::GoalLayout;
use acs2_core::population::Population;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::support::knowledge::{anticipation_fraction, Transition};
use acs2_core::symbol::Symbol;
use acs2_envs::goal::maze::{CoordinateGoalMaze, Coordinates};
use acs2_envs::maze::geometries::geometry_by_id;
use acs2_measure::knowledge::maze_transitions;
use acs2_measure::runner::{CoreAgent, GoalAgent, Preset};

const OFFSETS: [(isize, isize); 8] = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
];

fn maze(name: &str) -> CoordinateGoalMaze {
    CoordinateGoalMaze::all_walkable(
        geometry_by_id(name).unwrap(),
        50,
        Box::new(ChaChaRandomSource::from_seed(0)),
    )
    .unwrap()
}

fn exact_classifier(transition: &Transition<10>) -> Classifier<10> {
    let mut classifier = Classifier::general(Some(transition.action), &Preset::thesis().config(8));
    classifier.condition.symbols = transition.p0.symbols;
    classifier.q = 1.0;
    for index in 0..10 {
        if transition.p0.symbols[index] != transition.p1.symbols[index] {
            classifier.effect.set(index, transition.p1.symbols[index]);
        }
    }
    classifier
}

fn fraction(population: &Population<10>, maze: &CoordinateGoalMaze) -> f64 {
    let preset = Preset::thesis();
    let agent = CoreAgent {
        agent: acs2_core::agent::Agent::<10, _>::with_population(
            preset.config(8),
            ChaChaRandomSource::from_seed(42),
            Population::from_classifiers(population.classifiers().to_vec()),
        ),
        preset,
        replay: false,
        steps: 0,
        updates: 0,
    };
    <CoreAgent<_, 10> as GoalAgent<8, 2, 10>>::knowledge(
        &agent,
        &maze_transitions::<Coordinates, 2, 10>(maze),
    )
    .unwrap()
}

fn independent_fraction(population: &Population<10>, maze: &CoordinateGoalMaze) -> (usize, usize) {
    let topology = maze.topology();
    let mut hits = 0;
    let mut total = 0;
    for &goal_cell in topology.walkable_cells() {
        for &start in topology.walkable_cells() {
            if start == goal_cell {
                continue;
            }
            for (action, &(dr, dc)) in OFFSETS.iter().enumerate() {
                let next = (
                    (start.0 as isize + dr) as usize,
                    (start.1 as isize + dc) as usize,
                );
                if !topology.walkable_cells().contains(&next) {
                    continue;
                }
                let desired = maze.goal_at(goal_cell);
                let before = GoalLayout::<8, 2, 10>::join(&topology.perception_at(start), &desired);
                let after = GoalLayout::<8, 2, 10>::join(&topology.perception_at(next), &desired);
                total += 1;
                if population.iter().any(|classifier| {
                    classifier.q > 0.9
                        && classifier.action == Some(action)
                        && (0..10).all(|index| {
                            let condition = classifier.condition.symbols[index];
                            let effect = classifier.effect.symbols[index];
                            (condition == Symbol::Wildcard || condition == before.symbols[index])
                                && if effect == Symbol::Wildcard {
                                    before.symbols[index] == after.symbols[index]
                                } else {
                                    before.symbols[index] != after.symbols[index]
                                        && effect == after.symbols[index]
                                }
                        })
                }) {
                    hits += 1;
                }
            }
        }
    }
    (hits, total)
}

#[test]
fn reliable_anticipation_of_every_transition_under_every_goal_gives_one() {
    for name in [
        "Maze4-v0",
        "Maze5-v0",
        "Maze6-v0",
        "Maze7-v0",
        "Woods14-v0",
        "MazeMA-v0",
    ] {
        let maze = maze(name);
        let transitions = maze_transitions::<Coordinates, 2, 10>(&maze);
        let classifiers = transitions.iter().map(exact_classifier).collect();
        let population = Population::from_classifiers(classifiers);
        assert_eq!(fraction(&population, &maze), 1.0);
        let independent = independent_fraction(&population, &maze);
        assert_eq!(independent.0, independent.1);
        assert_eq!(independent.1, transitions.len());
        assert!(transitions.iter().all(|transition| {
            transition.p0 != transition.p1
                && transition.p0.symbols[8..] == transition.p1.symbols[8..]
        }));
    }
}

#[test]
fn empty_population_and_quality_at_the_threshold_give_zero() {
    let maze = maze("MazeMA-v0");
    assert_eq!(fraction(&Population::new(), &maze), 0.0);
    let transitions = maze_transitions::<Coordinates, 2, 10>(&maze);
    let population = Population::from_classifiers(
        transitions
            .iter()
            .map(|transition| {
                let mut classifier = exact_classifier(transition);
                classifier.q = 0.9;
                classifier
            })
            .collect(),
    );
    assert_eq!(fraction(&population, &maze), 0.0);
}

#[test]
fn a_goal_specific_classifier_counts_only_for_its_own_goal() {
    let maze = maze("MazeMA-v0");
    let transitions = maze_transitions::<Coordinates, 2, 10>(&maze);
    let first = &transitions[0];
    let population = Population::from_classifiers(vec![exact_classifier(first)]);
    let own_goal = GoalLayout::<8, 2, 10>::split(&first.p0).1;
    let (hits, total) = independent_fraction(&population, &maze);
    assert_eq!(hits, 1);
    assert_eq!(fraction(&population, &maze), 1.0 / total as f64);
    let other_goals = transitions
        .into_iter()
        .filter(|transition| GoalLayout::<8, 2, 10>::split(&transition.p0).1 != own_goal);
    assert_eq!(anticipation_fraction(&population, 0.9, other_goals), 0.0);
}

#[test]
fn fixed_mixed_population_equals_independent_complete_enumeration() {
    let maze = maze("Maze6-v0");
    let transitions = maze_transitions::<Coordinates, 2, 10>(&maze);
    let classifiers = transitions
        .iter()
        .step_by(13)
        .enumerate()
        .map(|(index, transition)| {
            let mut classifier = exact_classifier(transition);
            match index % 4 {
                0 => classifier.q = 0.9,
                1 => {
                    classifier.condition.set(8, Symbol::Wildcard);
                    classifier.condition.set(9, Symbol::Wildcard);
                }
                2 => classifier.action = Some((transition.action + 1) % 8),
                _ => {}
            }
            classifier
        })
        .collect();
    let population = Population::from_classifiers(classifiers);
    let (hits, total) = independent_fraction(&population, &maze);
    assert!(hits > 0 && hits < total);
    assert_eq!(total, transitions.len());
    assert_eq!(fraction(&population, &maze), hits as f64 / total as f64);
}
