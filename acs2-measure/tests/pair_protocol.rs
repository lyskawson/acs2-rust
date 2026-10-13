use std::collections::BTreeSet;

use acs2_core::checkpoint::Checkpointed;
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::perception::Perception;
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::geometry_by_id;
use acs2_measure::pair_protocol::{
    goal_starts, registered_configuration, validate_partition, MazePairDistributions, MazePairTask,
};
use acs2_measure::reference::reference;
use acs2_measure::runner::{
    assert_protocol_evaluation_read_only, run_goal_agent_protocol_with_sink,
    run_protocol_with_sink, AgentKind, CoreAgent, GoalAgent, MeasuredEnvironment, Preset,
    RunMetadata, RunSettings,
};
use acs2_measure::task::Task;
use serde_json::{json, Value};

fn names() -> [&'static str; 6] {
    ["maze4", "maze5", "maze6", "maze7", "woods14", "mazema"]
}

fn distributions(name: &str) -> MazePairDistributions<Coordinates, 2> {
    let configuration = registered_configuration(name);
    MazePairDistributions::registered(
        name,
        geometry_by_id(configuration["geometry"].as_str().unwrap()).unwrap(),
        configuration["cap"].as_u64().unwrap() as u32,
        "coordinates",
    )
}

#[test]
fn every_goal_is_in_all_pair_blocks_and_missing_or_overlapping_pairs_are_refused() {
    for name in names() {
        let distributions = distributions(name);
        let configuration = &distributions.registration;
        let geometry = distributions.training.maze.geometry;
        validate_partition(geometry, configuration);
        let mut missing = configuration.clone();
        missing["blocks"]["test"]["goal_starts"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(std::panic::catch_unwind(|| validate_partition(geometry, &missing)).is_err());
        let mut overlap = configuration.clone();
        overlap["blocks"]["test"]["goal_starts"][0]["starts"][0] =
            configuration["blocks"]["training"]["goal_starts"][0]["starts"][0].clone();
        assert!(std::panic::catch_unwind(|| validate_partition(geometry, &overlap)).is_err());
    }
}

#[derive(Default)]
struct ExposureProbe {
    registered: Vec<Goal<2>>,
    exposed: Vec<Goal<2>>,
    starts: Vec<(Goal<2>, Goal<2>)>,
    steps: u64,
}

impl GoalAgent<8, 2, 10> for ExposureProbe {
    fn name(&self) -> &'static str {
        "pair_exposure_probe"
    }
    fn register_training_goals(&mut self, goals: &[Goal<2>]) {
        self.registered = goals.to_vec();
    }
    fn train_episode<E: MeasuredEnvironment<8, 2, 10>>(&mut self, env: &mut E, _time: u64) {
        self.exposed = env.training_goals().unwrap().to_vec();
        let joined = env.reset();
        let start = env.episode_start().unwrap();
        assert_eq!(GoalLayout::<8, 2, 10>::split(&joined).1, start.desired);
        self.starts.push((start.achieved, start.desired));
        loop {
            let outcome = env.step(0);
            self.steps += 1;
            if outcome.terminated || outcome.truncated {
                break;
            }
        }
    }
    fn declared_policy(&self) -> &'static str {
        "action_zero"
    }
    fn eval_action(&self, _state: &Perception<10>, _rng: &mut dyn RandomSource) -> (usize, f64) {
        (0, 0.0)
    }
    fn online_updates(&self) -> u64 {
        self.steps
    }
    fn replay_updates(&self) -> u64 {
        0
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn population_classifiers(&self) -> usize {
        0
    }
    fn population_numerosity(&self) -> u32 {
        0
    }
    fn population_logical_bytes(&self) -> usize {
        0
    }
    fn population_mark_entries(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn agent_parameters(&self) -> Value {
        json!({})
    }
}

#[test]
fn agent_boundary_exposes_all_goals_but_draws_only_registered_training_pairs() {
    for name in names() {
        let distributions = distributions(name);
        let task = &distributions.training;
        let pairs: BTreeSet<_> = goal_starts(&distributions.registration, "training")
            .into_iter()
            .flat_map(|(goal, starts)| {
                starts.into_iter().map(move |start| {
                    (
                        task.maze.template.goal_at(start),
                        task.maze.template.goal_at(goal),
                    )
                })
            })
            .collect();
        for seed in 42..=46 {
            let mut agent = ExposureProbe::default();
            run_goal_agent_protocol_with_sink::<_, _, _, 8, 2, 10>(
                &distributions.protocol(),
                &mut agent,
                seed,
                &[5_000],
                &RunMetadata::test("test"),
                false,
                &mut |_| {},
            );
            let all_goals = task.maze.template.goal_pool();
            assert_eq!(agent.registered, all_goals);
            assert_eq!(agent.exposed, all_goals);
            assert!(
                agent.starts.iter().all(|pair| pairs.contains(pair)),
                "a held-out pair reached training"
            );
            assert_eq!(
                agent
                    .starts
                    .iter()
                    .map(|&(_, goal)| goal)
                    .collect::<BTreeSet<_>>()
                    .len(),
                all_goals.len()
            );
        }
    }
}

#[test]
fn each_pair_block_has_its_own_pairs_exact_weights_floor_ceiling_and_knowledge() {
    for name in names() {
        let distributions = distributions(name);
        let mut rows = Vec::new();
        run_protocol_with_sink::<_, _, 8, 2, 10>(
            &distributions.protocol(),
            AgentKind::Acs2,
            42,
            &[50],
            &RunMetadata::test("test"),
            RunSettings {
                evaluate_points: true,
                capture_final_state: false,
            },
            &mut |row| rows.push(row),
        );
        let row = &rows[0];
        assert_eq!(row["schema"], 5);
        assert!(row.get("success").is_none());
        assert_eq!(row["goal_protocol"]["id"], "maze-pairs-v1");
        assert!(row["knowledge"]
            .as_f64()
            .is_some_and(|value| (0.0..=1.0).contains(&value)));
        let transitions = <MazePairTask<Coordinates, 2> as Task<8, 2, 10>>::knowledge_transitions(
            &distributions.training,
        )
        .unwrap();
        assert_eq!(row["knowledge_transition_goal_pairs"], transitions.len());
        for (name, task) in [
            ("training", &distributions.training),
            ("validation", &distributions.validation),
            ("test", &distributions.test),
        ] {
            let pairs = <MazePairTask<Coordinates, 2> as Task<8, 2, 10>>::pairs(task);
            let evaluated = &row["evaluation_blocks"][name];
            let registered = &distributions.registration["blocks"][name];
            assert_eq!(
                evaluated["evaluation_pairs"],
                registered["evaluation_pairs"]
            );
            let independent = reference::<_, 8, 2, 10>(&task.maze, &pairs);
            assert!(
                (independent.random_success - registered["random_floor"].as_f64().unwrap()).abs()
                    < 1e-14
            );
            assert!(
                (independent.reachable_within_cap
                    - registered["reachable_within_cap"].as_f64().unwrap())
                .abs()
                    < 1e-12
            );
            assert_eq!(evaluated["random_floor"], registered["random_floor"]);
            assert_eq!(
                evaluated["reachable_within_cap"],
                registered["reachable_within_cap"]
            );
            let starts = evaluated["starts"].as_array().unwrap();
            assert_eq!(starts.len(), pairs.len());
            let mut success = 0.0;
            for (actual, &(start, goal, weight)) in starts.iter().zip(&pairs) {
                assert_eq!(actual["start"], format!("{start:?}"));
                assert_eq!(actual["goal"], format!("{goal:?}"));
                assert_eq!(actual["weight"].as_f64().unwrap(), 1.0 / pairs.len() as f64);
                if actual["success"].as_bool().unwrap() {
                    success += weight;
                }
            }
            assert_eq!(evaluated["success"].as_f64().unwrap(), success);
        }
    }
}

#[test]
fn pair_evaluation_and_knowledge_preserve_agent_replay_rng_and_training_counters() {
    let preset = Preset::thesis();
    for name in names() {
        let distributions = distributions(name);
        assert_protocol_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
            &distributions.protocol(),
            42,
            &[50, 100],
            || CoreAgent {
                agent: acs2_agents::acs2::Agent::<10, _>::new(
                    preset.config(8),
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                )
                .with_truncation_mode(preset.truncation),
                preset,
                replay: false,
                steps: 0,
                updates: 0,
            },
            |agent| {
                format!(
                    "{:?};{};{}",
                    agent.agent.capture(),
                    agent.steps,
                    agent.updates
                )
            },
        );
        assert_protocol_evaluation_read_only::<_, _, _, _, 8, 2, 10>(
            &distributions.protocol(),
            42,
            &[50, 100],
            || CoreAgent {
                agent: acs2_agents::acs2er::Agent::<10, _>::new(
                    preset.config(8),
                    acs2_agents::acs2er::replay::ReplayConfiguration {
                        buffer_size: 10_000,
                        min_samples: 1,
                        samples_number: 3,
                    },
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                )
                .with_truncation_mode(preset.truncation),
                preset,
                replay: true,
                steps: 0,
                updates: 0,
            },
            |agent| {
                format!(
                    "{:?};{};{}",
                    agent.agent.capture(),
                    agent.steps,
                    agent.updates
                )
            },
        );
    }
}
