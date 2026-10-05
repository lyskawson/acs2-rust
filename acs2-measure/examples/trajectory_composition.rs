use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::time::Instant;

use acs2_core::environment::Environment;
use acs2_core::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::trial::TruncationMode;
use acs2_envs::goal::bit_flipping::BitFlipping;
use acs2_envs::goal::hand_eye::{position_goal, HandEye4, HandEye5};
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::goal::taxi::passenger_goal;
use acs2_envs::maze::geometries::pyalcs::{MAZE4, MAZE6, MAZE7};
use acs2_envs::maze::geometries::MazeGeometry;
use acs2_envs::maze::topology::MazeTopology;
use acs2_measure::runner::{
    MeasuredEnvironment, TrainingEnvironment, AGENT_STREAM, ENVIRONMENT_STREAM, POOL_STREAM,
};
use acs2_measure::task::{BitTask, HandEyeTask, MazeTask, Task, TaxiTask};
use acs2_measure::trajectory::MeasuredGoalEvaluator;
use acs2_trajectory::{
    episode_composition, Admissibility, ExpectedComposition, GoalStrategy, ObjectiveCost,
    SamplerConfiguration, Selection, TrajectoryStore, SAMPLER_STREAM,
};
use serde_json::{json, Value};

#[path = "../src/output.rs"]
mod output;

fn configurations() -> Vec<SamplerConfiguration> {
    let mut result = Vec::new();
    for strategy in [
        GoalStrategy::Final,
        GoalStrategy::Future,
        GoalStrategy::Episode,
        GoalStrategy::UniformReal,
    ] {
        for admissibility in [
            Admissibility::EveryTransition,
            Admissibility::FromNonGoalState,
            Admissibility::CounterfactualEpisode,
        ] {
            for candidate_filter in [false, true] {
                result.push(SamplerConfiguration {
                    selection: Selection {
                        strategy,
                        admissibility,
                        candidate_filter,
                    },
                    relabeled_proportion: 1.0,
                    truncation: TruncationMode::Bootstrap,
                });
            }
        }
    }
    result
}

fn expected_json(expected: ExpectedComposition) -> Value {
    json!({
        "admissible_share": expected.admissible_share, "no_admissible_goal": expected.no_admissible_goal,
        "original": expected.original, "relabeled": expected.relabeled, "fallback": expected.fallback,
        "already_reached": expected.already_reached, "after_counterfactual_end": expected.after_counterfactual_end,
        "done": expected.done, "outside_candidates": expected.outside_candidates, "mean_reward": expected.mean_reward,
        "mean_objective_evaluations": expected.mean_objective_evaluations(),
        "mean_reward_evaluations": expected.mean_reward_evaluations, "mean_reach_evaluations": expected.mean_reach_evaluations
    })
}

fn collect<T: Task<S, G, M>, const S: usize, const G: usize, const M: usize>(
    task: &T,
    id: &str,
    seed: u64,
    target: u64,
    writer: &mut impl Write,
) {
    let started = Instant::now();
    let mut env = TrainingEnvironment::new(
        task,
        task.environment(ChaChaRandomSource::from_seed_and_stream(
            seed,
            ENVIRONMENT_STREAM,
        )),
        ChaChaRandomSource::from_seed_and_stream(seed, POOL_STREAM),
    );
    let mut policy = ChaChaRandomSource::from_seed_and_stream(seed, AGENT_STREAM);
    let mut store = TrajectoryStore::new(10_000);
    let configurations = configurations();
    let mut totals = vec![ExpectedComposition::default(); configurations.len()];
    let mut analysis_cost = ObjectiveCost::default();
    let mut successful_episodes = 0u64;
    while env.steps < target {
        env.begin_episode();
        env.reset();
        let mut episode = store.begin_episode(env.episode_start().unwrap());
        loop {
            let action = policy.gen_range(task.actions());
            let outcome = env.step(action);
            episode.push(action, env.last_transition().unwrap().step);
            if outcome.terminated || outcome.truncated {
                successful_episodes += u64::from(outcome.reward > 0.0);
                break;
            }
        }
        env.end_episode();
        let id = store.insert(episode).unwrap();
        let episode = store.episode(id).unwrap();
        let evaluator = MeasuredGoalEvaluator::<_, S, G, M>::new(&env);
        for (index, &configuration) in configurations.iter().enumerate() {
            let composition = episode_composition::<S, G, M>(
                episode,
                store.candidates(),
                configuration,
                &evaluator,
            )
            .unwrap();
            totals[index].add_weighted(composition.expected, episode.len() as f64);
            analysis_cost.add(composition.analysis_cost);
        }
    }
    assert!(env.steps - target < u64::from(task.cap()));
    for (configuration, total) in configurations.into_iter().zip(totals) {
        let mut expected = ExpectedComposition::default();
        expected.add_weighted(total, 1.0 / env.steps as f64);
        let row = json!({
            "schema": 1, "configuration": id, "task": task.name(), "cap": task.cap(), "goal_pool": task.pool_label(),
            "goal_encoding": task.encoding(), "seed": seed, "target_steps": target, "actual_steps": env.steps,
            "episodes": env.episodes, "success_rate": successful_episodes as f64 / env.episodes as f64,
            "candidate_count_at_end": store.candidates().len(), "retained_steps": store.len(),
            "strategy": configuration.selection.strategy.name(), "rule": configuration.selection.admissibility.name(),
            "candidate_filter": configuration.selection.candidate_filter, "relabeled_proportion": 1.0,
            "expected": expected_json(expected), "commit": env!("ACS2_BUILD_COMMIT"), "source_state": env!("ACS2_BUILD_SOURCE_STATE"),
            "plan_sha256": "862f77785f1cf73fc58e81a88fabeac4b708ee908082f227a499d5cb2e4efb98",
            "transition_distribution": "uniform_over_collected_steps", "candidate_time": "each_episode_completion",
            "random_streams": {"policy": AGENT_STREAM, "environment": ENVIRONMENT_STREAM, "restricted_pool": POOL_STREAM,
                "reserved_sampler": SAMPLER_STREAM}, "composition_uses_rng": false,
            "run_analysis_cost": {"reward_evaluations": analysis_cost.reward_evaluations,
                "reach_evaluations": analysis_cost.reach_evaluations}, "wall_seconds": started.elapsed().as_secs_f64()
        });
        writeln!(writer, "{row}").unwrap();
    }
    writer.flush().unwrap();
    eprintln!(
        "{id} seed={seed} steps={} episodes={} seconds={:.3}",
        env.steps,
        env.episodes,
        started.elapsed().as_secs_f64()
    );
}

fn maze(
    name: &str,
    geometry: &'static MazeGeometry,
    cap: u32,
    pool: Option<Vec<(usize, usize)>>,
) -> MazeTask<Coordinates, 2> {
    let cells = pool.unwrap_or_else(|| {
        MazeTopology::new(geometry)
            .unwrap()
            .walkable_cells()
            .to_vec()
    });
    MazeTask::new(name.to_owned(), geometry, cells, cap, "coordinates")
}

fn main() {
    let mut out = None;
    let mut steps = 20_000;
    let mut seeds: Vec<u64> = (42..=61).collect();
    let mut only = None;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let value = args.next().expect("option value");
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value)),
            "--steps" => steps = value.parse().unwrap(),
            "--seeds" => {
                seeds = value
                    .split(',')
                    .map(|value| value.parse().unwrap())
                    .collect()
            }
            "--configuration" => only = Some(value),
            _ => panic!("unknown option {flag}"),
        }
    }
    assert!(steps > 0 && !seeds.is_empty());
    let mut writer = BufWriter::new(
        File::create(output::output_path(&out.expect("--out is required"))).unwrap(),
    );
    let ids = [
        "maze4_c5_full",
        "maze4_c5_p4",
        "maze6_c10_full",
        "maze6_c10_p4",
        "handeye4_c50_full",
        "handeye4_c50_p4",
        "taxi_c200_full",
        "bitflip8_c8_full",
        "handeye5_c50_full",
        "maze7_c10_full",
    ];
    assert!(only
        .as_ref()
        .is_none_or(|name| ids.contains(&name.as_str())));
    for id in ids {
        if only.as_ref().is_some_and(|name| name != id) {
            continue;
        }
        for &seed in &seeds {
            match id {
                "maze4_c5_full" => collect::<_, 8, 2, 10>(
                    &maze("maze4", &MAZE4, 5, None),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "maze4_c5_p4" => collect::<_, 8, 2, 10>(
                    &maze(
                        "maze4",
                        &MAZE4,
                        5,
                        Some(vec![(2, 5), (5, 5), (6, 3), (6, 4)]),
                    ),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "maze6_c10_full" => collect::<_, 8, 2, 10>(
                    &maze("maze6", &MAZE6, 10, None),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "maze6_c10_p4" => collect::<_, 8, 2, 10>(
                    &maze(
                        "maze6",
                        &MAZE6,
                        10,
                        Some(vec![(3, 1), (3, 6), (4, 3), (7, 7)]),
                    ),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "maze7_c10_full" => collect::<_, 8, 2, 10>(
                    &maze("maze7", &MAZE7, 10, None),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "handeye4_c50_full" => {
                    let template = HandEye4::new(50, Box::new(ChaChaRandomSource::from_seed(0)));
                    collect::<_, 17, 2, 19>(
                        &HandEyeTask::<4, 17>::new(
                            "handeye4".to_owned(),
                            50,
                            template.goal_pool().to_vec(),
                        ),
                        id,
                        seed,
                        steps,
                        &mut writer,
                    );
                }
                "handeye4_c50_p4" => collect::<_, 17, 2, 19>(
                    &HandEyeTask::<4, 17>::new(
                        "handeye4".to_owned(),
                        50,
                        [(0, 0), (2, 0), (3, 2), (1, 3)]
                            .into_iter()
                            .map(position_goal)
                            .collect(),
                    ),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "handeye5_c50_full" => {
                    let template = HandEye5::new(50, Box::new(ChaChaRandomSource::from_seed(0)));
                    collect::<_, 26, 2, 28>(
                        &HandEyeTask::<5, 26>::new(
                            "handeye5".to_owned(),
                            50,
                            template.goal_pool().to_vec(),
                        ),
                        id,
                        seed,
                        steps,
                        &mut writer,
                    );
                }
                "taxi_c200_full" => collect::<_, 3, 1, 4>(
                    &TaxiTask::new(200, (0..4).map(passenger_goal).collect()),
                    id,
                    seed,
                    steps,
                    &mut writer,
                ),
                "bitflip8_c8_full" => {
                    let pool = BitFlipping::<8>::new(Box::new(ChaChaRandomSource::from_seed(0)))
                        .goal_pool()
                        .collect();
                    collect::<_, 8, 8, 16>(
                        &BitTask::<8>::new(8, pool),
                        id,
                        seed,
                        steps,
                        &mut writer,
                    );
                }
                _ => unreachable!("configuration list"),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_covers_every_strategy_rule_and_filter_once() {
        let configurations = configurations();
        assert_eq!(configurations.len(), 24);
        for (index, configuration) in configurations.iter().enumerate() {
            assert!(!configurations[..index].contains(configuration));
            assert_eq!(configuration.relabeled_proportion, 1.0);
            assert_eq!(configuration.truncation, TruncationMode::Bootstrap);
        }
    }

    #[test]
    fn composition_aggregation_weights_steps_instead_of_episodes() {
        let mut total = ExpectedComposition::default();
        total.add_weighted(
            ExpectedComposition {
                done: 1.0,
                mean_reward: 1000.0,
                ..ExpectedComposition::default()
            },
            1.0,
        );
        total.add_weighted(ExpectedComposition::default(), 9.0);
        let mut average = ExpectedComposition::default();
        average.add_weighted(total, 0.1);
        assert_eq!(average.done, 0.1);
        assert_eq!(average.mean_reward, 100.0);
    }
}
