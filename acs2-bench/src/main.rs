use std::time::Instant;

use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::trial::LearningAgent;
use acs2_core::config::Configuration;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::maze::{Maze, MAZE_PERCEPTION_LEN};
use acs2_envs::maze_data::{geometry_by_id, MazeGeometry, MAZE_GEOMETRIES};

const EXPLORE_EPSILON: f64 = 0.8;

struct Options {
    mazes: Vec<String>,
    n_exp: u32,
    seed: u64,
    do_ga: bool,
    explore_trials: u32,
    exploit_trials: u32,
    exploit_phases: u32,
    out: String,
}

impl Options {
    fn parse() -> Self {
        let mut options = Options {
            mazes: MAZE_GEOMETRIES.iter().map(|g| g.id.to_string()).collect(),
            n_exp: 10,
            seed: 42,
            do_ga: false,
            explore_trials: 500,
            exploit_trials: 200,
            exploit_phases: 3,
            out: "reports/bench_rust.csv".to_string(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--mazes" => {
                    options.mazes = args
                        .next()
                        .expect("--mazes needs a value")
                        .split(',')
                        .map(|item| item.to_string())
                        .collect()
                }
                "--n-exp" => options.n_exp = args.next().unwrap().parse().unwrap(),
                "--seed" => options.seed = args.next().unwrap().parse().unwrap(),
                "--do-ga" => options.do_ga = true,
                "--explore-trials" => options.explore_trials = args.next().unwrap().parse().unwrap(),
                "--exploit-trials" => options.exploit_trials = args.next().unwrap().parse().unwrap(),
                "--exploit-phases" => options.exploit_phases = args.next().unwrap().parse().unwrap(),
                "--out" => options.out = args.next().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        options
    }
}

struct RepeatResult {
    final_window_mean_steps: f64,
    macro_population: usize,
    numerosity: u32,
    reliable: usize,
    explore_seconds: f64,
    exploit_seconds: f64,
}

struct MazeSummary {
    maze: String,
    n_exp: u32,
    exploit_steps_mean: f64,
    exploit_steps_std: f64,
    macro_population_mean: f64,
    numerosity_mean: f64,
    reliable_mean: f64,
    explore_seconds_total: f64,
    exploit_seconds_total: f64,
    total_seconds: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn population_std(values: &[f64]) -> f64 {
    let average = mean(values);
    let variance = values.iter().map(|v| (v - average).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn run_repeat(geometry: &MazeGeometry, options: &Options, seed: u64) -> RepeatResult {
    let mut config = Configuration::default_protocol();
    config.do_ga = options.do_ga;
    config.epsilon = EXPLORE_EPSILON;

    let mut env = Maze::from_geometry(geometry, Box::new(ChaChaRandomSource::from_seed(seed)));
    let mut agent =
        Agent::<MAZE_PERCEPTION_LEN, _>::new(config, ChaChaRandomSource::from_seed(seed));
    let selector = EpsilonGreedy {
        number_of_possible_actions: agent.config().number_of_possible_actions,
        epsilon: EXPLORE_EPSILON,
    };
    let bootstrap = MaxFitnessBootstrap;

    let mut time: u64 = 0;

    let explore_start = Instant::now();
    for _ in 0..options.explore_trials {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }
    let explore_seconds = explore_start.elapsed().as_secs_f64();

    let mut final_window: Vec<f64> = Vec::new();
    let exploit_start = Instant::now();
    for phase in 0..options.exploit_phases {
        let mut phase_steps: Vec<f64> = Vec::with_capacity(options.exploit_trials as usize);
        for _ in 0..options.exploit_trials {
            let metrics = agent.run_exploit_trial(&mut env, &bootstrap, time);
            time += metrics.steps as u64;
            phase_steps.push(metrics.steps as f64);
        }
        if phase == options.exploit_phases - 1 {
            final_window = phase_steps;
        }
    }
    let exploit_seconds = exploit_start.elapsed().as_secs_f64();

    let population = agent.population();
    RepeatResult {
        final_window_mean_steps: mean(&final_window),
        macro_population: population.len(),
        numerosity: population.numerosity(),
        reliable: population.reliable_count(agent.config().theta_r),
        explore_seconds,
        exploit_seconds,
    }
}

fn run_maze(maze_id: &str, options: &Options) -> MazeSummary {
    let geometry = geometry_by_id(maze_id).unwrap_or_else(|| panic!("unknown maze {maze_id}"));

    let mut repeat_means: Vec<f64> = Vec::new();
    let mut macro_values: Vec<f64> = Vec::new();
    let mut numerosity_values: Vec<f64> = Vec::new();
    let mut reliable_values: Vec<f64> = Vec::new();
    let mut explore_total = 0.0;
    let mut exploit_total = 0.0;

    for repeat in 0..options.n_exp {
        let result = run_repeat(geometry, options, options.seed + repeat as u64);
        repeat_means.push(result.final_window_mean_steps);
        macro_values.push(result.macro_population as f64);
        numerosity_values.push(result.numerosity as f64);
        reliable_values.push(result.reliable as f64);
        explore_total += result.explore_seconds;
        exploit_total += result.exploit_seconds;
    }

    MazeSummary {
        maze: maze_id.to_string(),
        n_exp: options.n_exp,
        exploit_steps_mean: mean(&repeat_means),
        exploit_steps_std: population_std(&repeat_means),
        macro_population_mean: mean(&macro_values),
        numerosity_mean: mean(&numerosity_values),
        reliable_mean: mean(&reliable_values),
        explore_seconds_total: explore_total,
        exploit_seconds_total: exploit_total,
        total_seconds: explore_total + exploit_total,
    }
}

fn csv_header() -> String {
    "maze,n_exp,exploit_steps_mean,exploit_steps_std,macro_pop_mean,numerosity_mean,\
     reliable_mean,explore_time_total_s,exploit_time_total_s,total_time_s"
        .to_string()
}

fn csv_row(summary: &MazeSummary) -> String {
    format!(
        "{},{},{:.4},{:.4},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4}",
        summary.maze,
        summary.n_exp,
        summary.exploit_steps_mean,
        summary.exploit_steps_std,
        summary.macro_population_mean,
        summary.numerosity_mean,
        summary.reliable_mean,
        summary.explore_seconds_total,
        summary.exploit_seconds_total,
        summary.total_seconds,
    )
}

fn main() {
    let options = Options::parse();
    let mut lines = vec![csv_header()];

    println!(
        "acs2-bench: n_exp={} seed={} do_ga={} explore={} exploit={}x{}",
        options.n_exp,
        options.seed,
        options.do_ga,
        options.explore_trials,
        options.exploit_trials,
        options.exploit_phases,
    );

    for maze_id in &options.mazes {
        let summary = run_maze(maze_id, &options);
        println!(
            "{:<12} exploit_steps={:.3}±{:.3} macro={:.1} reliable={:.1} total_time={:.3}s",
            summary.maze,
            summary.exploit_steps_mean,
            summary.exploit_steps_std,
            summary.macro_population_mean,
            summary.reliable_mean,
            summary.total_seconds,
        );
        lines.push(csv_row(&summary));
    }

    std::fs::write(&options.out, lines.join("\n") + "\n").expect("write csv");
    println!("wrote {}", options.out);
}
