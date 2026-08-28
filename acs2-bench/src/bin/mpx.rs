use std::time::Instant;

use acs2_bench::{parse_u_max_mode, parse_variant, resolve_u_max, variant_label, UMaxMode};
use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::config::{AlpGenVariant, Configuration};
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::trial::LearningAgent;
use acs2_envs::multiplexer::{evaluate_knowledge, Multiplexer};

const EXPLORE_EPSILON: f64 = 0.8;
const SAMPLE_INPUTS: u64 = 50_000;
const SAMPLE_SEED: u64 = 0x6D70_7831;

struct Options {
    sizes: Vec<usize>,
    n_exp: u32,
    seed: u64,
    explore_trials_override: Option<u32>,
    exploit_trials: u32,
    exploit_phases: u32,
    skip_knowledge: bool,
    do_ga: bool,
    u_max_mode: UMaxMode,
    alp_gen_variant: AlpGenVariant,
    out: String,
}

impl Options {
    fn parse() -> Self {
        let mut options = Options {
            sizes: vec![6, 11],
            n_exp: 10,
            seed: 42,
            explore_trials_override: None,
            exploit_trials: 200,
            exploit_phases: 3,
            skip_knowledge: false,
            do_ga: false,
            u_max_mode: UMaxMode::Default,
            alp_gen_variant: AlpGenVariant::Pyalcs,
            out: "reports/mpx_rust.csv".to_string(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--sizes" => {
                    options.sizes = args
                        .next()
                        .expect("--sizes needs a value")
                        .split(',')
                        .map(|item| item.parse().expect("size must be an integer"))
                        .collect()
                }
                "--n-exp" => options.n_exp = args.next().unwrap().parse().unwrap(),
                "--seed" => options.seed = args.next().unwrap().parse().unwrap(),
                "--explore-trials" => {
                    options.explore_trials_override = Some(args.next().unwrap().parse().unwrap())
                }
                "--exploit-trials" => options.exploit_trials = args.next().unwrap().parse().unwrap(),
                "--exploit-phases" => options.exploit_phases = args.next().unwrap().parse().unwrap(),
                "--skip-knowledge" => options.skip_knowledge = true,
                "--do-ga" => options.do_ga = true,
                "--u-max" => {
                    options.u_max_mode =
                        parse_u_max_mode(&args.next().expect("--u-max needs a value"))
                }
                "--alp-gen-variant" => {
                    options.alp_gen_variant =
                        parse_variant(&args.next().expect("--alp-gen-variant needs a value"))
                }
                "--out" => options.out = args.next().unwrap(),
                other => panic!("unknown flag {other}"),
            }
        }
        options
    }
}

fn explore_budget_for(size: usize) -> u32 {
    match size {
        6 => 20_000,
        11 => 200_000,
        20 => 300_000,
        37 => 150_000,
        70 => 400_000,
        135 => 1_000_000,
        _ => panic!("no explore budget configured for {size}-bit multiplexer"),
    }
}

struct RepeatResult {
    knowledge: f64,
    macro_population: usize,
    reliable: usize,
    mean_reliable_specificity: f64,
    explore_seconds: f64,
    exploit_seconds: f64,
}

fn run_repeat<const N: usize>(explore_trials: u32, options: &Options, seed: u64) -> RepeatResult {
    let mut config = Configuration::mpx();
    config.epsilon = EXPLORE_EPSILON;
    config.do_ga = options.do_ga;
    config.alp_gen_variant = options.alp_gen_variant;
    config.u_max = resolve_u_max(options.u_max_mode, config.u_max, N - 1, options.alp_gen_variant);

    let mut env = Multiplexer::<N>::new(Box::new(ChaChaRandomSource::from_seed(seed)));
    let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
    let selector = EpsilonGreedy {
        number_of_possible_actions: Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
        epsilon: EXPLORE_EPSILON,
    };
    let bootstrap = MaxFitnessBootstrap;
    let mut time: u64 = 0;

    let explore_start = Instant::now();
    for _ in 0..explore_trials {
        let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
        time += metrics.steps as u64;
    }
    let explore_seconds = explore_start.elapsed().as_secs_f64();

    let exploit_start = Instant::now();
    for _ in 0..(options.exploit_trials * options.exploit_phases) {
        let metrics = agent.run_exploit_trial(&mut env, &bootstrap, time);
        time += metrics.steps as u64;
    }
    let exploit_seconds = exploit_start.elapsed().as_secs_f64();

    let theta_r = agent.config().theta_r;
    let reliable_specificities: Vec<f64> = agent
        .population()
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .map(|classifier| classifier.condition.specificity() as f64)
        .collect();
    let mean_reliable_specificity = if reliable_specificities.is_empty() {
        0.0
    } else {
        mean(&reliable_specificities)
    };

    let knowledge = if options.skip_knowledge {
        f64::NAN
    } else {
        evaluate_knowledge(agent.population(), theta_r, SAMPLE_INPUTS as usize, SAMPLE_SEED)
    };

    RepeatResult {
        knowledge,
        macro_population: agent.population().len(),
        reliable: agent.population().reliable_count(theta_r),
        mean_reliable_specificity,
        explore_seconds,
        exploit_seconds,
    }
}

fn run_size_dispatch(size: usize, explore_trials: u32, options: &Options, seed: u64) -> RepeatResult {
    match size {
        6 => run_repeat::<7>(explore_trials, options, seed),
        11 => run_repeat::<12>(explore_trials, options, seed),
        20 => run_repeat::<21>(explore_trials, options, seed),
        37 => run_repeat::<38>(explore_trials, options, seed),
        70 => run_repeat::<71>(explore_trials, options, seed),
        135 => run_repeat::<136>(explore_trials, options, seed),
        other => panic!("unsupported multiplexer size {other}"),
    }
}

struct SizeSummary {
    size: usize,
    u_max: u32,
    explore_trials: u32,
    knowledge_mean: f64,
    knowledge_min: f64,
    reached_full_knowledge: u32,
    macro_population_mean: f64,
    macro_population_std: f64,
    reliable_mean: f64,
    reliable_std: f64,
    mean_reliable_specificity: f64,
    specificity_std: f64,
    explore_seconds_total: f64,
    exploit_seconds_total: f64,
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn population_std(values: &[f64]) -> f64 {
    let average = mean(values);
    let variance = values.iter().map(|value| (value - average).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt()
}

fn run_size(size: usize, options: &Options) -> SizeSummary {
    let explore_trials = options.explore_trials_override.unwrap_or_else(|| explore_budget_for(size));
    let mut knowledge_values: Vec<f64> = Vec::new();
    let mut macro_values: Vec<f64> = Vec::new();
    let mut reliable_values: Vec<f64> = Vec::new();
    let mut specificity_values: Vec<f64> = Vec::new();
    let mut explore_total = 0.0;
    let mut exploit_total = 0.0;

    for repeat in 0..options.n_exp {
        let result = run_size_dispatch(size, explore_trials, options, options.seed + repeat as u64);
        knowledge_values.push(result.knowledge);
        macro_values.push(result.macro_population as f64);
        reliable_values.push(result.reliable as f64);
        specificity_values.push(result.mean_reliable_specificity);
        explore_total += result.explore_seconds;
        exploit_total += result.exploit_seconds;
    }

    let reached_full_knowledge = knowledge_values.iter().filter(|&&k| k >= 1.0).count() as u32;

    SizeSummary {
        size,
        u_max: resolve_u_max(options.u_max_mode, Configuration::mpx().u_max, size, options.alp_gen_variant),
        explore_trials,
        knowledge_mean: mean(&knowledge_values),
        knowledge_min: knowledge_values.iter().copied().fold(f64::INFINITY, f64::min),
        reached_full_knowledge,
        macro_population_mean: mean(&macro_values),
        macro_population_std: population_std(&macro_values),
        reliable_mean: mean(&reliable_values),
        reliable_std: population_std(&reliable_values),
        mean_reliable_specificity: mean(&specificity_values),
        specificity_std: population_std(&specificity_values),
        explore_seconds_total: explore_total,
        exploit_seconds_total: exploit_total,
    }
}

fn csv_header() -> String {
    "size,explore_trials,n_exp,knowledge_mean,knowledge_min,reached_full,macro_pop_mean,macro_pop_std,\
     reliable_mean,reliable_std,reliable_spec_mean,reliable_spec_std,\
     explore_time_total_s,exploit_time_total_s,total_time_s,u_max,alp_gen_variant"
        .to_string()
}

fn csv_row(summary: &SizeSummary, n_exp: u32, variant: &str) -> String {
    format!(
        "{},{},{},{:.6},{:.6},{}/{},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{},{}",
        summary.size,
        summary.explore_trials,
        n_exp,
        summary.knowledge_mean,
        summary.knowledge_min,
        summary.reached_full_knowledge,
        n_exp,
        summary.macro_population_mean,
        summary.macro_population_std,
        summary.reliable_mean,
        summary.reliable_std,
        summary.mean_reliable_specificity,
        summary.specificity_std,
        summary.explore_seconds_total,
        summary.exploit_seconds_total,
        summary.explore_seconds_total + summary.exploit_seconds_total,
        summary.u_max,
        variant,
    )
}

fn main() {
    let options = Options::parse();
    let mut lines = vec![csv_header()];

    println!(
        "acs2-bench mpx: sizes={:?} n_exp={} seed={} exploit={}x{} do_ga={} alp_gen_variant={}",
        options.sizes,
        options.n_exp,
        options.seed,
        options.exploit_trials,
        options.exploit_phases,
        options.do_ga,
        variant_label(options.alp_gen_variant),
    );

    for &size in &options.sizes {
        let summary = run_size(size, &options);
        println!(
            "mpx-{:<4} u_max={:<7} explore={:<8} knowledge={:.4} (min {:.4}, {}/{} at 1.0) macro={:.1}±{:.1} reliable={:.1}±{:.1} spec={:.2}±{:.2}/{} time={:.3}s",
            summary.size,
            summary.u_max,
            summary.explore_trials,
            summary.knowledge_mean,
            summary.knowledge_min,
            summary.reached_full_knowledge,
            options.n_exp,
            summary.macro_population_mean,
            summary.macro_population_std,
            summary.reliable_mean,
            summary.reliable_std,
            summary.mean_reliable_specificity,
            summary.specificity_std,
            summary.size + 1,
            summary.explore_seconds_total + summary.exploit_seconds_total,
        );
        lines.push(csv_row(&summary, options.n_exp, variant_label(options.alp_gen_variant)));
    }

    std::fs::write(&options.out, lines.join("\n") + "\n").expect("write csv");
    println!("wrote {}", options.out);
}
