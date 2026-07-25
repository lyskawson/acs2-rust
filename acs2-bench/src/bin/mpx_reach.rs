use std::mem::size_of;
use std::time::{Duration, Instant};

use acs2_bench::{parse_u_max_mode, parse_variant, resolve_u_max, variant_label, UMaxMode};
use acs2_core::action_selection::EpsilonGreedy;
use acs2_core::agent::Agent;
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::config::{AlpGenVariant, Configuration};
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_envs::multiplexer::{evaluate_knowledge, Multiplexer};

const EXPLORE_EPSILON: f64 = 0.8;
const SAMPLE_INPUTS: usize = 50_000;
const SAMPLE_SEED: u64 = 0x6D70_7831;
const RSS_CAP_BYTES: u64 = 5_600_000_000;
const DEFAULT_TIME_CAP_SECS: u64 = 600;
const TIME_CHECK_BATCH: u32 = 500;
const DEFAULT_KNOWLEDGE_EVAL_INTERVAL: u64 = 6_000;
const TRIALS_CAP_MULTIPLIER: u128 = 10;
const TRIALS_BASE_AT_K6: u128 = 20_000;

fn peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) };
    if status != 0 {
        return 0;
    }
    usage.ru_maxrss as u64
}

fn trials_cap_for(size: usize) -> u64 {
    let exponent = (size - 6) as u32;
    let estimate = TRIALS_BASE_AT_K6.checked_shl(exponent);
    match estimate {
        Some(value) => {
            let capped = value.saturating_mul(TRIALS_CAP_MULTIPLIER);
            u64::try_from(capped).unwrap_or(u64::MAX)
        }
        None => u64::MAX,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Verdict {
    Success,
    TrialsLimited,
    MemoryLimited,
    TimeLimited,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Verdict::Success => "SUCCESS",
            Verdict::TrialsLimited => "TRIALS-LIMITED",
            Verdict::MemoryLimited => "MEMORY-LIMITED",
            Verdict::TimeLimited => "TIME-LIMITED",
        }
    }
}

struct ReachOutcome {
    verdict: Verdict,
    trials_used: u64,
    final_knowledge: f64,
    reliable_count: usize,
    mean_reliable_specificity: f64,
    peak_macro_population: usize,
    peak_rss_bytes: u64,
    wall_seconds: f64,
}

#[derive(Clone, Copy)]
struct GenConfig {
    do_ga: bool,
    u_max: u32,
    alp_gen_variant: AlpGenVariant,
}

fn run_reach_repeat<const N: usize>(
    size: usize,
    trials_cap: u64,
    time_cap: Duration,
    seed: u64,
    gen: GenConfig,
    eval_interval: u64,
    log_trajectory: bool,
) -> ReachOutcome {
    let mut config = Configuration::mpx();
    config.epsilon = EXPLORE_EPSILON;
    config.do_ga = gen.do_ga;
    config.u_max = gen.u_max;
    config.alp_gen_variant = gen.alp_gen_variant;

    let mut env = Multiplexer::<N>::new(Box::new(ChaChaRandomSource::from_seed(seed)));
    let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
    let selector = EpsilonGreedy {
        number_of_possible_actions: Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
        epsilon: EXPLORE_EPSILON,
    };
    let bootstrap = MaxFitnessBootstrap;
    let theta_r = agent.config().theta_r;

    let start = Instant::now();
    let mut time: u64 = 0;
    let mut trials_used: u64 = 0;
    let mut peak_macro_population = 0usize;
    let mut peak_rss = peak_rss_bytes();
    let mut final_knowledge = 0.0;
    let mut trials_since_eval: u64 = 0;

    let verdict = loop {
        for _ in 0..TIME_CHECK_BATCH {
            let metrics = agent.run_explore_trial(&mut env, &selector, &bootstrap, time);
            time += metrics.steps as u64;
            trials_used += 1;
            trials_since_eval += 1;
        }

        peak_macro_population = peak_macro_population.max(agent.population().len());
        peak_rss = peak_rss.max(peak_rss_bytes());

        if peak_rss > RSS_CAP_BYTES {
            break Verdict::MemoryLimited;
        }
        if start.elapsed() > time_cap {
            break Verdict::TimeLimited;
        }
        if trials_since_eval >= eval_interval {
            trials_since_eval = 0;
            final_knowledge = evaluate_knowledge(agent.population(), theta_r, SAMPLE_INPUTS, SAMPLE_SEED);
            if log_trajectory {
                let (reliable, spec_sum) = agent
                    .population()
                    .iter()
                    .filter(|classifier| classifier.is_reliable(theta_r))
                    .fold((0usize, 0.0f64), |(count, sum), classifier| {
                        (count + 1, sum + classifier.condition.specificity() as f64)
                    });
                let spec = if reliable == 0 { 0.0 } else { spec_sum / reliable as f64 };
                println!(
                    "  mpx-{size} traj: trials={trials_used} wall={:.0}s knowledge={final_knowledge:.4} reliable={reliable} spec={spec:.2} pop={}",
                    start.elapsed().as_secs_f64(),
                    agent.population().len(),
                );
            }
            if final_knowledge >= 1.0 {
                break Verdict::Success;
            }
        }
        if trials_used >= trials_cap {
            break Verdict::TrialsLimited;
        }
    };

    let reliable_specificities: Vec<f64> = agent
        .population()
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .map(|classifier| classifier.condition.specificity() as f64)
        .collect();
    let reliable_count = reliable_specificities.len();
    let mean_reliable_specificity = if reliable_count == 0 {
        0.0
    } else {
        reliable_specificities.iter().sum::<f64>() / reliable_count as f64
    };

    ReachOutcome {
        verdict,
        trials_used,
        final_knowledge,
        reliable_count,
        mean_reliable_specificity,
        peak_macro_population,
        peak_rss_bytes: peak_rss,
        wall_seconds: start.elapsed().as_secs_f64(),
    }
}

fn run_reach_dispatch(
    size: usize,
    trials_cap: u64,
    time_cap: Duration,
    seed: u64,
    gen: GenConfig,
    eval_interval: u64,
    log_trajectory: bool,
) -> ReachOutcome {
    match size {
        37 => run_reach_repeat::<38>(size, trials_cap, time_cap, seed, gen, eval_interval, log_trajectory),
        70 => run_reach_repeat::<71>(size, trials_cap, time_cap, seed, gen, eval_interval, log_trajectory),
        135 => run_reach_repeat::<136>(size, trials_cap, time_cap, seed, gen, eval_interval, log_trajectory),
        20 => run_reach_repeat::<21>(size, trials_cap, time_cap, seed, gen, eval_interval, log_trajectory),
        other => panic!("reach not configured for {other}-bit multiplexer"),
    }
}

fn component_memory<const N: usize>(size: usize) {
    let condition = size_of::<Condition<N>>();
    let effect = size_of::<Effect<N>>();
    let mark = size_of::<Mark<N>>();
    let classifier = size_of::<Classifier<N>>();
    let pop_threshold = RSS_CAP_BYTES / classifier as u64;
    println!(
        "  mem mpx-{size} (N={N}): condition={condition}B effect={effect}B mark={mark}B (stack) \
         classifier={classifier}B  mark/classifier={:.1}%  rss-cap pop-threshold={pop_threshold}",
        100.0 * mark as f64 / classifier as f64,
    );
}

fn report_component_memory(size: usize) {
    match size {
        6 => component_memory::<7>(size),
        11 => component_memory::<12>(size),
        20 => component_memory::<21>(size),
        37 => component_memory::<38>(size),
        70 => component_memory::<71>(size),
        135 => component_memory::<136>(size),
        other => panic!("no memory layout for {other}"),
    }
}

struct Options {
    sizes: Vec<usize>,
    n_exp: u32,
    seed: u64,
    time_cap_secs: u64,
    do_ga: bool,
    u_max_mode: UMaxMode,
    alp_gen_variant: AlpGenVariant,
    eval_interval: u64,
    log_trajectory: bool,
}

impl Options {
    fn parse() -> Self {
        let mut options = Options {
            sizes: vec![37, 70, 135],
            n_exp: 3,
            seed: 42,
            time_cap_secs: DEFAULT_TIME_CAP_SECS,
            do_ga: true,
            u_max_mode: UMaxMode::Default,
            alp_gen_variant: AlpGenVariant::Pyalcs,
            eval_interval: DEFAULT_KNOWLEDGE_EVAL_INTERVAL,
            log_trajectory: false,
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
                "--time-cap-secs" => options.time_cap_secs = args.next().unwrap().parse().unwrap(),
                "--do-ga" => {
                    options.do_ga = args
                        .next()
                        .expect("--do-ga needs true|false")
                        .parse()
                        .expect("--do-ga must be true or false")
                }
                "--u-max" => {
                    options.u_max_mode =
                        parse_u_max_mode(&args.next().expect("--u-max needs a value"))
                }
                "--alp-gen-variant" => {
                    options.alp_gen_variant =
                        parse_variant(&args.next().expect("--alp-gen-variant needs a value"))
                }
                "--eval-interval" => {
                    options.eval_interval = args.next().unwrap().parse().unwrap()
                }
                "--log-trajectory" => options.log_trajectory = true,
                other => panic!("unknown flag {other}"),
            }
        }
        options
    }
}

fn main() {
    let options = Options::parse();
    println!(
        "acs2-bench mpx-reach: sizes={:?} n_exp={} seed={} rss_cap={}GB time_cap={}s do_ga={} alp_gen_variant={}",
        options.sizes,
        options.n_exp,
        options.seed,
        RSS_CAP_BYTES as f64 / 1e9,
        options.time_cap_secs,
        options.do_ga,
        variant_label(options.alp_gen_variant),
    );

    for &size in &options.sizes {
        let trials_cap = trials_cap_for(size);
        report_component_memory(size);
        let u_max = resolve_u_max(
            options.u_max_mode,
            Configuration::mpx().u_max,
            size,
            options.alp_gen_variant,
        );
        let gen = GenConfig {
            do_ga: options.do_ga,
            u_max,
            alp_gen_variant: options.alp_gen_variant,
        };
        println!("  mpx-{size} trials_cap={trials_cap} (= 20000*2^(k-6)*10, clamped to u64::MAX) u_max={u_max}");

        let mut verdicts: Vec<Verdict> = Vec::new();
        let time_cap = Duration::from_secs(options.time_cap_secs);
        for repeat in 0..options.n_exp {
            let outcome = run_reach_dispatch(
                size,
                trials_cap,
                time_cap,
                options.seed + repeat as u64,
                gen,
                options.eval_interval,
                options.log_trajectory,
            );
            verdicts.push(outcome.verdict);
            println!(
                "  mpx-{size} repeat {repeat}: {} trials={} knowledge={:.4} reliable={} spec={:.2}/{} peak_macro={} peak_rss={:.2}GB wall={:.1}s",
                outcome.verdict.label(),
                outcome.trials_used,
                outcome.final_knowledge,
                outcome.reliable_count,
                outcome.mean_reliable_specificity,
                size + 1,
                outcome.peak_macro_population,
                outcome.peak_rss_bytes as f64 / 1e9,
                outcome.wall_seconds,
            );
        }
        let agree = verdicts.iter().all(|&v| v == verdicts[0]);
        println!(
            "  mpx-{size} verdict agreement across {} repeats: {}",
            options.n_exp,
            if agree { "ALL AGREE" } else { "DISAGREE" },
        );
    }
}
