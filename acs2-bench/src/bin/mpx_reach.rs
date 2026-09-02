use std::mem::size_of;
use std::time::{Duration, Instant};

use acs2_bench::{
    parse_u_max_mode, parse_variant, resolve_u_max, variant_label, AgentChoice, AgentOptions,
    UMaxMode,
};
use acs2_core::acs2er::Acs2ErAgent;
use acs2_core::action_selection::{ActionSelector, BestAction, EpsilonGreedy};
use acs2_core::agent::Agent;
use acs2_core::classifier::Classifier;
use acs2_core::condition::Condition;
use acs2_core::config::{AlpGenVariant, Configuration};
use acs2_core::effect::Effect;
use acs2_core::mark::Mark;
use acs2_core::population::Population;
use acs2_core::rl::MaxFitnessBootstrap;
use acs2_core::rng::ChaChaRandomSource;
use acs2_core::trial::LearningAgent;
use acs2_envs::multiplexer::{
    control_bits_for, evaluate_knowledge, parse_encoding, sampled_transitions, transition_is_correct,
    Encoding, Multiplexer,
};

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

/// Task performance: how often greedy action choice answers correctly.
///
/// `knowledge` asks whether the population anticipates every transition, including
/// the null ones a wrong answer produces. Choosing correctly needs only the
/// change-anticipating side, so accuracy can be high while knowledge is capped --
/// and accuracy is what the multiplexer literature reports, so it is the number
/// that makes these runs comparable to published results.
fn answer_accuracy<const N: usize>(
    population: &Population<N>,
    number_of_possible_actions: usize,
    encoding: Encoding,
) -> f64 {
    let selector = BestAction {
        number_of_possible_actions,
    };
    // a private RNG: evaluation must not disturb the agent's stream
    let mut rng = ChaChaRandomSource::from_seed(SAMPLE_SEED);

    let mut asked = 0usize;
    let mut correct = 0usize;
    for transition in sampled_transitions::<N>(SAMPLE_INPUTS, SAMPLE_SEED, encoding) {
        if transition.action != 0 {
            continue;
        }
        let match_set = population.form_match_set(&transition.p0);
        let chosen = selector.select(population, &match_set, &mut rng);
        // we only walk the action-0 transitions, so correctness of that
        // transition tells us directly which answer was the right one
        let right_answer = if transition_is_correct(&transition) { 0 } else { 1 };
        asked += 1;
        if chosen == right_answer {
            correct += 1;
        }
    }

    if asked == 0 {
        0.0
    } else {
        correct as f64 / asked as f64
    }
}

struct QuadrantDetail {
    covered_by_any: [usize; 4],
    best_quality: [f64; 4],
    total: [usize; 4],
}

impl QuadrantDetail {
    fn fraction(&self, cell: usize) -> f64 {
        if self.total[cell] == 0 {
            0.0
        } else {
            self.covered_by_any[cell] as f64 / self.total[cell] as f64
        }
    }
}

fn quadrant_detail<const N: usize>(population: &Population<N>, encoding: Encoding) -> QuadrantDetail {
    let mut detail = QuadrantDetail {
        covered_by_any: [0; 4],
        best_quality: [0.0; 4],
        total: [0; 4],
    };

    for transition in sampled_transitions::<N>(SAMPLE_INPUTS, SAMPLE_SEED, encoding) {
        let correct = transition_is_correct(&transition);
        let cell = (transition.action << 1) | usize::from(correct);
        detail.total[cell] += 1;

        let mut predicted = false;
        for classifier in population.iter() {
            if classifier.action != Some(transition.action)
                || !classifier.does_match(&transition.p0)
                || !classifier.does_anticipate_correctly(&transition.p0, &transition.p1)
            {
                continue;
            }
            predicted = true;
            if classifier.q > detail.best_quality[cell] {
                detail.best_quality[cell] = classifier.q;
            }
        }
        if predicted {
            detail.covered_by_any[cell] += 1;
        }
    }

    detail
}

struct KnowledgeBreakdown {
    covered: [usize; 4],
    total: [usize; 4],
    matched_but_wrong: usize,
}

impl KnowledgeBreakdown {
    fn fraction(&self, cell: usize) -> f64 {
        if self.total[cell] == 0 {
            0.0
        } else {
            self.covered[cell] as f64 / self.total[cell] as f64
        }
    }

    fn overall(&self) -> f64 {
        let covered: usize = self.covered.iter().sum();
        let total: usize = self.total.iter().sum();
        if total == 0 {
            0.0
        } else {
            covered as f64 / total as f64
        }
    }
}

fn knowledge_breakdown<const N: usize>(
    population: &Population<N>,
    theta_r: f64,
    encoding: Encoding,
) -> KnowledgeBreakdown {
    let reliable: Vec<&Classifier<N>> = population
        .iter()
        .filter(|classifier| classifier.is_reliable(theta_r))
        .collect();

    let mut breakdown = KnowledgeBreakdown {
        covered: [0; 4],
        total: [0; 4],
        matched_but_wrong: 0,
    };

    for transition in sampled_transitions::<N>(SAMPLE_INPUTS, SAMPLE_SEED, encoding) {
        let correct = transition_is_correct(&transition);
        let cell = (transition.action << 1) | usize::from(correct);
        breakdown.total[cell] += 1;

        let mut matched = false;
        let mut predicted = false;
        for classifier in &reliable {
            if classifier.action != Some(transition.action) || !classifier.does_match(&transition.p0)
            {
                continue;
            }
            matched = true;
            if classifier.does_anticipate_correctly(&transition.p0, &transition.p1) {
                predicted = true;
                break;
            }
        }
        if predicted {
            breakdown.covered[cell] += 1;
        } else if matched {
            breakdown.matched_but_wrong += 1;
        }
    }

    breakdown
}

struct PopulationDiagnostics {
    micro_size: u64,
    specificity_mean: f64,
    specificity_max: usize,
    quality_mean: f64,
    quality_max: f64,
    above_half_quality: usize,
    marked_fraction: f64,
    mark_density: f64,
    experience_mean: f64,
    address_specified_mean: f64,
    address_complete_fraction: f64,
    structurally_correct: usize,
    address_random_baseline: f64,
}

fn population_diagnostics<const N: usize>(population: &Population<N>) -> PopulationDiagnostics {
    let mut micro_size = 0u64;
    let mut specificity_sum = 0.0;
    let mut specificity_max = 0usize;
    let mut quality_sum = 0.0;
    let mut quality_max = 0.0f64;
    let mut above_half_quality = 0usize;
    let mut marked = 0usize;
    let mut mark_density_sum = 0.0;
    let mut experience_sum = 0.0;
    let control_bits = control_bits_for(N);
    let input_bits = N - 1;
    let mut address_specified_sum = 0.0;
    let mut address_complete = 0usize;
    let mut structurally_correct = 0usize;

    for classifier in population.iter() {
        micro_size += classifier.num as u64;
        let specificity = classifier.condition.specificity();
        specificity_sum += specificity as f64;
        specificity_max = specificity_max.max(specificity);
        quality_sum += classifier.q;
        quality_max = quality_max.max(classifier.q);
        if classifier.q > 0.5 {
            above_half_quality += 1;
        }
        experience_sum += classifier.exp as f64;
        let marked_attributes = classifier
            .mark
            .attributes
            .iter()
            .filter(|attribute| !attribute.is_empty())
            .count();
        if marked_attributes > 0 {
            marked += 1;
            mark_density_sum += marked_attributes as f64 / N as f64;
        }

        let address_specified = (0..control_bits)
            .filter(|index| !classifier.condition.symbols[*index].is_wildcard())
            .count();
        address_specified_sum += address_specified as f64;
        if address_specified == control_bits {
            address_complete += 1;
            let mut address = 0usize;
            for index in 0..control_bits {
                let bit = match classifier.condition.symbols[index].token() {
                    Some(token) => (token - b'0') as usize,
                    None => 0,
                };
                address = (address << 1) | bit;
            }
            let data_index = control_bits + address;
            if data_index < input_bits
                && !classifier.condition.symbols[data_index].is_wildcard()
                && specificity == control_bits + 1
            {
                structurally_correct += 1;
            }
        }
    }

    let size = population.len().max(1) as f64;
    PopulationDiagnostics {
        micro_size,
        specificity_mean: specificity_sum / size,
        specificity_max,
        quality_mean: quality_sum / size,
        quality_max,
        above_half_quality,
        marked_fraction: marked as f64 / size,
        mark_density: if marked == 0 { 0.0 } else { mark_density_sum / marked as f64 },
        experience_mean: experience_sum / size,
        address_specified_mean: address_specified_sum / size,
        address_complete_fraction: address_complete as f64 / size,
        structurally_correct,
        address_random_baseline: (specificity_sum / size) * control_bits as f64 / input_bits as f64,
    }
}

struct ReachLimits {
    trials_cap: u64,
    time_cap: Duration,
    eval_interval: u64,
    log_trajectory: bool,
    log_diagnostics: bool,
    log_coverage: bool,
    log_quadrant_detail: bool,
    encoding: Encoding,
    epsilon: f64,
    log_accuracy: bool,
}

fn run_reach_protocol<const N: usize, A>(
    agent: &mut A,
    env: &mut Multiplexer<N>,
    selector: &EpsilonGreedy,
    size: usize,
    limits: &ReachLimits,
) -> ReachOutcome
where
    A: LearningAgent<N>,
{
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
            let metrics = agent.run_explore_trial(env, selector, &bootstrap, time);
            time += metrics.steps as u64;
            trials_used += 1;
            trials_since_eval += 1;
        }

        peak_macro_population = peak_macro_population.max(agent.population().len());
        peak_rss = peak_rss.max(peak_rss_bytes());

        if peak_rss > RSS_CAP_BYTES {
            break Verdict::MemoryLimited;
        }
        if start.elapsed() > limits.time_cap {
            break Verdict::TimeLimited;
        }
        if trials_since_eval >= limits.eval_interval {
            trials_since_eval = 0;
            final_knowledge =
                evaluate_knowledge(
                agent.population(),
                theta_r,
                SAMPLE_INPUTS,
                SAMPLE_SEED,
                limits.encoding,
            );
            if limits.log_trajectory {
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
            if limits.log_diagnostics {
                let diagnostics = population_diagnostics(agent.population());
                println!(
                    "  mpx-{size} diag: trials={trials_used} micro={} pop_spec={:.2} spec_max={} q_mean={:.3} q_max={:.3} q_above_half={} marked={:.3} mark_density={:.3} exp_mean={:.1} addr_spec={:.3} addr_random={:.3} addr_full={:.4} correct={}",
                    diagnostics.micro_size,
                    diagnostics.specificity_mean,
                    diagnostics.specificity_max,
                    diagnostics.quality_mean,
                    diagnostics.quality_max,
                    diagnostics.above_half_quality,
                    diagnostics.marked_fraction,
                    diagnostics.mark_density,
                    diagnostics.experience_mean,
                    diagnostics.address_specified_mean,
                    diagnostics.address_random_baseline,
                    diagnostics.address_complete_fraction,
                    diagnostics.structurally_correct,
                );
            }
            if limits.log_coverage {
                let breakdown = knowledge_breakdown(agent.population(), theta_r, limits.encoding);
                println!(
                    "  mpx-{size} cover: trials={trials_used} overall={:.4} a0_nochange={:.4} a0_change={:.4} a1_nochange={:.4} a1_change={:.4} matched_but_wrong={}",
                    breakdown.overall(),
                    breakdown.fraction(0),
                    breakdown.fraction(1),
                    breakdown.fraction(2),
                    breakdown.fraction(3),
                    breakdown.matched_but_wrong,
                );
            }
            if limits.log_quadrant_detail {
                let detail = quadrant_detail(agent.population(), limits.encoding);
                println!(
                    "  mpx-{size} qdetail: trials={trials_used} a0nc_any={:.4} a0nc_q={:.3} a0c_any={:.4} a0c_q={:.3} a1nc_any={:.4} a1nc_q={:.3} a1c_any={:.4} a1c_q={:.3}",
                    detail.fraction(0), detail.best_quality[0],
                    detail.fraction(1), detail.best_quality[1],
                    detail.fraction(2), detail.best_quality[2],
                    detail.fraction(3), detail.best_quality[3],
                );
            }
            if limits.log_accuracy {
                let accuracy = answer_accuracy(
                    agent.population(),
                    Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
                    limits.encoding,
                );
                println!("  mpx-{size} acc: trials={trials_used} accuracy={accuracy:.4}");
            }
            if final_knowledge >= 1.0 {
                break Verdict::Success;
            }
        }
        if trials_used >= limits.trials_cap {
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

fn run_reach_repeat<const N: usize>(
    size: usize,
    seed: u64,
    gen: GenConfig,
    agent_options: AgentOptions,
    limits: &ReachLimits,
) -> ReachOutcome {
    let mut config = Configuration::mpx();
    config.epsilon = limits.epsilon;
    config.do_ga = gen.do_ga;
    config.u_max = gen.u_max;
    config.alp_gen_variant = gen.alp_gen_variant;

    let mut env =
        Multiplexer::<N>::with_encoding(Box::new(ChaChaRandomSource::from_seed(seed)), limits.encoding);
    let selector = EpsilonGreedy {
        number_of_possible_actions: Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
        epsilon: limits.epsilon,
    };

    match agent_options.agent {
        AgentChoice::Acs2 => {
            let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
            run_reach_protocol(&mut agent, &mut env, &selector, size, limits)
        }
        AgentChoice::Acs2Er => {
            let mut agent = Acs2ErAgent::<N, _>::new(
                config,
                agent_options.replay,
                ChaChaRandomSource::from_seed(seed),
            );
            run_reach_protocol(&mut agent, &mut env, &selector, size, limits)
        }
    }
}

fn run_reach_dispatch(
    size: usize,
    seed: u64,
    gen: GenConfig,
    agent_options: AgentOptions,
    limits: &ReachLimits,
) -> ReachOutcome {
    match size {
        37 => run_reach_repeat::<38>(size, seed, gen, agent_options, limits),
        70 => run_reach_repeat::<71>(size, seed, gen, agent_options, limits),
        135 => run_reach_repeat::<136>(size, seed, gen, agent_options, limits),
        20 => run_reach_repeat::<21>(size, seed, gen, agent_options, limits),
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
    log_diagnostics: bool,
    log_coverage: bool,
    log_quadrant_detail: bool,
    encoding: Encoding,
    epsilon: f64,
    log_accuracy: bool,
    agent: AgentOptions,
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
            log_diagnostics: false,
            log_coverage: false,
            log_quadrant_detail: false,
            encoding: Encoding::Flip,
            epsilon: EXPLORE_EPSILON,
            log_accuracy: false,
            agent: AgentOptions::default(),
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
                "--log-diagnostics" => options.log_diagnostics = true,
                "--log-coverage" => options.log_coverage = true,
                "--log-quadrant-detail" => options.log_quadrant_detail = true,
                "--epsilon" => options.epsilon = args.next().unwrap().parse().unwrap(),
                "--log-accuracy" => options.log_accuracy = true,
                "--encoding" => {
                    options.encoding = parse_encoding(&args.next().expect("--encoding needs flip|outcome"))
                }
                other => {
                    if !options.agent.try_parse_flag(other, &mut args) {
                        panic!("unknown flag {other}")
                    }
                }
            }
        }
        options
    }
}

fn main() {
    let options = Options::parse();
    println!(
        "acs2-bench mpx-reach: {} sizes={:?} n_exp={} seed={} rss_cap={}GB time_cap={}s do_ga={} alp_gen_variant={} epsilon={}",
        options.agent.describe(),
        options.sizes,
        options.n_exp,
        options.seed,
        RSS_CAP_BYTES as f64 / 1e9,
        options.time_cap_secs,
        options.do_ga,
        variant_label(options.alp_gen_variant),
        options.epsilon,
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
        let limits = ReachLimits {
            trials_cap,
            time_cap: Duration::from_secs(options.time_cap_secs),
            eval_interval: options.eval_interval,
            log_trajectory: options.log_trajectory,
            log_diagnostics: options.log_diagnostics,
            log_coverage: options.log_coverage,
            log_quadrant_detail: options.log_quadrant_detail,
            encoding: options.encoding,
            epsilon: options.epsilon,
            log_accuracy: options.log_accuracy,
        };
        for repeat in 0..options.n_exp {
            let outcome = run_reach_dispatch(
                size,
                options.seed + repeat as u64,
                gen,
                options.agent,
                &limits,
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
