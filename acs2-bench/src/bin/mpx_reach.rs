use std::mem::size_of;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use acs2_bench::checkpoint::{self, CheckpointSettings, RunState};
use acs2_bench::{
    parse_u_max_mode, parse_variant, resolve_u_max, variant_label, AgentChoice, AgentOptions,
    UMaxMode,
};
use acs2_core::acs2er::Acs2ErAgent;
use acs2_core::action_selection::{ActionSelector, BestAction, EpsilonGreedy};
use acs2_core::agent::Agent;
use acs2_core::checkpoint::Checkpointed;
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
const DEFAULT_RSS_CAP_BYTES: u64 = 5_600_000_000;
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
    // ru_maxrss is bytes on macOS and kilobytes on Linux; treating the Linux value
    // as bytes silently disabled the RSS cap on the cluster by a factor of 1024.
    let scale: u64 = if cfg!(target_os = "linux") { 1024 } else { 1 };
    (usage.ru_maxrss as u64).saturating_mul(scale)
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

    fn from_label(label: &str) -> Self {
        match label {
            "SUCCESS" => Verdict::Success,
            "TRIALS-LIMITED" => Verdict::TrialsLimited,
            "MEMORY-LIMITED" => Verdict::MemoryLimited,
            "TIME-LIMITED" => Verdict::TimeLimited,
            other => panic!("checkpoint carries an unknown verdict {other}"),
        }
    }
}

struct ReachOutcome {
    verdict: Verdict,
    already_finished: bool,
    trials_used: u64,
    final_knowledge: Option<f64>,
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

fn next_accuracy_trial(trials_used: u64, step: u64) -> u64 {
    (trials_used / step).saturating_add(1).saturating_mul(step)
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
    accuracy_every: u64,
    rss_cap_bytes: u64,
    strict_resource_limits: bool,
}

impl ReachLimits {
    fn accuracy_step(&self) -> u64 {
        self.eval_interval
            .saturating_mul(self.accuracy_every)
            .max(self.eval_interval)
    }

    fn resource_verdict(&self, elapsed: Duration, peak_rss: u64) -> Option<Verdict> {
        if peak_rss > self.rss_cap_bytes {
            Some(Verdict::MemoryLimited)
        } else if elapsed > self.time_cap {
            Some(Verdict::TimeLimited)
        } else {
            None
        }
    }
}

struct CheckpointPlan<'a> {
    settings: &'a CheckpointSettings,
    identity: String,
}

struct ProtocolProgress {
    time: u64,
    trials_used: u64,
    trials_since_eval: u64,
    peak_macro_population: usize,
    peak_rss_bytes: u64,
    wall_seconds: f64,
    final_knowledge: f64,
    knowledge_trials: Option<u64>,
}

fn save_checkpoint<const N: usize, A>(
    plan: &CheckpointPlan,
    eval_interval: u64,
    progress: &ProtocolProgress,
    verdict: Option<Verdict>,
    agent: &A,
    env: &Multiplexer<N>,
) where
    A: Checkpointed<N>,
{
    let state = RunState {
        identity: plan.identity.clone(),
        eval_interval,
        trials_used: progress.trials_used,
        time: progress.time,
        trials_since_eval: progress.trials_since_eval,
        peak_macro_population: progress.peak_macro_population,
        peak_rss_bytes: progress.peak_rss_bytes,
        wall_seconds: progress.wall_seconds,
        final_knowledge: progress.final_knowledge,
        knowledge_trials: progress.knowledge_trials,
        verdict: verdict.map(|verdict| verdict.label().to_string()),
        env_rng: env
            .rng()
            .capture_state()
            .expect("the environment random source cannot be checkpointed"),
        agent: agent.capture(),
    };
    checkpoint::write(&plan.settings.path, &state);
}

fn run_reach_protocol<const N: usize, A>(
    agent: &mut A,
    env: &mut Multiplexer<N>,
    selector: &EpsilonGreedy,
    size: usize,
    limits: &ReachLimits,
    checkpoint: Option<&CheckpointPlan>,
) -> ReachOutcome
where
    A: LearningAgent<N> + Checkpointed<N>,
{
    let bootstrap = MaxFitnessBootstrap;
    let theta_r = agent.config().theta_r;

    let start = Instant::now();
    let mut time: u64 = 0;
    let mut trials_used: u64 = 0;
    let mut peak_macro_population = 0usize;
    let mut peak_rss = peak_rss_bytes();
    let mut final_knowledge = 0.0;
    let mut knowledge_trials = None;
    let mut trials_since_eval: u64 = 0;
    let mut trials_since_checkpoint: u64 = 0;
    // Wall-clock has two readings once a run spans several jobs: `limits.time_cap` bounds
    // THIS process, because that is what the queue kills, while the reported figure is the
    // elapsed time along the RETAINED checkpoint history -- work a job did after its last
    // save and then lost to a kill is not in it. SLURM's own accounting is the authority on
    // what a chain actually spent.
    let mut carried_wall_seconds = 0.0f64;
    let mut resumed_verdict: Option<Verdict> = None;
    let mut already_reported = false;
    // A resource cap can stop a job on the very batch an evaluation was due, because
    // the cap is checked first. The checkpoint then carries a measurement the run
    // owes: it must be paid before any further training, or every later evaluation
    // lands at a different trial than an uninterrupted run's and the reported
    // trials-to-success moves with it.
    let mut train_before_measuring = true;

    if let Some(plan) = checkpoint {
        if plan.settings.path.exists() {
            let restored = checkpoint::read::<N>(&plan.settings.path);
            assert_eq!(
                restored.identity, plan.identity,
                "the checkpoint at {} was written for another configuration",
                plan.settings.path.display()
            );
            // The evaluation interval does not change what the agent learns, but it
            // decides which trials can be observed -- and trials-to-success is the
            // number this project reports. Splicing two sampling rates into one run
            // makes that number mean nothing, so it takes an explicit decision.
            if restored.eval_interval != limits.eval_interval {
                assert!(
                    plan.settings.allow_eval_interval_change,
                    "the checkpoint at {} was measured every {} trials, this job every {}; \
pass --checkpoint-allow-eval-change to splice the two sampling rates deliberately",
                    plan.settings.path.display(),
                    restored.eval_interval,
                    limits.eval_interval,
                );
                println!(
                    "  mpx-{size} eval-interval-changed: from={} to={} (trials-to-success is no longer comparable across this run)",
                    restored.eval_interval, limits.eval_interval,
                );
            }
            time = restored.time;
            trials_used = restored.trials_used;
            trials_since_eval = restored.trials_since_eval;
            peak_macro_population = restored.peak_macro_population;
            peak_rss = restored.peak_rss_bytes;
            carried_wall_seconds = restored.wall_seconds;
            final_knowledge = restored.final_knowledge;
            knowledge_trials = restored.knowledge_trials;
            assert!(
                env.rng_mut().restore_state(&restored.env_rng),
                "the environment random source cannot be checkpointed"
            );
            let stored_verdict = restored.verdict.as_deref().map(Verdict::from_label);
            agent.restore(restored.agent);
            train_before_measuring = trials_since_eval < limits.eval_interval;
            if stored_verdict == Some(Verdict::Success) {
                resumed_verdict = stored_verdict;
                already_reported = true;
            } else if knowledge_trials == Some(trials_used) && final_knowledge >= 1.0 {
                // `--strict-resource-limits` can stop a job between measuring knowledge
                // 1.0 and declaring SUCCESS. The run is solved at the trial the
                // measurement names; training on is what would move that number.
                resumed_verdict = Some(Verdict::Success);
            }
            println!(
                "  mpx-{size} resumed: trials={trials_used} time={time} since_eval={trials_since_eval} knowledge={final_knowledge:.4} carried_wall={carried_wall_seconds:.0}s from={}",
                plan.settings.path.display(),
            );
        }
    }

    let mut accuracy_due_at = next_accuracy_trial(trials_used, limits.accuracy_step());

    let verdict = match resumed_verdict {
        Some(verdict) => verdict,
        None => loop {
            if train_before_measuring {
                if trials_used >= limits.trials_cap {
                    break Verdict::TrialsLimited;
                }

                for _ in 0..TIME_CHECK_BATCH {
                    let metrics = agent.run_explore_trial(env, selector, &bootstrap, time);
                    time += metrics.steps as u64;
                    trials_used += 1;
                    trials_since_eval += 1;
                }
                trials_since_checkpoint += TIME_CHECK_BATCH as u64;

                peak_macro_population = peak_macro_population.max(agent.population().len());
                peak_rss = peak_rss.max(peak_rss_bytes());

                if peak_rss > limits.rss_cap_bytes {
                    break Verdict::MemoryLimited;
                }
                if start.elapsed() > limits.time_cap {
                    break Verdict::TimeLimited;
                }
            }
            train_before_measuring = true;

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
                knowledge_trials = Some(trials_used);
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
                        carried_wall_seconds + start.elapsed().as_secs_f64(),
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
                if limits.log_accuracy && trials_used >= accuracy_due_at {
                    accuracy_due_at = next_accuracy_trial(trials_used, limits.accuracy_step());
                    let accuracy = answer_accuracy(
                        agent.population(),
                        Multiplexer::<N>::NUMBER_OF_POSSIBLE_ACTIONS,
                        limits.encoding,
                    );
                    println!("  mpx-{size} acc: trials={trials_used} accuracy={accuracy:.4}");
                }
                if limits.strict_resource_limits {
                    peak_rss = peak_rss.max(peak_rss_bytes());
                    if let Some(verdict) = limits.resource_verdict(start.elapsed(), peak_rss) {
                        break verdict;
                    }
                }
                if final_knowledge >= 1.0 {
                    break Verdict::Success;
                }
            }

            if let Some(plan) = checkpoint {
                if plan.settings.every > 0 && trials_since_checkpoint >= plan.settings.every {
                    trials_since_checkpoint = 0;
                    let progress = ProtocolProgress {
                        time,
                        trials_used,
                        trials_since_eval,
                        peak_macro_population,
                        peak_rss_bytes: peak_rss,
                        wall_seconds: carried_wall_seconds + start.elapsed().as_secs_f64(),
                        final_knowledge,
                        knowledge_trials,
                    };
                    save_checkpoint(plan, limits.eval_interval, &progress, None, agent, env);
                }
            }
        },
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

    let wall_seconds = if already_reported {
        carried_wall_seconds
    } else {
        carried_wall_seconds + start.elapsed().as_secs_f64()
    };

    if let (Some(plan), false) = (checkpoint, already_reported) {
        let progress = ProtocolProgress {
            time,
            trials_used,
            trials_since_eval,
            peak_macro_population,
            peak_rss_bytes: peak_rss,
            wall_seconds,
            final_knowledge,
            knowledge_trials,
        };
        save_checkpoint(plan, limits.eval_interval, &progress, Some(verdict), agent, env);
    }

    ReachOutcome {
        verdict,
        already_finished: already_reported,
        trials_used,
        final_knowledge: knowledge_trials.filter(|&trial| trial == trials_used).map(|_| final_knowledge),
        reliable_count,
        mean_reliable_specificity,
        peak_macro_population,
        peak_rss_bytes: peak_rss,
        wall_seconds,
    }
}

/// Everything a resumed run must agree with the checkpoint about.
///
/// A checkpoint carries a learning state, not a configuration: resuming it under a
/// different seed, encoding or generalization setting would silently splice two
/// unrelated runs into one trajectory. The identity is compared verbatim, so any
/// field added here becomes a resume-time gate. The stopping limits -- trials, wall
/// clock, RSS -- are deliberately absent: a chained run raises them per job, and they
/// change when a run stops, not what it learns.
///
/// The whole `Configuration` goes in through its `Debug` rendering rather than a
/// hand-listed subset. Flags are what an operator sets, but the learning constants are
/// what a trial obeys, and a later executable could change `beta` or a threshold and
/// still accept a checkpoint written by an earlier one. Rendering the struct means a
/// field added later gates resumes without anyone remembering to add it here.
fn checkpoint_identity(
    size: usize,
    seed: u64,
    config: &Configuration,
    agent_options: AgentOptions,
    limits: &ReachLimits,
) -> String {
    format!(
        "size={size} seed={seed} {} encoding={} eval_sample={SAMPLE_INPUTS}/{SAMPLE_SEED} config={}",
        agent_options.describe(),
        encoding_label(limits.encoding),
        format!("{config:?}").replace(' ', ""),
    )
}

fn run_reach_repeat<const N: usize>(
    size: usize,
    seed: u64,
    gen: GenConfig,
    agent_options: AgentOptions,
    limits: &ReachLimits,
    checkpoint: Option<&CheckpointSettings>,
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
    let plan = checkpoint.map(|settings| CheckpointPlan {
        settings,
        identity: checkpoint_identity(size, seed, &config, agent_options, limits),
    });

    match agent_options.agent {
        AgentChoice::Acs2 => {
            let mut agent = Agent::<N, _>::new(config, ChaChaRandomSource::from_seed(seed));
            run_reach_protocol(&mut agent, &mut env, &selector, size, limits, plan.as_ref())
        }
        AgentChoice::Acs2Er => {
            let mut agent = Acs2ErAgent::<N, _>::new(
                config,
                agent_options.replay,
                ChaChaRandomSource::from_seed(seed),
            );
            run_reach_protocol(&mut agent, &mut env, &selector, size, limits, plan.as_ref())
        }
    }
}

fn run_reach_dispatch(
    size: usize,
    seed: u64,
    gen: GenConfig,
    agent_options: AgentOptions,
    limits: &ReachLimits,
    checkpoint: Option<&CheckpointSettings>,
) -> ReachOutcome {
    match size {
        37 => run_reach_repeat::<38>(size, seed, gen, agent_options, limits, checkpoint),
        70 => run_reach_repeat::<71>(size, seed, gen, agent_options, limits, checkpoint),
        135 => run_reach_repeat::<136>(size, seed, gen, agent_options, limits, checkpoint),
        264 => run_reach_repeat::<265>(size, seed, gen, agent_options, limits, checkpoint),
        20 => run_reach_repeat::<21>(size, seed, gen, agent_options, limits, checkpoint),
        other => panic!("reach not configured for {other}-bit multiplexer"),
    }
}

fn component_memory<const N: usize>(size: usize, rss_cap_bytes: u64) {
    let condition = size_of::<Condition<N>>();
    let effect = size_of::<Effect<N>>();
    let mark = size_of::<Mark<N>>();
    let classifier = size_of::<Classifier<N>>();
    let pop_threshold = rss_cap_bytes / classifier as u64;
    println!(
        "  mem mpx-{size} (N={N}): condition={condition}B effect={effect}B mark={mark}B (stack) \
         classifier={classifier}B  mark/classifier={:.1}%  rss-cap pop-threshold={pop_threshold}",
        100.0 * mark as f64 / classifier as f64,
    );
}

fn encoding_label(encoding: Encoding) -> &'static str {
    match encoding {
        Encoding::Flip => "flip",
        Encoding::Outcome => "outcome",
    }
}

fn report_component_memory(size: usize, rss_cap_bytes: u64) {
    match size {
        6 => component_memory::<7>(size, rss_cap_bytes),
        11 => component_memory::<12>(size, rss_cap_bytes),
        20 => component_memory::<21>(size, rss_cap_bytes),
        37 => component_memory::<38>(size, rss_cap_bytes),
        70 => component_memory::<71>(size, rss_cap_bytes),
        135 => component_memory::<136>(size, rss_cap_bytes),
        264 => component_memory::<265>(size, rss_cap_bytes),
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
    accuracy_every: u64,
    rss_cap_bytes: u64,
    agent: AgentOptions,
    strict_resource_limits: bool,
    isolate_repeats: bool,
    checkpoint_path: Option<PathBuf>,
    checkpoint_every: u64,
    checkpoint_allow_eval_change: bool,
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
            accuracy_every: 1,
            rss_cap_bytes: DEFAULT_RSS_CAP_BYTES,
            agent: AgentOptions::default(),
            strict_resource_limits: false,
            isolate_repeats: false,
            checkpoint_path: None,
            checkpoint_every: 0,
            checkpoint_allow_eval_change: false,
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
                "--accuracy-every" => {
                    options.accuracy_every = args
                        .next()
                        .expect("--accuracy-every needs a value")
                        .parse()
                        .expect("--accuracy-every must be a positive integer");
                    assert!(
                        options.accuracy_every > 0,
                        "--accuracy-every must be at least 1"
                    );
                }
                "--strict-resource-limits" => options.strict_resource_limits = true,
                "--isolate-repeats" => options.isolate_repeats = true,
                "--checkpoint-path" => {
                    options.checkpoint_path = Some(PathBuf::from(
                        args.next().expect("--checkpoint-path needs a value"),
                    ))
                }
                "--checkpoint-allow-eval-change" => {
                    options.checkpoint_allow_eval_change = true
                }
                "--checkpoint-every" => {
                    options.checkpoint_every = args
                        .next()
                        .expect("--checkpoint-every needs a value")
                        .parse()
                        .expect("--checkpoint-every must be a trial count")
                }
                "--rss-cap-gb" => {
                    let gb: f64 = args
                        .next()
                        .expect("--rss-cap-gb needs a value")
                        .parse()
                        .expect("--rss-cap-gb must be a number");
                    assert!(gb > 0.0, "--rss-cap-gb must be positive");
                    options.rss_cap_bytes = (gb * 1e9) as u64;
                }
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
    let isolated_worker = std::env::var_os("ACS2_REACH_REPEAT_WORKER").is_some();
    if options.isolate_repeats && !isolated_worker {
        let executable = std::env::current_exe().expect("cannot locate mpx_reach executable");
        let original_args: Vec<_> = std::env::args_os().skip(1).collect();
        for &size in &options.sizes {
            for repeat in 0..options.n_exp {
                let status = Command::new(&executable)
                    .args(&original_args)
                    .args(["--sizes", &size.to_string(), "--n-exp", "1", "--seed",
                           &(options.seed + repeat as u64).to_string()])
                    .env("ACS2_REACH_REPEAT_WORKER", "1")
                    .status()
                    .expect("cannot start isolated repeat");
                if !status.success() {
                    std::process::exit(status.code().unwrap_or(1));
                }
            }
        }
        return;
    }
    let checkpoint = options.checkpoint_path.clone().map(|path| {
        assert_eq!(
            options.n_exp, 1,
            "--checkpoint-path holds one run: use --n-exp 1"
        );
        assert_eq!(
            options.sizes.len(),
            1,
            "--checkpoint-path holds one run: pass a single --sizes value"
        );
        assert!(
            !options.isolate_repeats,
            "--checkpoint-path and --isolate-repeats cannot be combined"
        );
        CheckpointSettings {
            path,
            every: options.checkpoint_every,
            allow_eval_interval_change: options.checkpoint_allow_eval_change,
        }
    });

    println!(
        "acs2-bench mpx-reach: {} sizes={:?} n_exp={} seed={} rss_cap={}GB time_cap={}s do_ga={} alp_gen_variant={} epsilon={} encoding={} eval_interval={} accuracy_every={} strict_resource_limits={} rss_scope={} checkpoint={} checkpoint_every={}",
        options.agent.describe(),
        options.sizes,
        options.n_exp,
        options.seed,
        options.rss_cap_bytes as f64 / 1e9,
        options.time_cap_secs,
        options.do_ga,
        variant_label(options.alp_gen_variant),
        options.epsilon,
        encoding_label(options.encoding),
        options.eval_interval,
        options.accuracy_every,
        options.strict_resource_limits,
        if options.isolate_repeats && isolated_worker { "repeat-process" } else { "process-lifetime" },
        if checkpoint.is_some() { "on" } else { "off" },
        options.checkpoint_every,
    );

    for &size in &options.sizes {
        let trials_cap = trials_cap_for(size);
        report_component_memory(size, options.rss_cap_bytes);
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
            accuracy_every: options.accuracy_every,
            rss_cap_bytes: options.rss_cap_bytes,
            strict_resource_limits: options.strict_resource_limits,
        };
        for repeat in 0..options.n_exp {
            let outcome = run_reach_dispatch(
                size,
                options.seed + repeat as u64,
                gen,
                options.agent,
                &limits,
                checkpoint.as_ref(),
            );
            verdicts.push(outcome.verdict);
            // A chained run's later jobs find a finished checkpoint and have nothing to
            // do, but they must still print a verdict line. The job that closed the run
            // can be killed between saving its checkpoint and printing, and then this is
            // the only place the SUCCESS ever reaches the archive. The parser keeps one
            // verdict per run, so the repetition costs nothing.
            let reopened = if outcome.already_finished { " reopened=true" } else { "" };
            let knowledge = outcome.final_knowledge
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "unmeasured".to_string());
            let knowledge_trials = outcome.final_knowledge
                .map(|_| outcome.trials_used.to_string())
                .unwrap_or_else(|| "unmeasured".to_string());
            println!(
                "  mpx-{size} repeat {repeat}: {} trials={} knowledge={knowledge} knowledge_trials={knowledge_trials} reliable={} spec={:.2}/{} peak_macro={} peak_rss={:.2}GB wall={:.1}s{reopened}",
                outcome.verdict.label(),
                outcome.trials_used,
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
