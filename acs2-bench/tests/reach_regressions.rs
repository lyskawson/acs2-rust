include!("../src/bin/mpx_reach.rs");

#[cfg(test)]
mod regression {
    use super::*;
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use acs2_core::action_selection::ActionSelector;
    use acs2_core::environment::Environment;
    use acs2_core::acs2er::ReplayConfiguration;
    use acs2_core::checkpoint::AgentState;
    use acs2_core::rl::BootstrapEstimator;
    use acs2_core::rng::RandomSource;
    use acs2_core::symbol::Symbol;
    use acs2_core::trial::TrialMetrics;

    struct FrozenAgent {
        population: Population<7>,
        config: Configuration,
        reads: Cell<usize>,
        evaluation_delay: Duration,
    }

    impl LearningAgent<7> for FrozenAgent {
        fn run_explore_trial<E: Environment<7>, S: ActionSelector<7>, B: BootstrapEstimator<7>>(
            &mut self, _: &mut E, _: &S, _: &B, _: u64,
        ) -> TrialMetrics {
            TrialMetrics { steps: 1, reward: 0.0 }
        }

        fn run_exploit_trial<E: Environment<7>, B: BootstrapEstimator<7>>(
            &mut self, _: &mut E, _: &B, _: u64,
        ) -> TrialMetrics { unreachable!() }

        fn population(&self) -> &Population<7> {
            self.reads.set(self.reads.get() + 1);
            if self.reads.get() == 2 {
                std::thread::sleep(self.evaluation_delay);
            }
            &self.population
        }

        fn config(&self) -> &Configuration { &self.config }
    }

    impl Checkpointed<7> for FrozenAgent {
        fn capture(&self) -> AgentState<7> {
            AgentState {
                population: self.population.classifiers().to_vec(),
                rng: ChaChaRandomSource::from_seed(0).capture_state().unwrap(),
                replay: None,
            }
        }

        fn restore(&mut self, state: AgentState<7>) {
            self.population = Population::from_classifiers(state.population);
        }
    }

    fn limits() -> ReachLimits {
        ReachLimits {
            trials_cap: 1500, time_cap: Duration::from_secs(60), eval_interval: 1000,
            log_trajectory: false, log_diagnostics: false, log_coverage: false,
            log_quadrant_detail: false, encoding: Encoding::Flip, epsilon: 0.8,
            log_accuracy: false, rss_cap_bytes: u64::MAX, strict_resource_limits: false,
        }
    }

    fn run(limits: &ReachLimits, complete: bool, delay: Duration) -> ReachOutcome {
        let mut classifiers = Vec::new();
        if complete {
            for action in 0..2 {
                for change in [false, true] {
                    let mut classifier = Classifier::general(Some(action), &Configuration::mpx());
                    classifier.q = 1.0;
                    if change { classifier.effect.set(6, Symbol::Token(b'1')); }
                    classifiers.push(classifier);
                }
            }
        }
        let mut agent = FrozenAgent {
            population: Population::from_classifiers(classifiers), config: Configuration::mpx(),
            reads: Cell::new(0), evaluation_delay: delay,
        };
        let mut env = Multiplexer::<7>::with_encoding(
            Box::new(ChaChaRandomSource::from_seed(42)), Encoding::Flip);
        run_reach_protocol(&mut agent, &mut env,
            &EpsilonGreedy { number_of_possible_actions: 2, epsilon: 0.8 }, 6, limits, None)
    }

    #[test]
    fn verdict_requires_a_measurement_at_the_terminal_trial() {
        let mut limits = limits();
        let stale = run(&limits, false, Duration::ZERO);
        assert_eq!(stale.trials_used, 1500);
        assert_eq!(stale.verdict, Verdict::TrialsLimited);
        assert_eq!(stale.final_knowledge, None);
        limits.eval_interval = 2000;
        assert_eq!(run(&limits, false, Duration::ZERO).final_knowledge, None);
        limits.eval_interval = 500;
        assert_eq!(run(&limits, false, Duration::ZERO).final_knowledge, Some(0.0));
        let success = run(&limits, true, Duration::ZERO);
        assert_eq!(success.verdict, Verdict::Success);
        assert_eq!(success.trials_used, 500);
        assert_eq!(success.final_knowledge, Some(1.0));
    }

    #[test]
    fn strict_limits_check_evaluation_cost_before_success() {
        let mut limits = limits();
        limits.time_cap = Duration::from_millis(300);
        limits.eval_interval = 500;
        let delay = Duration::from_millis(400);
        assert_eq!(run(&limits, true, delay).verdict, Verdict::Success);
        limits.strict_resource_limits = true;
        let strict = run(&limits, true, delay);
        assert_eq!(strict.verdict, Verdict::TimeLimited);
        assert_eq!(strict.final_knowledge, Some(1.0));
        limits.rss_cap_bytes = 100;
        assert_eq!(limits.resource_verdict(Duration::ZERO, 101), Some(Verdict::MemoryLimited));
        assert_eq!(limits.resource_verdict(Duration::ZERO, 100), None);
    }

    #[test]
    fn fresh_processes_do_not_share_a_previous_repeat_peak() {
        if let Some(mode) = std::env::var_os("ACS2_RSS_REGRESSION") {
            let bytes = if mode == "large" { 64 * 1024 * 1024 } else { 1024 };
            let allocation = vec![1u8; bytes];
            std::hint::black_box(&allocation);
            println!("rss_measurement={}", peak_rss_bytes());
            return;
        }
        let worker = |mode| {
            let output = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "regression::fresh_processes_do_not_share_a_previous_repeat_peak", "--nocapture"])
                .env("ACS2_RSS_REGRESSION", mode).output().unwrap();
            assert!(output.status.success());
            String::from_utf8(output.stdout).unwrap().lines()
                .find_map(|line| line.strip_prefix("rss_measurement=")?.parse::<u64>().ok()).unwrap()
        };
        let large = worker("large");
        let small = worker("small");
        assert!(large > small + 32 * 1024 * 1024, "large={large} small={small}");
    }

    const CHECKPOINT_SIZE: usize = 6;
    const CHECKPOINT_SEED: u64 = 42;
    const CHECKPOINT_EVAL_INTERVAL: u64 = 750;
    const CHECKPOINT_SPLIT_TRIALS: u64 = 1500;
    const CHECKPOINT_TOTAL_TRIALS: u64 = 3000;
    const CHECKPOINT_TEST: &str =
        "regression::a_resumed_run_reproduces_an_uninterrupted_trajectory";

    fn checkpoint_limits(trials_cap: u64) -> ReachLimits {
        ReachLimits {
            trials_cap,
            time_cap: Duration::from_secs(3600),
            eval_interval: CHECKPOINT_EVAL_INTERVAL,
            log_trajectory: true,
            log_diagnostics: false,
            log_coverage: false,
            log_quadrant_detail: false,
            encoding: Encoding::Flip,
            epsilon: 0.8,
            log_accuracy: false,
            rss_cap_bytes: u64::MAX,
            strict_resource_limits: false,
        }
    }

    fn checkpoint_agent(kind: &str) -> AgentOptions {
        match kind {
            "acs2" => AgentOptions::default(),
            "acs2er" => AgentOptions {
                agent: AgentChoice::Acs2Er,
                replay: ReplayConfiguration {
                    buffer_size: 64,
                    min_samples: 32,
                    samples_number: 1,
                },
            },
            other => panic!("unknown checkpoint worker agent {other}"),
        }
    }

    fn checkpoint_worker(mode: &str) {
        let directory = PathBuf::from(std::env::var_os("ACS2_CHECKPOINT_DIR").unwrap());
        let (segment, kind) = mode.split_once(':').expect("worker mode is segment:agent");
        let (path, trials_cap, every) = match segment {
            "whole" => (
                directory.join(format!("{kind}-whole.ckpt")),
                CHECKPOINT_TOTAL_TRIALS,
                0,
            ),
            "part1" => (
                directory.join(format!("{kind}-split.ckpt")),
                CHECKPOINT_SPLIT_TRIALS,
                500,
            ),
            "part2" => (
                directory.join(format!("{kind}-split.ckpt")),
                CHECKPOINT_TOTAL_TRIALS,
                500,
            ),
            other => panic!("unknown checkpoint worker segment {other}"),
        };
        let settings = CheckpointSettings { path, every };
        let gen = GenConfig {
            do_ga: true,
            u_max: 4,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        };
        run_reach_repeat::<7>(
            CHECKPOINT_SIZE,
            CHECKPOINT_SEED,
            gen,
            checkpoint_agent(kind),
            &checkpoint_limits(trials_cap),
            Some(&settings),
        );
    }

    fn without_volatile_fields(text: &str) -> String {
        text.lines()
            .map(|line| {
                line.split_whitespace()
                    .filter(|token| !token.starts_with("wall=") && !token.starts_with("peak_rss="))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn run_checkpoint_worker(mode: &str, directory: &Path) -> Vec<String> {
        let output = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", CHECKPOINT_TEST, "--nocapture"])
            .env("ACS2_CHECKPOINT_REGRESSION", mode)
            .env("ACS2_CHECKPOINT_DIR", directory)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "worker {mode} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .lines()
            .filter(|line| line.contains(" traj: "))
            .map(without_volatile_fields)
            .collect()
    }

    #[test]
    fn a_resumed_run_reproduces_an_uninterrupted_trajectory() {
        if let Some(mode) = std::env::var_os("ACS2_CHECKPOINT_REGRESSION") {
            checkpoint_worker(&mode.to_string_lossy());
            return;
        }

        let directory = std::env::temp_dir().join(format!(
            "acs2-checkpoint-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();

        for kind in ["acs2", "acs2er"] {
            let uninterrupted = run_checkpoint_worker(&format!("whole:{kind}"), &directory);
            let before_the_break = run_checkpoint_worker(&format!("part1:{kind}"), &directory);
            let after_the_break = run_checkpoint_worker(&format!("part2:{kind}"), &directory);

            assert!(
                !before_the_break.is_empty(),
                "{kind}: the first segment measured nothing"
            );
            assert!(
                !after_the_break.is_empty(),
                "{kind}: the resumed segment measured nothing"
            );
            assert_eq!(
                [before_the_break, after_the_break].concat(),
                uninterrupted,
                "{kind}: a resumed trajectory must be identical to an uninterrupted one, trial for trial"
            );

            let whole_state =
                std::fs::read_to_string(directory.join(format!("{kind}-whole.ckpt"))).unwrap();
            let resumed_state =
                std::fs::read_to_string(directory.join(format!("{kind}-split.ckpt"))).unwrap();
            assert_eq!(
                without_volatile_fields(&resumed_state),
                without_volatile_fields(&whole_state),
                "{kind}: population, RNG streams and counters must land where an uninterrupted run leaves them"
            );
        }

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn a_finished_checkpoint_is_reported_once_and_never_relearned() {
        let directory = std::env::temp_dir().join(format!(
            "acs2-checkpoint-finished-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let settings = CheckpointSettings {
            path: directory.join("run.ckpt"),
            every: 0,
        };
        let gen = GenConfig {
            do_ga: true,
            u_max: 4,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        };
        let solve = |trials_cap| {
            run_reach_repeat::<7>(
                CHECKPOINT_SIZE,
                CHECKPOINT_SEED,
                gen,
                AgentOptions::default(),
                &checkpoint_limits(trials_cap),
                Some(&settings),
            )
        };

        let solved = solve(CHECKPOINT_TOTAL_TRIALS);
        assert_eq!(solved.verdict, Verdict::Success);
        assert!(!solved.already_finished);
        let written = std::fs::read_to_string(&settings.path).unwrap();

        let replayed = solve(CHECKPOINT_TOTAL_TRIALS * 4);
        assert!(replayed.already_finished, "a closed run must not learn again");
        assert_eq!(replayed.verdict, Verdict::Success);
        assert_eq!(replayed.trials_used, solved.trials_used);
        assert_eq!(replayed.reliable_count, solved.reliable_count);
        assert_eq!(
            std::fs::read_to_string(&settings.path).unwrap(),
            written,
            "resuming a closed run must leave its checkpoint untouched"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn a_checkpoint_refuses_a_configuration_it_was_not_written_for() {
        let directory = std::env::temp_dir().join(format!(
            "acs2-checkpoint-identity-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("run.ckpt");

        let settings = CheckpointSettings { path: path.clone(), every: 0 };
        let gen = GenConfig {
            do_ga: true,
            u_max: 4,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        };
        run_reach_repeat::<7>(
            CHECKPOINT_SIZE,
            CHECKPOINT_SEED,
            gen,
            AgentOptions::default(),
            &checkpoint_limits(500),
            Some(&settings),
        );

        let previous_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let resumed_under_another_seed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_reach_repeat::<7>(
                CHECKPOINT_SIZE,
                CHECKPOINT_SEED + 1,
                gen,
                AgentOptions::default(),
                &checkpoint_limits(1000),
                Some(&settings),
            )
        }));
        std::panic::set_hook(previous_hook);
        assert!(
            resumed_under_another_seed.is_err(),
            "a checkpoint written for another seed must not be resumed silently"
        );

        std::fs::remove_dir_all(&directory).ok();
    }
}
