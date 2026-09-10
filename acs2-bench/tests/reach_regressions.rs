include!("../src/bin/mpx_reach.rs");

#[cfg(test)]
mod regression {
    use super::*;
    use std::cell::Cell;
    use std::path::{Path, PathBuf};
    use acs2_core::action_selection::ActionSelector;
    use acs2_core::environment::Environment;
    use acs2_bench::derived_u_max;
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
        let settings = CheckpointSettings { path, every, allow_eval_interval_change: false };
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
            .filter(|line| !line.starts_with("sha256 "))
            .map(|line| {
                line.split_whitespace()
                    .filter(|token| !token.starts_with("wall=") && !token.starts_with("peak_rss="))
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn worker_command(switch: &str, mode: &str, test: &str, directory: &Path) -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", test, "--nocapture"])
            .env(switch, mode)
            .env("ACS2_CHECKPOINT_DIR", directory);
        command
    }

    fn run_worker_process(switch: &str, mode: &str, test: &str, directory: &Path) -> Vec<String> {
        let output = worker_command(switch, mode, test, directory).output().unwrap();
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

    fn run_checkpoint_worker(mode: &str, directory: &Path) -> Vec<String> {
        run_worker_process("ACS2_CHECKPOINT_REGRESSION", mode, CHECKPOINT_TEST, directory)
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
            allow_eval_interval_change: false,
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

        let settings = CheckpointSettings {
            path: path.clone(),
            every: 0,
            allow_eval_interval_change: false,
        };
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

    fn checkpoint_directory(purpose: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "acs2-checkpoint-{purpose}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    fn settings_at(path: PathBuf) -> CheckpointSettings {
        CheckpointSettings {
            path,
            every: 0,
            allow_eval_interval_change: false,
        }
    }

    fn mpx6_gen() -> GenConfig {
        GenConfig {
            do_ga: true,
            u_max: 4,
            alp_gen_variant: AlpGenVariant::Pyalcs,
        }
    }

    fn run_mpx6(limits: &ReachLimits, checkpoint: Option<&CheckpointSettings>) -> ReachOutcome {
        run_reach_repeat::<7>(
            CHECKPOINT_SIZE,
            CHECKPOINT_SEED,
            mpx6_gen(),
            AgentOptions::default(),
            limits,
            checkpoint,
        )
    }

    fn panics(work: impl FnOnce()) -> bool {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work));
        std::panic::set_hook(previous);
        outcome.is_err()
    }

    /// A resource cap is checked before the evaluation block, so a job can stop on the
    /// very batch an evaluation was due and leave the measurement unpaid. Resuming
    /// straight into another batch then shifts that evaluation and every later one --
    /// and trials-to-success moves with them. Measured before the fix at k=20: the
    /// evaluation points ran 1500/2500/3500 against 1000/2000/3000 and SUCCESS was
    /// reported at 66,500 trials instead of 67,000.
    #[test]
    fn a_resource_cap_does_not_swallow_an_evaluation_that_was_due() {
        let directory = checkpoint_directory("due");
        let uninterrupted = settings_at(directory.join("whole.ckpt"));
        let interrupted = settings_at(directory.join("split.ckpt"));

        let mut limits = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
        limits.eval_interval = 1000;
        run_mpx6(&limits, Some(&uninterrupted));

        // Two jobs that stop the instant their clock is checked. The second leaves the
        // run owing an evaluation: 1000 trials since the last one, none taken.
        let mut stopped = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
        stopped.eval_interval = 1000;
        stopped.time_cap = Duration::ZERO;
        run_mpx6(&stopped, Some(&interrupted));
        run_mpx6(&stopped, Some(&interrupted));

        let owed = std::fs::read_to_string(&interrupted.path).unwrap();
        assert!(
            owed.contains("trials=1000 ") && owed.contains("since_eval=1000 "),
            "the fixture must stop with an evaluation outstanding: {}",
            owed.lines().nth(2).unwrap_or_default()
        );

        run_mpx6(&limits, Some(&interrupted));

        assert_eq!(
            without_volatile_fields(&std::fs::read_to_string(&interrupted.path).unwrap()),
            without_volatile_fields(&std::fs::read_to_string(&uninterrupted.path).unwrap()),
            "an evaluation owed at the moment a job stopped must be paid before the next batch"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// `--strict-resource-limits` rechecks the clock after evaluation, so a job can
    /// measure knowledge 1.0 and still stop TIME-LIMITED before declaring SUCCESS. The
    /// run is solved at the trial that measurement names; training on would move the
    /// only number this project reports.
    #[test]
    fn a_run_measured_complete_before_its_clock_ran_out_resumes_as_solved() {
        let directory = checkpoint_directory("solved");
        let settings = settings_at(directory.join("run.ckpt"));

        let solved = run_mpx6(&checkpoint_limits(CHECKPOINT_TOTAL_TRIALS), Some(&settings));
        assert_eq!(solved.verdict, Verdict::Success);

        let mut stored = checkpoint::read::<7>(&settings.path);
        stored.verdict = Some("TIME-LIMITED".to_string());
        checkpoint::write(&settings.path, &stored);

        let resumed = run_mpx6(&checkpoint_limits(CHECKPOINT_TOTAL_TRIALS * 4), Some(&settings));
        assert_eq!(resumed.verdict, Verdict::Success);
        assert_eq!(resumed.trials_used, solved.trials_used);
        assert_eq!(resumed.reliable_count, solved.reliable_count);
        assert!(
            !resumed.already_finished,
            "a verdict no earlier job reported must still reach the log"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A closed run's cost is what it cost. Reopening its checkpoint must not add the
    /// new process's start-up time to the archived wall-clock or its footprint to the
    /// archived peak.
    #[test]
    fn reopening_a_closed_run_reports_the_resources_it_finished_with() {
        let directory = checkpoint_directory("closed");
        let settings = settings_at(directory.join("run.ckpt"));

        let solved = run_mpx6(&checkpoint_limits(CHECKPOINT_TOTAL_TRIALS), Some(&settings));
        assert_eq!(solved.verdict, Verdict::Success);

        // Forced below anything this process could be using, so reporting the live peak
        // instead of the saved one cannot pass by accident.
        let mut stored = checkpoint::read::<7>(&settings.path);
        stored.peak_rss_bytes = 4096;
        checkpoint::write(&settings.path, &stored);

        let reopened = run_mpx6(&checkpoint_limits(CHECKPOINT_TOTAL_TRIALS), Some(&settings));

        assert!(reopened.already_finished);
        assert_eq!(reopened.wall_seconds.to_bits(), solved.wall_seconds.to_bits());
        assert_eq!(reopened.peak_rss_bytes, 4096);
        assert_eq!(reopened.peak_macro_population, solved.peak_macro_population);
        assert_eq!(reopened.trials_used, solved.trials_used);

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The evaluation interval does not change what the agent learns, only which trials
    /// can be observed -- which is the number this project reports. Splicing two
    /// sampling rates into one run takes an explicit decision.
    #[test]
    fn resuming_at_another_evaluation_interval_takes_an_explicit_decision() {
        let directory = checkpoint_directory("evalrate");
        let settings = settings_at(directory.join("run.ckpt"));
        run_mpx6(&checkpoint_limits(CHECKPOINT_SPLIT_TRIALS), Some(&settings));

        let mut faster = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
        faster.eval_interval = CHECKPOINT_EVAL_INTERVAL * 2;
        assert!(
            panics(|| {
                run_mpx6(&faster, Some(&settings));
            }),
            "a changed evaluation interval must not pass unremarked"
        );

        let allowed = CheckpointSettings {
            allow_eval_interval_change: true,
            ..settings_at(settings.path.clone())
        };
        assert!(
            !panics(|| {
                run_mpx6(&faster, Some(&allowed));
            }),
            "the override must let a deliberate change through"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The identity gates the learning configuration, not just the flags that set it: a
    /// later executable could change a threshold and still meet a hand-listed subset.
    #[test]
    fn a_checkpoint_refuses_a_changed_learning_configuration() {
        let directory = checkpoint_directory("config");
        let settings = settings_at(directory.join("run.ckpt"));
        run_mpx6(&checkpoint_limits(CHECKPOINT_SPLIT_TRIALS), Some(&settings));

        let limits = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
        let resume_with = |gen: GenConfig| {
            run_reach_repeat::<7>(
                CHECKPOINT_SIZE,
                CHECKPOINT_SEED,
                gen,
                AgentOptions::default(),
                &limits,
                Some(&settings),
            );
        };

        assert!(
            panics(|| resume_with(GenConfig { u_max: 5, ..mpx6_gen() })),
            "a changed generalization limit must not be resumed silently"
        );
        assert!(
            panics(|| resume_with(GenConfig { do_ga: false, ..mpx6_gen() })),
            "a changed GA setting must not be resumed silently"
        );
        assert!(
            panics(|| {
                let mut other = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
                other.epsilon = 1.0;
                run_mpx6(&other, Some(&settings));
            }),
            "a changed exploration rate must not be resumed silently"
        );
        assert!(
            !panics(|| resume_with(mpx6_gen())),
            "the configuration it was written for must still resume"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The identity must state the learning configuration, not the flags that set it.
    /// Nothing on the command line reaches `beta`, so a later executable could change it
    /// and still satisfy a hand-listed subset of fields.
    #[test]
    fn the_identity_gates_learning_constants_no_flag_can_reach() {
        let limits = checkpoint_limits(CHECKPOINT_TOTAL_TRIALS);
        let mut config = Configuration::mpx();
        config.u_max = 4;
        let baseline = checkpoint_identity(
            CHECKPOINT_SIZE,
            CHECKPOINT_SEED,
            &config,
            AgentOptions::default(),
            &limits,
        );

        for altered in [
            Configuration { beta: 0.06, ..config.clone() },
            Configuration { theta_r: 0.95, ..config.clone() },
            Configuration { theta_ga: 50, ..config.clone() },
            Configuration { do_subsumption: false, ..config.clone() },
        ] {
            assert_ne!(
                checkpoint_identity(
                    CHECKPOINT_SIZE,
                    CHECKPOINT_SEED,
                    &altered,
                    AgentOptions::default(),
                    &limits,
                ),
                baseline,
                "a changed learning constant must change the identity"
            );
        }

        assert_eq!(
            checkpoint_identity(
                CHECKPOINT_SIZE,
                CHECKPOINT_SEED,
                &config,
                AgentOptions::default(),
                &limits,
            ),
            baseline,
            "the identity must be stable for an unchanged configuration"
        );
    }

    const KILL_SIZE: usize = 20;
    const KILL_EVAL_INTERVAL: u64 = 1000;
    const KILL_RESUME_AT: u64 = 8000;
    const KILL_TRIALS: u64 = 20_000;
    const KILL_TEST: &str = "regression::a_periodic_checkpoint_outlives_a_killed_process";

    fn kill_limits(trials_cap: u64) -> ReachLimits {
        ReachLimits {
            eval_interval: KILL_EVAL_INTERVAL,
            ..checkpoint_limits(trials_cap)
        }
    }

    fn kill_worker(mode: &str) {
        let directory = PathBuf::from(std::env::var_os("ACS2_CHECKPOINT_DIR").unwrap());
        let path = directory.join("killed.ckpt");
        let periodic = CheckpointSettings { every: 2000, ..settings_at(path) };
        // The victim gets a cap it cannot reach before the parent kills it, so the file
        // the resume reads is a periodic save with no verdict recorded.
        let (settings, trials_cap) = match mode {
            "whole" => (None, KILL_TRIALS),
            "killable" => (Some(periodic), u64::MAX),
            "resume" => (Some(periodic), KILL_TRIALS),
            other => panic!("unknown kill worker mode {other}"),
        };
        run_reach_repeat::<21>(
            KILL_SIZE,
            CHECKPOINT_SEED,
            GenConfig {
                do_ga: true,
                u_max: derived_u_max(KILL_SIZE, AlpGenVariant::Pyalcs),
                alp_gen_variant: AlpGenVariant::Pyalcs,
            },
            AgentOptions::default(),
            &kill_limits(trials_cap),
            settings.as_ref(),
        );
    }

    fn run_kill_worker(mode: &str, directory: &Path) -> Vec<String> {
        run_worker_process("ACS2_CHECKPOINT_KILL", mode, KILL_TEST, directory)
    }

    fn checkpoint_trials(path: &Path) -> Option<u64> {
        let text = std::fs::read_to_string(path).ok()?;
        let run = text.lines().find(|line| line.starts_with("run "))?;
        run.split_whitespace()
            .find_map(|field| field.strip_prefix("trials=")?.parse().ok())
    }

    fn measurements_after(lines: &[String], trial: u64) -> Vec<String> {
        lines
            .iter()
            .filter(|line| {
                line.split_whitespace()
                    .find_map(|field| field.strip_prefix("trials=")?.parse::<u64>().ok())
                    .is_some_and(|trials| trials > trial)
            })
            .cloned()
            .collect()
    }

    /// A node failure kills a job outright: no verdict is written and the last periodic
    /// save is all that survives. Over a 504 h job that is the disaster path, and every
    /// other test here resumes from a clean stop instead.
    #[test]
    fn a_periodic_checkpoint_outlives_a_killed_process() {
        if let Some(mode) = std::env::var_os("ACS2_CHECKPOINT_KILL") {
            kill_worker(&mode.to_string_lossy());
            return;
        }

        let directory = checkpoint_directory("kill");
        let path = directory.join("killed.ckpt");
        let uninterrupted = run_kill_worker("whole", &directory);

        let mut victim = worker_command("ACS2_CHECKPOINT_KILL", "killable", KILL_TEST, &directory)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(120);
        let saved_at = loop {
            if let Some(trials) = checkpoint_trials(&path) {
                if trials >= KILL_RESUME_AT {
                    break trials;
                }
            }
            assert!(Instant::now() < deadline, "no periodic checkpoint was ever written");
            std::thread::sleep(Duration::from_millis(10));
        };
        victim.kill().unwrap();
        victim.wait().unwrap();
        let _ = saved_at;
        // Read the surviving position *after* the kill: another checkpoint can publish
        // between the poll that saw one and the signal that lands.
        let saved_at = checkpoint_trials(&path).expect("the published checkpoint must be whole");

        let survivor = std::fs::read_to_string(&path).unwrap();
        assert!(
            survivor.contains("verdict=-"),
            "the fixture must kill the run before it records a verdict"
        );
        // A kill during the staged write legitimately leaves the staging file; what must
        // hold is that the *published* checkpoint is complete.
        assert_eq!(checkpoint::parse::<21>(&survivor).trials_used, saved_at);

        let resumed = run_kill_worker("resume", &directory);
        let expected = measurements_after(&uninterrupted, saved_at);
        assert!(!expected.is_empty(), "nothing left to compare after trial {saved_at}");
        assert_eq!(
            resumed, expected,
            "the whole resumed trajectory must be the uninterrupted one's tail -- filtering \
             the resumed side too would let a run that restarted from zero pass"
        );

        std::fs::remove_dir_all(&directory).ok();
    }

    /// The archive's only record of a success can be a reopening job's line: the job that
    /// closed the run writes its checkpoint before it prints, and a kill in between leaves
    /// the verdict nowhere else. The Python side pins that the parser reads such a line;
    /// this pins that the binary still writes one.
    #[test]
    fn a_reopened_run_restates_its_verdict_on_stdout() {
        let directory = checkpoint_directory("reopen");
        let path = directory.join("run.ckpt");
        let run = |cap: &str| {
            let output = Command::new(env!("CARGO_BIN_EXE_mpx_reach"))
                .args([
                    "--sizes", "20", "--n-exp", "1", "--seed", "42", "--u-max", "derived",
                    "--eval-interval", "500", "--time-cap-secs", cap,
                    "--checkpoint-path", path.to_str().unwrap(),
                ])
                .output()
                .unwrap();
            assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
            String::from_utf8(output.stdout).unwrap()
        };

        let stopped = run("0");
        assert!(stopped.contains("repeat 0: TIME-LIMITED"));
        assert!(!stopped.contains("reopened=true"));

        // Stand in for the job that saved a SUCCESS and was killed before printing it.
        let mut stored = checkpoint::read::<21>(&path);
        stored.verdict = Some("SUCCESS".to_string());
        checkpoint::write(&path, &stored);

        let reopened = run("600");
        let verdict = reopened
            .lines()
            .find(|line| line.contains("repeat 0:"))
            .expect("a reopened run must still print a verdict line");
        assert!(verdict.contains("SUCCESS"), "{verdict}");
        assert!(verdict.contains("reopened=true"), "{verdict}");

        std::fs::remove_dir_all(&directory).ok();
    }
}
