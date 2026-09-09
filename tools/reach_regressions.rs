include!("../acs2-bench/src/bin/mpx_reach.rs");

#[cfg(test)]
mod regression {
    use super::*;
    use std::cell::Cell;
    use acs2_core::action_selection::ActionSelector;
    use acs2_core::environment::Environment;
    use acs2_core::rl::BootstrapEstimator;
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
            &EpsilonGreedy { number_of_possible_actions: 2, epsilon: 0.8 }, 6, limits)
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
}
