use super::*;
use acs2_core::ports::environment::StepOutcome;
use acs2_core::ports::goal::ExactMatch;
use acs2_core::representation::symbol::Symbol;
use acs2_trajectory::{Admissibility, GoalStrategy, ObjectiveEvaluator, Selection};

struct Fixed;
impl RandomSource for Fixed {
    fn gen_bool(&mut self, _: f64) -> bool {
        true
    }
    fn gen_range(&mut self, _: usize) -> usize {
        0
    }
    fn gen_unit(&mut self) -> f64 {
        0.0
    }
}

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}
fn observation(value: u8) -> Perception<1> {
    Perception::new([Symbol::Token(value)])
}

struct Route {
    length: u8,
    position: u8,
}
impl GoalEvaluator<1, 1> for Route {
    fn outcome(
        &self,
        step: &GoalStep<1, 1>,
        desired: &Goal<1>,
    ) -> acs2_core::ports::goal::GoalOutcome {
        step.outcome(
            &ExactMatch {
                reward_on_reach: 1000.0,
            },
            desired,
        )
    }
    fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
        achieved == desired
    }
}
impl Environment<2> for Route {
    fn reset(&mut self) -> Perception<2> {
        self.position = 0;
        GoalLayout::<1, 1, 2>::join(&observation(0), &goal(self.length))
    }
    fn step(&mut self, _: usize) -> StepOutcome<2> {
        self.position += 1;
        let outcome = self.outcome(&self.last_step(), &goal(self.length));
        StepOutcome {
            observation: GoalLayout::<1, 1, 2>::join(
                &observation(self.position),
                &goal(self.length),
            ),
            reward: outcome.reward,
            terminated: outcome.terminated,
            truncated: outcome.truncated,
            info: Default::default(),
        }
    }
}
impl ReplayEnvironment<1, 1, 2> for Route {
    fn episode_start(&self) -> GoalStart<1, 1> {
        GoalStart {
            observation: observation(0),
            achieved: goal(0),
            desired: goal(self.length),
        }
    }
    fn last_step(&self) -> GoalStep<1, 1> {
        GoalStep {
            observation: observation(self.position),
            achieved: goal(self.position),
            terminal_state: false,
            time_limit_reached: self.position == self.length,
        }
    }
}

fn learner(plan: ReplayPlan) -> EpisodicReplay<1, 1, 2, Fixed, Fixed> {
    EpisodicReplay::new(
        Configuration {
            number_of_possible_actions: 1,
            beta: 0.05,
            ..Configuration::default_protocol()
        },
        EpisodicConfiguration {
            capacity: 100,
            warmup: 1,
            updates_per_step: 3,
            truncation: TruncationMode::Bootstrap,
        },
        SamplerConfiguration {
            selection: Selection {
                strategy: GoalStrategy::Future,
                admissibility: Admissibility::CounterfactualEpisode,
                candidate_filter: false,
            },
            relabeled_proportion: 0.0,
            truncation: TruncationMode::Bootstrap,
        },
        CopyUpdate::Full,
        Fixed,
        Fixed,
    )
    .with_replay_plan(
        plan,
        (plan.order == ReplayOrder::RandomPass).then_some(Fixed),
    )
}

#[test]
fn end_clock_schedule_keeps_the_tie_early_and_charges_a_straddling_episode_entirely_late() {
    for threshold in [3, 4] {
        let mut agent = learner(ReplayPlan {
            schedule: ReplaySchedule::ThreeToOne {
                switch_steps: threshold,
            },
            order: ReplayOrder::Independent,
        });
        let mut route = Route {
            length: 3,
            position: 0,
        };
        agent.train_episode(&mut route, 0);
        assert_eq!(agent.counts().full, 9);
        agent.train_episode(&mut route, 3);
        assert_eq!(agent.counts().full, 12);
        assert_eq!(
            agent.schedule_counts(),
            ScheduleCounts {
                early_steps: 3,
                late_steps: 3,
                first_late_episode_end: Some(6)
            }
        );
        agent.train_episode(&mut route, 6);
        assert_eq!(agent.counts().full, 15);
    }
}

#[test]
fn backward_and_random_pass_learning_follow_the_expected_distinct_full_update_sequences() {
    for order in [ReplayOrder::Backward, ReplayOrder::RandomPass] {
        let plan = ReplayPlan {
            schedule: ReplaySchedule::Constant,
            order,
        };
        let mut actual = learner(plan);
        actual.train_episode(
            &mut Route {
                length: 3,
                position: 0,
            },
            0,
        );
        let mut expected = learner(ReplayPlan::default());
        let mut sampler = Sampler::new(actual.sampler.configuration(), Fixed);
        let objective = ObjectiveEvaluator(&ExactMatch {
            reward_on_reach: 1000.0,
        });
        for _ in 0..3 {
            let pass = sampler
                .draw_pass::<_, _, 2>(&actual.store, 3, &objective)
                .unwrap();
            let samples = pass.samples;
            let indices: &[usize] = if order == ReplayOrder::Backward {
                &[0, 1, 2]
            } else {
                &[1, 2, 0]
            };
            for &index in indices {
                expected.learn_draw(&samples[index], 3);
            }
        }
        assert_eq!(
            format!("{:?}", actual.population.classifiers()),
            format!("{:?}", expected.population.classifiers())
        );
        assert_eq!(actual.counts.full, 9);
        assert_eq!(actual.passes.passes, 3);
        assert_eq!(actual.passes.emitted_steps, 9);
    }
    let mut backward = learner(ReplayPlan {
        order: ReplayOrder::Backward,
        ..ReplayPlan::default()
    });
    let mut random = learner(ReplayPlan {
        order: ReplayOrder::RandomPass,
        ..ReplayPlan::default()
    });
    backward.train_episode(
        &mut Route {
            length: 3,
            position: 0,
        },
        0,
    );
    random.train_episode(
        &mut Route {
            length: 3,
            position: 0,
        },
        0,
    );
    assert_ne!(
        format!("{:?}", backward.population.classifiers()),
        format!("{:?}", random.population.classifiers())
    );
}

#[test]
fn cut_passes_keep_the_exact_episode_budget_even_when_stored_lengths_differ() {
    let mut agent = learner(ReplayPlan {
        order: ReplayOrder::Backward,
        ..ReplayPlan::default()
    });
    agent.train_episode(
        &mut Route {
            length: 4,
            position: 0,
        },
        0,
    );
    agent.train_episode(
        &mut Route {
            length: 2,
            position: 0,
        },
        4,
    );
    agent.train_episode(
        &mut Route {
            length: 1,
            position: 0,
        },
        6,
    );
    assert_eq!(agent.counts.full, 21);
    assert_eq!(agent.sampler.counters().pooled().draws, 21);
    assert_eq!(agent.passes.emitted_steps, 21);
    assert_eq!(agent.passes.cut_passes, 2);
}

#[test]
fn shuffling_does_not_change_pass_goals_transitions_cuts_or_the_sampler_stream() {
    use acs2_core::support::rng::ChaChaRandomSource;
    let make = |order| {
        let base = learner(ReplayPlan::default());
        let mut sampler = base.sampler.configuration();
        sampler.relabeled_proportion = 0.4;
        EpisodicReplay::<1, 1, 2, _, _>::new(
            base.config,
            base.replay,
            sampler,
            CopyUpdate::Full,
            Fixed,
            ChaChaRandomSource::from_seed_and_stream(42, 7),
        )
        .with_replay_plan(
            ReplayPlan {
                order,
                ..ReplayPlan::default()
            },
            (order == ReplayOrder::RandomPass)
                .then(|| ChaChaRandomSource::from_seed_and_stream(42, ORDER_STREAM)),
        )
    };
    let mut backward = make(ReplayOrder::Backward);
    let mut random = make(ReplayOrder::RandomPass);
    let mut time = 0;
    for length in [5, 2, 1, 3, 4, 1, 2, 3, 5, 2] {
        backward.train_episode(
            &mut Route {
                length,
                position: 0,
            },
            time,
        );
        random.train_episode(
            &mut Route {
                length,
                position: 0,
            },
            time,
        );
        time += u64::from(length);
        assert_eq!(backward.store, random.store);
        assert_eq!(backward.sampler.counters(), random.sampler.counters());
        assert_eq!(backward.passes, random.passes);
        assert_eq!(
            backward.sampler.random_source().capture_state(),
            random.sampler.random_source().capture_state()
        );
        assert_eq!(backward.counts.full, 3 * time);
        assert_eq!(random.counts.full, 3 * time);
    }
    assert!(backward.passes.cut_passes > 0);
    assert!(backward.sampler.counters().relabeled.draws > 0);
    assert_ne!(
        format!("{:?}", backward.population.classifiers()),
        format!("{:?}", random.population.classifiers())
    );
}
