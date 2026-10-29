use acs2_core::ports::goal::{ExactMatch, Goal, GoalStart, GoalStep};
use acs2_core::representation::{perception::Perception, symbol::Symbol};
use acs2_core::support::rng::{ChaChaRandomSource, RandomSource};
use acs2_core::trial::TruncationMode;
use acs2_trajectory::*;

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}

fn store(paths: &[&[u8]]) -> TrajectoryStore<1, 1> {
    let mut store = TrajectoryStore::new(100);
    for path in paths {
        let mut episode = store.begin_episode(GoalStart {
            observation: Perception::new([Symbol::Token(path[0])]),
            achieved: goal(path[0]),
            desired: goal(9),
        });
        for (index, value) in path[1..].iter().enumerate() {
            episode.push(
                index,
                GoalStep {
                    observation: Perception::new([Symbol::Token(*value)]),
                    achieved: goal(*value),
                    terminal_state: false,
                    time_limit_reached: index + 2 == path.len(),
                },
            );
        }
        store.insert(episode).unwrap();
    }
    store
}

fn configuration(rule: Admissibility, p: f64) -> SamplerConfiguration {
    SamplerConfiguration {
        selection: Selection {
            strategy: GoalStrategy::Future,
            admissibility: rule,
            candidate_filter: false,
        },
        relabeled_proportion: p,
        truncation: TruncationMode::Bootstrap,
    }
}

#[test]
fn episode_passes_refuse_from_non_goal_state_before_sampling() {
    let store = store(&[&[0, 1, 0]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let relabeled = relabel_episode::<_, _, 2>(
        store.episodes().next().unwrap(),
        &goal(0),
        Admissibility::FromNonGoalState,
        &evaluator,
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert_eq!(
        relabeled
            .samples
            .iter()
            .map(|sample| sample.transition)
            .collect::<Vec<_>>(),
        vec![1]
    );
    for strategy in [
        GoalStrategy::Original,
        GoalStrategy::Future,
        GoalStrategy::Episode,
    ] {
        for proportion in [0.0, 2.0 / 3.0, 1.0] {
            let mut config = configuration(Admissibility::FromNonGoalState, proportion);
            config.selection.strategy = strategy;
            let mut sampler = Sampler::new(config, ChaChaRandomSource::from_seed(42));
            let rng_before = sampler.random_source().capture_state();
            let error = match sampler.draw_pass::<_, _, 2>(&store, 100, &evaluator) {
                Err(error) => error,
                Ok(_) => panic!("FromNonGoalState episode passes must be refused"),
            };
            assert_eq!(error, SampleError::UnsupportedPassAdmissibility);
            assert_eq!(error.to_string(), "episode passes support only EveryTransition (every transition) and CounterfactualEpisode (counterfactual episode); FromNonGoalState is unsupported");
            assert_eq!(sampler.random_source().capture_state(), rng_before);
            assert_eq!(sampler.counters(), &ReplayCounters::default());
        }
    }
}

#[test]
fn every_transition_passes_keep_start_goals_and_return_transitions() {
    let store = store(&[&[0, 1, 0]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let mut sampler = Sampler::new(configuration(Admissibility::EveryTransition, 1.0), Fixed);
    let pass = sampler
        .draw_pass::<_, _, 2>(&store, 100, &evaluator)
        .unwrap();
    assert_eq!(pass.available_steps, 2);
    assert_eq!(
        pass.samples
            .iter()
            .map(|sample| sample.transition)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert!(pass.samples.iter().all(
        |sample| sample.goal == goal(0) && sample.provenance.origin == SampleOrigin::Relabeled
    ));
    assert_eq!(sampler.counters().relabeled.draws, 2);
    assert_eq!(sampler.counters().relabeled.already_reached, 1);
    assert_eq!(sampler.counters().relabeled.after_counterfactual_end, 2);
}

#[test]
fn passes_pick_retained_episodes_uniformly_and_reverse_every_transition() {
    let store = store(&[&[0, 1], &[0, 2, 3, 4]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let mut sampler = Sampler::new(
        configuration(Admissibility::EveryTransition, 0.0),
        ChaChaRandomSource::from_seed_and_stream(42, 7),
    );
    let mut selections = [0_u64; 2];
    for _ in 0..12_000 {
        let pass = sampler
            .draw_pass::<_, _, 2>(&store, 100, &evaluator)
            .unwrap();
        let first = &pass.samples[0];
        selections[first.episode.0 as usize] += 1;
        let length = if first.episode.0 == 0 { 1 } else { 3 };
        assert_eq!(
            pass.samples
                .iter()
                .map(|draw| draw.transition)
                .collect::<Vec<_>>(),
            (0..length).rev().collect::<Vec<_>>()
        );
        assert!(pass.samples.iter().all(|draw| draw.goal == goal(9)));
    }
    assert!((selections[1] as f64 / 12_000.0 - 0.5).abs() < 0.02);
}

#[test]
fn original_passes_give_every_retained_transition_equal_expected_updates() {
    struct Enumerated(usize);
    impl RandomSource for Enumerated {
        fn gen_range(&mut self, high: usize) -> usize {
            let selected = self.0 % high;
            self.0 += 1;
            selected
        }
        fn gen_bool(&mut self, _: f64) -> bool {
            panic!("original-goal passes do not draw a goal coin")
        }
        fn gen_unit(&mut self) -> f64 {
            panic!("original-goal passes only select an episode")
        }
    }
    let store = store(&[&[0, 1], &[0, 2, 3, 4]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let mut sampler = Sampler::new(
        configuration(Admissibility::EveryTransition, 0.0),
        Enumerated(0),
    );
    let mut updates = [[0; 3]; 2];
    for _ in 0..12 {
        let pass = sampler
            .draw_pass::<_, _, 2>(&store, 100, &evaluator)
            .unwrap();
        for sample in pass.samples {
            assert_eq!(sample.goal, goal(9));
            updates[sample.episode.0 as usize][sample.transition] += 1;
        }
    }
    assert_eq!(updates, [[6, 0, 0], [6, 6, 6]]);
    assert_eq!(sampler.counters().pooled().draws, 24);
}

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

#[test]
fn one_episode_goal_keeps_duplicate_weights_and_stops_at_first_arrival() {
    let store = store(&[&[0, 1, 2, 1, 3]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let mut sampler = Sampler::new(
        configuration(Admissibility::CounterfactualEpisode, 1.0),
        ChaChaRandomSource::from_seed(42),
    );
    let mut goals = [0; 4];
    for _ in 0..8_000 {
        let pass = sampler
            .draw_pass::<_, _, 2>(&store, 100, &evaluator)
            .unwrap();
        let selected = pass.samples[0].goal;
        let first_arrival = (1..=4)
            .find(|&index| *store.episodes().next().unwrap().achieved(index) == selected)
            .unwrap();
        assert_eq!(pass.available_steps, first_arrival);
        assert_eq!(
            pass.samples
                .iter()
                .map(|draw| draw.transition)
                .collect::<Vec<_>>(),
            (0..first_arrival).rev().collect::<Vec<_>>()
        );
        assert!(pass.samples.iter().all(|draw| draw.goal == selected
            && !draw.provenance.already_reached
            && !draw.provenance.after_counterfactual_end));
        goals[match selected {
            value if value == goal(1) => 1,
            value if value == goal(2) => 2,
            _ => 3,
        }] += 1;
    }
    assert!((goals[1] as f64 / 8_000.0 - 0.5).abs() < 0.025);
    assert!((goals[2] as f64 / 8_000.0 - 0.25).abs() < 0.025);
    assert_eq!(sampler.counters().relabeled.already_reached, 0);
    assert_eq!(sampler.counters().relabeled.after_counterfactual_end, 0);
}

#[test]
fn pass_cuts_are_backward_suffixes_and_empty_goals_fall_back_without_losing_updates() {
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let store = store(&[&[0, 1, 2, 3, 4]]);
    let mut sampler = Sampler::new(configuration(Admissibility::EveryTransition, 0.0), Fixed);
    let pass = sampler.draw_pass::<_, _, 2>(&store, 2, &evaluator).unwrap();
    assert_eq!(pass.available_steps, 4);
    assert_eq!(
        pass.samples
            .iter()
            .map(|draw| draw.transition)
            .collect::<Vec<_>>(),
        vec![3, 2]
    );
    assert_eq!(sampler.counters().pooled().draws, 2);
    let stationary = self::store(&[&[0, 0, 0]]);
    let mut sampler = Sampler::new(
        configuration(Admissibility::CounterfactualEpisode, 1.0),
        Fixed,
    );
    let pass = sampler
        .draw_pass::<_, _, 2>(&stationary, 1, &evaluator)
        .unwrap();
    assert!(pass.fallback && pass.requested_relabel);
    assert_eq!(pass.samples.len(), 1);
    assert_eq!(pass.samples[0].goal, goal(9));
    assert_eq!(sampler.counters().fallbacks, 1);
    assert_eq!(sampler.counters().original.draws, 1);
}

#[test]
fn pass_coin_is_per_pass_and_preserves_requested_goal_probability() {
    let store = store(&[&[0, 1, 2, 3, 4]]);
    let evaluator = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let mut sampler = Sampler::new(
        configuration(Admissibility::CounterfactualEpisode, 0.4),
        ChaChaRandomSource::from_seed(42),
    );
    let mut requested = 0;
    for _ in 0..10_000 {
        let pass = sampler
            .draw_pass::<_, _, 2>(&store, 100, &evaluator)
            .unwrap();
        requested += usize::from(pass.requested_relabel);
        assert!(pass
            .samples
            .iter()
            .all(|sample| sample.goal == pass.samples[0].goal));
    }
    assert!((requested as f64 / 10_000.0 - 0.4).abs() < 0.02);
}

#[test]
fn pass_costs_count_every_objective_call_and_failures_record_work_without_draws() {
    use acs2_core::ports::goal::GoalOutcome;
    use std::cell::Cell;
    struct Counted {
        rewards: Cell<u64>,
        reaches: Cell<u64>,
        fail: bool,
    }
    impl GoalEvaluator<1, 1> for Counted {
        fn outcome(&self, step: &GoalStep<1, 1>, desired: &Goal<1>) -> GoalOutcome {
            self.rewards.set(self.rewards.get() + 1);
            self.reaches
                .set(self.reaches.get() + u64::from(!step.terminal_state));
            let mut result = step.outcome(
                &ExactMatch {
                    reward_on_reach: 1000.0,
                },
                desired,
            );
            if self.fail {
                result.reward = -1.0;
            }
            result
        }
        fn is_reached(&self, achieved: &Goal<1>, desired: &Goal<1>) -> bool {
            self.reaches.set(self.reaches.get() + 1);
            achieved == desired
        }
    }
    let store = store(&[&[0, 1, 2, 1, 3]]);
    for fail in [false, true] {
        let objective = Counted {
            rewards: Cell::new(0),
            reaches: Cell::new(0),
            fail,
        };
        let mut sampler = Sampler::new(
            configuration(Admissibility::CounterfactualEpisode, 1.0),
            Fixed,
        );
        let result = sampler.draw_pass::<_, _, 2>(&store, 1, &objective);
        if fail {
            assert!(matches!(result, Err(SampleError::NegativeReward)));
        } else {
            assert_eq!(result.unwrap().samples.len(), 1);
        }
        let counts = sampler.counters();
        let cost = counts.cost.total();
        assert_eq!(cost.reward_evaluations, objective.rewards.get());
        assert_eq!(cost.reach_evaluations, objective.reaches.get());
        assert_eq!(cost.reward_evaluations, 1);
        assert_eq!(counts.failed_draws, u64::from(fail));
        assert_eq!(counts.pooled().draws, u64::from(!fail));
    }
}
