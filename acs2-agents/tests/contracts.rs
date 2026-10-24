use acs2_agents::episodic_replay::{
    values_learning_step, CopyUpdate, EpisodicAgent, EpisodicConfiguration, EpisodicReplay,
    ReplayEnvironment,
};
use acs2_agents::acs2er::episodic;
use acs2_agents::acs2her;
use acs2_core::learning::replay::ReplaySample;
use acs2_core::trial::TruncationMode;
use acs2_core::ports::environment::{Environment, StepOutcome};
use acs2_core::ports::goal::{ExactMatch, Goal, GoalStart, GoalStep};
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::{
    config::Configuration,
    rng::{ChaChaRandomSource, RandomSource},
};
use acs2_trajectory::{
    Admissibility, Episode, GoalEvaluator, GoalStrategy, ObjectiveEvaluator, Sampler,
    SamplerConfiguration, Selection, TrajectoryStore,
};

fn p<const N: usize>(values: [u8; N]) -> Perception<N> {
    Perception {
        symbols: values.map(Symbol::Token),
    }
}
fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}
fn config() -> Configuration {
    Configuration {
        number_of_possible_actions: 2,
        epsilon: 0.0,
        do_ga: false,
        ..Configuration::default_protocol()
    }
}
fn replay(warmup: usize) -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity: 10,
        warmup,
        updates_per_step: 3,
        truncation: TruncationMode::Bootstrap,
    }
}
fn rng(stream: u64) -> ChaChaRandomSource {
    ChaChaRandomSource::from_seed_and_stream(42, stream)
}
fn sampler(proportion: f64, truncation: TruncationMode) -> SamplerConfiguration {
    SamplerConfiguration {
        selection: Selection {
            strategy: GoalStrategy::Future,
            admissibility: Admissibility::EveryTransition,
            candidate_filter: false,
        },
        relabeled_proportion: proportion,
        truncation,
    }
}

struct Steps {
    elapsed: u8,
    cap: u8,
    desired: u8,
    actions: Vec<usize>,
    resets: usize,
}
impl Steps {
    fn new(cap: u8, desired: u8) -> Self {
        Self {
            elapsed: 0,
            cap,
            desired,
            actions: vec![],
            resets: 0,
        }
    }
    fn raw(&self) -> GoalStep<1, 1> {
        GoalStep {
            observation: p([self.elapsed]),
            achieved: goal(self.elapsed),
            terminal_state: false,
            time_limit_reached: self.elapsed == self.cap,
        }
    }
}
impl Environment<2> for Steps {
    fn reset(&mut self) -> Perception<2> {
        self.elapsed = 0;
        self.resets += 1;
        p([0, self.desired])
    }
    fn step(&mut self, action: usize) -> StepOutcome<2> {
        self.elapsed += 1;
        self.actions.push(action);
        let raw = self.raw();
        let outcome = self.outcome(&raw, &goal(self.desired));
        StepOutcome {
            observation: p([self.elapsed, self.desired]),
            reward: outcome.reward,
            terminated: outcome.terminated,
            truncated: outcome.truncated,
            info: (),
        }
    }
}
impl GoalEvaluator<1, 1> for Steps {
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
impl ReplayEnvironment<1, 1, 2> for Steps {
    fn episode_start(&self) -> GoalStart<1, 1> {
        GoalStart {
            observation: p([0]),
            achieved: goal(0),
            desired: goal(self.desired),
        }
    }
    fn last_step(&self) -> GoalStep<1, 1> {
        self.raw()
    }
}

fn conditioned_population() -> Population<2> {
    let mut rules = vec![];
    for desired in [7, 9] {
        for action in [0, 1] {
            let mut cl = Classifier::general(Some(action), &config());
            cl.condition.symbols[1] = Symbol::Token(desired);
            cl.effect.symbols[0] = Symbol::Token(1);
            cl.r = if (desired == 7 && action == 0) || (desired == 9 && action == 1) {
                100.0
            } else {
                1.0
            };
            rules.push(cl);
        }
    }
    Population::from_classifiers(rules)
}

#[test]
fn exploration_and_greedy_selection_condition_on_the_entire_real_goal() {
    let mut agent = EpisodicReplay::<1, 1, 2, _, _>::new(
        config(),
        replay(1),
        sampler(1.0, TruncationMode::Bootstrap),
        CopyUpdate::ValuesOnly,
        rng(1),
        rng(7),
    )
    .with_population(conditioned_population());
    assert_eq!(agent.explore_action(&p([0, 7])), 0);
    assert_eq!(agent.explore_action(&p([0, 9])), 1);
    assert_eq!(agent.greedy_action(&p([0, 7]), &mut rng(3)).0, 0);
    assert_eq!(agent.greedy_action(&p([0, 9]), &mut rng(3)).0, 1);
    let mut env = Steps::new(3, 9);
    agent.train_episode(&mut env, 0);
    assert_eq!(env.actions, [1, 1, 1]);
    assert_eq!(env.resets, 1);
    assert_eq!(
        agent.store().episodes().next().unwrap().start().desired,
        goal(9)
    );
}

#[test]
fn warmup_is_checked_after_whole_episode_insertion_without_backpay() {
    let mut agent = episodic::Agent::<1, 1, 2, _, _>::new(config(), replay(4), rng(1), rng(7));
    let mut env = Steps::new(2, 9);
    agent.learner_mut().train_episode(&mut env, 0);
    assert_eq!(agent.learner().counts().full, 0);
    assert_eq!(agent.learner().counts().skipped_warmup_steps, 2);
    assert!(agent.learner().population().is_empty());
    agent.learner_mut().train_episode(&mut env, 2);
    assert_eq!(agent.learner().counts().full, 6);
    assert_eq!(agent.learner().sampler().counters().original.draws, 6);
    agent.learner_mut().train_episode(&mut env, 4);
    assert_eq!(agent.learner().counts().full, 12);
    assert_eq!(agent.learner().counts().environment_steps, 6);
}

#[test]
fn acs2er_episodic_never_relabels_and_replays_with_replacement_below_draw_count() {
    let mut agent = episodic::Agent::<1, 1, 2, _, _>::new(config(), replay(1), rng(1), rng(7));
    agent.learner_mut().train_episode(&mut Steps::new(1, 1), 0);
    let learner = agent.learner();
    assert_eq!(learner.store().len(), 1);
    assert_eq!(learner.counts().full, 3);
    assert_eq!(learner.counts().values_only, 0);
    assert_eq!(learner.sampler().counters().original.draws, 3);
    assert_eq!(learner.sampler().counters().relabeled.draws, 0);
    assert_eq!(learner.sampler().counters().original.reward, 3000.0);
    assert_eq!(learner.sampler().counters().original.done, 3);
}

#[test]
fn acs2her_presets_pin_copy_updates_goal_admissibility_and_proportions() {
    for variant in [
        acs2her::Preset::Acs2Her,
        acs2her::Preset::Acs2HerValues { proportion: 0.9 },
        acs2her::Preset::Acs2HerValuesShortened {
            proportion: 1.0 / 3.0,
        },
    ] {
        let agent = acs2her::Agent::<1, 1, 2, _, _>::new(config(), replay(1), variant, rng(1), rng(7));
        let learner = agent.learner();
        let sampler = learner.sampler().configuration();
        assert_eq!(
            sampler.relabeled_proportion.to_bits(),
            match variant {
                acs2her::Preset::Acs2Her | acs2her::Preset::Acs2HerNontrivial | acs2her::Preset::Acs2HerShortened => (2.0_f64 / 3.0).to_bits(),
                acs2her::Preset::Acs2HerValues { proportion }
                | acs2her::Preset::Acs2HerValuesShortened { proportion } => proportion.to_bits(),
            }
        );
        assert_eq!(
            sampler.truncation,
            learner.replay_configuration().truncation
        );
        assert_eq!(
            sampler.selection.admissibility,
            if matches!(variant, acs2her::Preset::Acs2HerValuesShortened { .. }) {
                Admissibility::CounterfactualEpisode
            } else {
                Admissibility::EveryTransition
            }
        );
        assert_eq!(
            learner.copy_update(),
            if variant == acs2her::Preset::Acs2Her {
                CopyUpdate::Full
            } else {
                CopyUpdate::ValuesOnly
            }
        );
    }
}

#[test]
#[should_panic(expected = "sampler and agent truncation must agree")]
fn a_sampler_truncation_different_from_the_agent_is_refused() {
    EpisodicReplay::<1, 1, 2, _, _>::new(
        config(),
        replay(1),
        sampler(1.0, TruncationMode::Pyalcs),
        CopyUpdate::Full,
        rng(1),
        rng(7),
    );
}

fn sample_store() -> TrajectoryStore<1, 1> {
    let mut store = TrajectoryStore::new(10);
    let mut episode = Episode::new(GoalStart {
        observation: p([0]),
        achieved: goal(0),
        desired: goal(9),
    });
    for state in 1..=2 {
        episode.push(
            0,
            GoalStep {
                observation: p([state]),
                achieved: goal(state),
                terminal_state: false,
                time_limit_reached: state == 2,
            },
        );
    }
    store.insert(episode).unwrap();
    store
}

#[test]
fn every_sample_uses_one_goal_and_task_reward_and_correct_end_flags() {
    let store = sample_store();
    let objective = ExactMatch {
        reward_on_reach: 1000.0,
    };
    let mut draws = Sampler::new(sampler(1.0, TruncationMode::Bootstrap), rng(7));
    let draws = draws
        .draw::<1, 1, 2>(&store, 100, &ObjectiveEvaluator(&objective))
        .unwrap();
    assert!(draws.iter().any(|d| d.scored.sample.done));
    assert!(draws.iter().any(|d| !d.scored.sample.done));
    for d in draws {
        let episode = store.episode(d.episode).unwrap();
        let raw = episode.steps()[d.transition].step;
        let expected = raw.outcome(&objective, &d.goal);
        assert_eq!(d.scored.outcome, expected);
        assert_eq!(d.scored.sample.reward, expected.reward);
        assert_eq!(d.scored.sample.done, expected.terminated);
        assert_eq!(d.scored.sample.state.symbols[1], d.goal.symbols[0]);
        assert_eq!(d.scored.sample.next_state.symbols[1], d.goal.symbols[0]);
    }
    let episode = store.episodes().next().unwrap();
    let original = acs2_trajectory::build_sample::<1, 1, 2>(
        episode,
        1,
        &goal(9),
        &ObjectiveEvaluator(&objective),
        TruncationMode::Bootstrap,
    )
    .unwrap();
    assert!(original.outcome.truncated);
    assert!(!original.sample.done);
}

#[test]
fn values_step_preserves_all_model_fields_and_randomness_but_full_step_can_create_rules() {
    let mut population = conditioned_population();
    let before = population.classifiers().to_vec();
    let sample = ReplaySample {
        state: p([0, 9]),
        next_state: p([1, 9]),
        action: 1,
        reward: 1000.0,
        done: true,
    };
    values_learning_step(&mut population, &config(), &sample);
    assert_ne!(population.get(3).r.to_bits(), before[3].r.to_bits());
    assert_ne!(population.get(3).ir.to_bits(), before[3].ir.to_bits());
    let mut normalized = population.classifiers().to_vec();
    for (a, b) in normalized.iter_mut().zip(&before) {
        a.r = b.r;
        a.ir = b.ir;
    }
    assert_eq!(format!("{normalized:?}"), format!("{before:?}"));
    let store = sample_store();
    let draw = Sampler::new(sampler(1.0, TruncationMode::Bootstrap), rng(7))
        .draw::<1, 1, 2>(
            &store,
            1,
            &ObjectiveEvaluator(&ExactMatch {
                reward_on_reach: 1000.0,
            }),
        )
        .unwrap()
        .remove(0);
    let mut values = acs2her::Agent::<1, 1, 2, _, _>::new(
        config(),
        replay(1),
        acs2her::Preset::Acs2HerValues { proportion: 1.0 },
        rng(1),
        rng(7),
    );
    let mut full =
        acs2her::Agent::<1, 1, 2, _, _>::new(config(), replay(1), acs2her::Preset::Acs2Her, rng(1), rng(7));
    values.learner_mut().learn_draw(&draw, 2);
    full.learner_mut().learn_draw(&draw, 2);
    assert!(values.learner().population().is_empty());
    assert!(!full.learner().population().is_empty());
    assert_eq!(values.learner().counts().values_only, 1);
    assert_eq!(full.learner().counts().full, 1);
    let mut original = draw;
    original.provenance.origin = acs2_trajectory::SampleOrigin::Original;
    values.learner_mut().learn_draw(&original, 3);
    assert!(!values.learner().population().is_empty());
    assert_eq!(values.learner().counts().full, 1);
}

#[test]
fn values_bootstrap_uses_the_same_next_goal_and_done_zeroes_it() {
    let mut rules = vec![];
    for (state, desired, action, reward, quality) in [
        (0, 9, 1, 100.0, 0.5),
        (1, 9, 0, 600.0, 0.8),
        (1, 7, 0, 4000.0, 1.0),
    ] {
        let mut classifier = Classifier::general(Some(action), &config());
        classifier.condition.symbols = [Symbol::Token(state), Symbol::Token(desired)];
        classifier.effect.symbols[0] = Symbol::Token(state + 1);
        classifier.r = reward;
        classifier.q = quality;
        rules.push(classifier);
    }
    let mut population = Population::from_classifiers(rules);
    let before = population.get(0).clone();
    let untouched = format!("{:?}", &population.classifiers()[1..]);
    let sample = ReplaySample {
        state: p([0, 9]),
        next_state: p([1, 9]),
        action: 1,
        reward: 0.0,
        done: false,
    };
    let cfg = config();
    assert_eq!(population.form_match_set(&sample.state), [0]);
    assert_eq!(population.form_match_set(&sample.next_state), [1]);
    let bootstrap: f64 = 480.0;
    assert_eq!(population.get(1).fitness().to_bits(), bootstrap.to_bits());
    values_learning_step(&mut population, &cfg, &sample);
    assert_eq!(
        population.get(0).r.to_bits(),
        (before.r + cfg.beta * (cfg.gamma * bootstrap - before.r)).to_bits()
    );
    let previous = population.get(0).r;
    values_learning_step(
        &mut population,
        &cfg,
        &ReplaySample {
            done: true,
            ..sample
        },
    );
    assert_eq!(
        population.get(0).r.to_bits(),
        (previous + cfg.beta * -previous).to_bits()
    );
    assert_eq!(format!("{:?}", &population.classifiers()[1..]), untouched);
    let random = rng(1);
    assert_eq!(random.capture_state().unwrap().stream, 1);
}

#[test]
fn a_successful_relabeled_draw_reaches_the_learner_with_terminal_reward() {
    let cfg = Configuration {
        number_of_possible_actions: 1,
        ..config()
    };
    let mut classifier = Classifier::general(Some(0), &cfg);
    classifier.condition.symbols[1] = Symbol::Token(1);
    classifier.effect.symbols[0] = Symbol::Token(1);
    classifier.r = 100.0;
    let mut learner = EpisodicReplay::<1, 1, 2, _, _>::new(
        cfg,
        replay(1),
        sampler(1.0, TruncationMode::Bootstrap),
        CopyUpdate::ValuesOnly,
        rng(1),
        rng(7),
    )
    .with_population(Population::from_classifiers(vec![classifier]));
    learner.train_episode(&mut Steps::new(1, 9), 0);
    let mut expected = 100.0;
    for _ in 0..3 {
        expected += config().beta * (1000.0 - expected);
    }
    assert_eq!(learner.population().get(0).r.to_bits(), expected.to_bits());
    assert_eq!(learner.population().get(0).q.to_bits(), 0.5f64.to_bits());
    assert_eq!(learner.counts().values_only, 3);
    assert_eq!(learner.sampler().counters().relabeled.done, 3);
}
