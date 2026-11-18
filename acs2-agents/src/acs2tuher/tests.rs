use acs2_core::checkpoint::Checkpointed;
use acs2_core::ports::goal::{ExactMatch, GoalStart, GoalStep};
use acs2_core::representation::classifier::Classifier;
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::Population;
use acs2_core::representation::symbol::Symbol::Token;
use acs2_core::support::measurement::{start_match_counting, stop_match_counting, MatchCounters};
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_trajectory::{
    Admissibility, Episode, GoalStrategy, ObjectiveEvaluator, SamplerConfiguration, Selection,
};

use super::*;

const OBJECTIVE: ObjectiveEvaluator<'_, ExactMatch> = ObjectiveEvaluator(&ExactMatch {
    reward_on_reach: 32.0,
});

fn goal(value: u8) -> Goal<1> {
    Goal::new([Token(value)])
}

fn raw(value: u8) -> GoalStep<1, 1> {
    GoalStep {
        observation: Perception::new([Token(value)]),
        achieved: goal(value),
        terminal_state: false,
        time_limit_reached: false,
    }
}

fn trajectory(desired: u8, states: &[u8]) -> Episode<1, 1> {
    let mut episode = Episode::new(GoalStart {
        observation: Perception::new([Token(states[0])]),
        achieved: goal(states[0]),
        desired: goal(desired),
    });
    for &state in &states[1..] {
        episode.push(0, raw(state));
    }
    episode
}

fn config() -> Configuration {
    Configuration {
        do_ga: false,
        beta: 0.5,
        gamma: 0.5,
        epsilon: 0.8,
        number_of_possible_actions: 2,
        ..Configuration::default_protocol()
    }
}

fn replay(capacity: usize) -> EpisodicConfiguration {
    EpisodicConfiguration {
        capacity,
        warmup: 1,
        updates_per_step: 3,
        truncation: TruncationMode::Bootstrap,
    }
}

fn rngs() -> UtilityRandom<ChaChaRandomSource> {
    UtilityRandom {
        routes: ChaChaRandomSource::from_seed_and_stream(42, ROUTE_STREAM),
        candidates: ChaChaRandomSource::from_seed_and_stream(42, CANDIDATE_STREAM),
    }
}

fn learner(goals: &[Goal<1>]) -> EpisodicReplay<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource> {
    let mut population = Population::new();
    for (state, action, anticipating) in [
        (0, 0, true),
        (1, 0, true),
        (1, 1, true),
        (0, 0, false),
        (2, 1, true),
        (3, 0, true),
    ] {
        let mut classifier = Classifier::general(Some(action), &config());
        classifier.condition.set(0, Token(state));
        classifier.q = 0.5;
        if anticipating {
            classifier.condition.set(1, Token(2));
            classifier.effect.set(0, Token(state + 1));
        }
        population.insert(classifier);
    }
    let sampler = SamplerConfiguration {
        selection: Selection {
            strategy: GoalStrategy::Future,
            admissibility: Admissibility::CounterfactualEpisode,
            candidate_filter: false,
        },
        relabeled_proportion: 2.0 / 3.0,
        truncation: TruncationMode::Bootstrap,
    };
    let mut learner = EpisodicReplay::new(
        config(),
        replay(20),
        sampler,
        crate::episodic_replay::CopyUpdate::Full,
        ChaChaRandomSource::from_seed(18),
        ChaChaRandomSource::from_seed(19),
    )
    .with_per_goal_rewards(goals)
    .with_population(population);
    let values = learner.per_goal_rewards_mut().unwrap();
    for (reference, r) in [4.0, 8.0, 16.0, 100.0, 20.0, 4.0].into_iter().enumerate() {
        values.entry_mut(reference, &goal(2)).r = r;
    }
    learner
}

#[test]
fn scores_use_the_counterfactual_prefix_and_agent_values() {
    let learner = learner(&[goal(2), goal(4), goal(9)]);
    let scoring = scoring(&learner);
    let mut store = TrajectoryStore::new(20);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    let episode = store.episode(id).unwrap();
    let mut cost = Cost::default();
    let matches = ObservationMatches::new(episode, 4, learner.population(), &mut cost);
    let score = |measure, g, cost: &mut Cost| {
        let prefix = first_reach(episode, &goal(g), &OBJECTIVE).unwrap_or(episode.len());
        scoring.score(
            measure,
            episode,
            (goal(g), prefix),
            &matches,
            &OBJECTIVE,
            cost,
        )
    };
    assert_eq!(score(Measure::MeanValue, 2, &mut cost).utility, 3.0);
    let trial = score(Measure::TrialUpdate, 2, &mut cost);
    assert_eq!(trial.utility, 0.5625);
    assert_eq!(trial.post_update_level, Some(6.0625));
    assert_eq!(score(Measure::MeanValue, 4, &mut cost).utility, 0.0);
    let trial_without_anticipation = score(Measure::TrialUpdate, 4, &mut cost);
    assert_eq!(trial_without_anticipation.utility, 0.0);
    assert_eq!(trial_without_anticipation.post_update_level, Some(0.0));
    assert_eq!(cost.trial_value_updates, 5);
    assert_eq!(cost.trial_reward_entries, 4);
    assert_eq!(cost.value_reads, 25);
    assert_eq!(cost.match_sets, 16);
}

#[test]
fn candidates_are_distinct_real_training_goals_with_a_nonempty_prefix() {
    let mut store = TrajectoryStore::new(20);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    for g in [0, 2, 4, 5, 6, 7] {
        store.insert(trajectory(g, &[8, 2])).unwrap();
    }
    let episode = store.episode(id).unwrap();
    let mut rng = rngs().candidates;
    let all = sample_candidates(episode, store.candidates(), 8, &OBJECTIVE, &mut rng);
    assert_eq!(
        all.iter().copied().collect::<BTreeSet<_>>(),
        [2, 4, 5, 6, 7].map(goal).into()
    );
    let mut included = [0; 5];
    for _ in 0..2_500 {
        let subset = sample_candidates(episode, store.candidates(), 4, &OBJECTIVE, &mut rng);
        assert_eq!(subset.iter().collect::<BTreeSet<_>>().len(), 4);
        for &g in &subset {
            included[all.iter().position(|&allowed| g == allowed).unwrap()] += 1;
        }
    }
    assert!(included.into_iter().all(|n| (1_850..2_150).contains(&n)));
}

#[test]
fn selection_breaks_ties_uniformly_and_random_has_the_same_choices() {
    let real = [0, 2, 4, 5, 6, 7, 9].map(goal);
    let empty = per_goal_reward::Agent::<1, 1, 2, _, _>::new(
        config(),
        replay(20),
        &real,
        ChaChaRandomSource::from_seed(18),
        ChaChaRandomSource::from_seed(19),
    );
    let mut store = TrajectoryStore::new(20);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    for &g in &[0, 2, 4, 5, 6, 7] {
        store.insert(trajectory(g, &[8, 2])).unwrap();
    }
    let parameters = |measure| UtilityConfiguration {
        measure,
        candidates: 4,
        refresh_steps: None,
    };
    let mut mean = Utility::new(parameters(Measure::MeanValue), rngs(), &real);
    let mut random = Utility::new(parameters(Measure::RandomGoal), rngs(), &real);
    let mut counts = BTreeMap::<_, usize>::new();
    for now in 1..=5_000 {
        mean.evaluate(&store, id, &OBJECTIVE, &scoring(empty.learner()), now, true);
        random.evaluate(&store, id, &OBJECTIVE, &scoring(empty.learner()), now, true);
        let chosen = mean.assignments[&id].goal.unwrap();
        assert_eq!(random.assignments[&id].goal, Some(chosen));
        *counts.entry(chosen).or_default() += 1;
    }
    assert_eq!(counts.len(), 5);
    assert!(counts.values().all(|&n| (875..1_125).contains(&n)));
    assert_eq!(random.cost.candidates_scored, 0);
    assert_eq!(mean.cost.candidates_scored, 20_000);
    let trained = learner(&real);
    for measure in [Measure::MeanValue, Measure::TrialUpdate] {
        let mut maximizing = Utility::new(UtilityConfiguration::pilot(measure), rngs(), &real);
        maximizing.evaluate(&store, id, &OBJECTIVE, &scoring(&trained), 0, true);
        assert_eq!(maximizing.assignments[&id].goal, Some(goal(2)));
    }
}

#[test]
fn trial_evaluation_leaves_learning_values_streams_and_counters_untouched() {
    let goals = [goal(2), goal(4), goal(9)];
    let learner = learner(&goals);
    let mut store = TrajectoryStore::new(20);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    store.insert(trajectory(2, &[0, 2])).unwrap();
    store.insert(trajectory(4, &[0, 1])).unwrap();
    let before = learner.snapshot();
    let mut utility = Utility::new(
        UtilityConfiguration::pilot(Measure::TrialUpdate),
        rngs(),
        &goals,
    );
    start_match_counting();
    utility.on_insertion(&store, id, &OBJECTIVE, &scoring(&learner), 7);
    assert_eq!(stop_match_counting(), Some(MatchCounters::default()));
    assert_eq!(learner.snapshot(), before);
    assert_eq!(utility.cost.candidates_scored, 2);
    assert_eq!(utility.cost.trial_value_updates, 5);
    assert!(utility.cost.value_reads > 0);
    assert_eq!(utility.reward_updates, ByRoute::default());
}

struct Last;

impl RandomSource for Last {
    fn gen_bool(&mut self, _: f64) -> bool {
        unreachable!()
    }
    fn gen_range(&mut self, bound: usize) -> usize {
        bound - 1
    }
    fn gen_unit(&mut self) -> f64 {
        unreachable!()
    }
}

#[test]
fn refresh_is_strictly_after_the_age_limit_and_resamples_seen_goals() {
    let real = [goal(2), goal(4), goal(9)];
    let mut store = TrajectoryStore::new(8);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    store.insert(trajectory(2, &[0, 2])).unwrap();
    let learner = learner(&real);
    let scoring = scoring(&learner);
    let mut utility = Utility::new(
        UtilityConfiguration::pilot(Measure::RandomGoal),
        UtilityRandom {
            routes: Last,
            candidates: Last,
        },
        &real,
    );
    utility.on_insertion(&store, id, &OBJECTIVE, &scoring, 17);
    let other = store.insert(trajectory(4, &[0, 1])).unwrap();
    utility.on_insertion(&store, other, &OBJECTIVE, &scoring, 2_017);
    assert_eq!(
        utility.assigned_goal(&store, id, &OBJECTIVE, &scoring, 2_017),
        Some(goal(2))
    );
    assert_eq!(utility.cost.refreshes, 0);
    assert_eq!(
        utility.assigned_goal(&store, id, &OBJECTIVE, &scoring, 2_018),
        Some(goal(4))
    );
    assert_eq!(utility.cost.refreshes, 1);
    assert_eq!(utility.cost.candidates_sampled, 5);
    assert_eq!(
        utility.assigned_goal(&store, id, &OBJECTIVE, &scoring, 4_018),
        Some(goal(4))
    );
    assert_eq!(utility.cost.refreshes, 1);
    let replacement = store.insert(trajectory(4, &[0, 0, 1, 1, 1, 1, 4])).unwrap();
    utility.on_insertion(&store, replacement, &OBJECTIVE, &scoring, 4_019);
    assert!(!utility.assignments.contains_key(&id));
    assert_eq!(utility.assignments.len(), 2);
}

#[test]
fn tu_draw_falls_back_after_the_goal_and_a_copy_updates_the_assigned_column() {
    let real = [goal(2), goal(9)];
    let mut learner = learner(&real);
    let mut store = TrajectoryStore::new(20);
    let id = store.insert(trajectory(9, &[0, 1, 2, 3])).unwrap();
    store.insert(trajectory(2, &[0, 2])).unwrap();
    let mut sampler = acs2_trajectory::Sampler::new(
        learner.sampler().configuration(),
        ChaChaRandomSource::from_seed(31),
    );
    let fallback = sampler
        .draw_routed::<1, 1, 2>(
            &store,
            (id, 2),
            &OBJECTIVE,
            DrawRoute::TrajectoryUtility(Some(goal(2))),
        )
        .unwrap();
    assert!(fallback.provenance.fallback);
    assert_eq!(fallback.provenance.origin, SampleOrigin::Original);
    assert_eq!(fallback.goal, goal(9));
    let no_assignment = sampler
        .draw_routed::<1, 1, 2>(
            &store,
            (id, 1),
            &OBJECTIVE,
            DrawRoute::TrajectoryUtility(None),
        )
        .unwrap();
    assert!(no_assignment.provenance.fallback);
    assert_eq!(no_assignment.goal, goal(9));
    let copy = sampler
        .draw_routed::<1, 1, 2>(
            &store,
            (id, 1),
            &OBJECTIVE,
            DrawRoute::TrajectoryUtility(Some(goal(2))),
        )
        .unwrap();
    assert!(!copy.provenance.fallback);
    assert_eq!(copy.provenance.strategy, GoalStrategy::TrajectoryUtility);
    assert_eq!(copy.goal, goal(2));
    assert_eq!(copy.scored.sample.reward, 32.0);
    assert!(copy.scored.sample.done);
    let untouched = *learner.per_goal_rewards().unwrap().entry(1, &goal(9));
    let experience = learner.population().get(1).exp;
    learner.learn_draw(&copy, 3);
    let values = learner.per_goal_rewards().unwrap();
    assert_eq!(*values.entry(1, &goal(9)), untouched);
    assert_eq!(values.entry(1, &goal(2)).r, 20.0);
    assert_eq!(values.entry(1, &goal(2)).ir, 16.0);
    assert_eq!(learner.population().get(1).exp, experience + 1);
    assert_eq!(learner.counts().full, 1);
    assert_eq!(learner.counts().values_only, 0);
}

#[test]
fn ga_and_checkpoint_restore_fail() {
    let real = [goal(2), goal(9)];
    let construct = |config| {
        Agent::<1, 1, 2, _, _>::new(
            config,
            replay(20),
            &real,
            ChaChaRandomSource::from_seed(18),
            ChaChaRandomSource::from_seed(19),
            Some((UtilityConfiguration::pilot(Measure::MeanValue), rngs())),
        )
    };
    assert!(std::panic::catch_unwind(|| construct(Configuration {
        do_ga: true,
        ..config()
    }))
    .is_err());
    let mut agent = construct(config());
    let other = crate::acs2::Agent::<2, _>::new(config(), ChaChaRandomSource::from_seed(3));
    let checkpoint = other.capture();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| agent.restore(checkpoint)))
            .is_err()
    );
}
