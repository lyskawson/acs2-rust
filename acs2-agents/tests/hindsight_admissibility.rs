use acs2_agents::acs2her::{Agent, Preset};
use acs2_agents::episodic_replay::{
    CopyUpdate, EpisodicAgent, EpisodicConfiguration, WallCounters,
};
use acs2_core::ports::goal::{ExactMatch, Goal, GoalStart, GoalStep};
use acs2_core::representation::{perception::Perception, symbol::Symbol};
use acs2_core::support::{config::Configuration, rng::ChaChaRandomSource};
use acs2_core::trial::TruncationMode;
use acs2_trajectory::{
    Admissibility, Episode, GoalStrategy, ObjectiveEvaluator, SampleOrigin, Sampler,
    TrajectoryStore,
};

fn goal(value: u8) -> Goal<1> {
    Goal::new([Symbol::Token(value)])
}

fn agent(preset: Preset) -> Agent<1, 1, 2, ChaChaRandomSource, ChaChaRandomSource> {
    Agent::new(
        Configuration {
            number_of_possible_actions: 2,
            do_ga: false,
            ..Configuration::default_protocol()
        },
        EpisodicConfiguration {
            capacity: 10000,
            warmup: 1,
            updates_per_step: 3,
            truncation: TruncationMode::Bootstrap,
        },
        preset,
        ChaChaRandomSource::from_seed_and_stream(42, 1),
        ChaChaRandomSource::from_seed_and_stream(42, 7),
    )
}

fn history(states: &[u8]) -> TrajectoryStore<1, 1> {
    let mut episode = Episode::new(GoalStart {
        observation: Perception {
            symbols: [Symbol::Token(states[0])],
        },
        achieved: goal(states[0]),
        desired: goal(9),
    });
    for (index, state) in states.iter().skip(1).enumerate() {
        episode.push(
            0,
            GoalStep {
                observation: Perception {
                    symbols: [Symbol::Token(*state)],
                },
                achieved: goal(*state),
                terminal_state: false,
                time_limit_reached: index + 2 == states.len(),
            },
        );
    }
    let mut store = TrajectoryStore::new(10000);
    store.insert(episode).unwrap();
    store
}

#[test]
fn restricted_full_presets_fix_every_configuration_axis() {
    for (preset, name, admissibility) in [
        (
            Preset::Acs2HerNontrivial,
            "acs2her_nontrivial",
            Admissibility::FromNonGoalState,
        ),
        (
            Preset::Acs2HerShortened { proportion: 2.0 / 3.0 },
            "acs2her_shortened",
            Admissibility::CounterfactualEpisode,
        ),
    ] {
        let agent = agent(preset);
        let learner = agent.learner();
        let sampler = learner.sampler().configuration();
        assert_eq!(agent.name(), name);
        assert_eq!(learner.copy_update(), CopyUpdate::Full);
        assert_eq!(sampler.selection.admissibility, admissibility);
        assert_eq!(sampler.selection.strategy, GoalStrategy::Future);
        assert!(!sampler.selection.candidate_filter);
        assert_eq!(
            sampler.relabeled_proportion.to_bits(),
            (2.0_f64 / 3.0).to_bits()
        );
        assert_eq!(sampler.truncation, TruncationMode::Bootstrap);
        assert_eq!(
            learner.replay_configuration(),
            EpisodicConfiguration {
                capacity: 10000,
                warmup: 1,
                updates_per_step: 3,
                truncation: TruncationMode::Bootstrap,
            }
        );
    }
}

#[test]
fn admissible_copies_keep_first_arrival_and_fallbacks_take_full_steps() {
    let store = history(&[0, 0, 1, 0, 2, 2]);
    let objective = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    for preset in [Preset::Acs2HerNontrivial, Preset::Acs2HerShortened { proportion: 2.0 / 3.0 }] {
        let mut agent = agent(preset);
        let mut sampler = Sampler::new(
            agent.learner().sampler().configuration(),
            ChaChaRandomSource::from_seed_and_stream(42, 7),
        );
        let draws = sampler.draw::<1, 1, 2>(&store, 3000, &objective).unwrap();
        let counters = sampler.counters();
        assert!(counters.relabeled.draws > 0);
        assert_eq!(counters.relabeled.already_reached, 0);
        if matches!(preset, Preset::Acs2HerShortened { .. }) {
            assert_eq!(counters.relabeled.after_counterfactual_end, 0);
        } else {
            assert!(counters.relabeled.after_counterfactual_end > 0);
        }
        assert!(draws.iter().any(|draw| draw.transition == 3
            && draw.goal == goal(2)
            && draw.scored.outcome.terminated));
        assert!(counters.fallbacks > 0);
        assert_eq!(
            counters.fallbacks,
            draws.iter().filter(|draw| draw.provenance.fallback).count() as u64
        );
        for draw in &draws {
            if draw.provenance.fallback {
                assert_eq!(draw.provenance.origin, SampleOrigin::Original);
                assert_eq!(draw.goal, goal(9));
                assert_eq!(draw.scored.sample.reward, 0.0);
                assert!(!draw.scored.sample.done);
            }
            agent.learner_mut().learn_draw(draw, 5);
        }
        assert_eq!(agent.learner().counts().full, 3000);
        assert_eq!(agent.learner().counts().values_only, 0);
        assert!(!agent.learner().population().is_empty());
    }
}

#[test]
fn wall_counters_use_relabeled_route_state_identity_and_goal_termination() {
    let states = [0, 0, 1, 0, 2, 2];
    let store = history(&states);
    let objective = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let agent = agent(Preset::Acs2Her { proportion: 2.0 / 3.0 });
    let mut sampler = Sampler::new(
        agent.learner().sampler().configuration(),
        ChaChaRandomSource::from_seed_and_stream(42, 7),
    );
    let draws = sampler.draw::<1, 1, 2>(&store, 3000, &objective).unwrap();
    let mut actual = WallCounters::default();
    let mut expected = WallCounters::default();
    let mut original_walls = 0;
    for draw in &draws {
        let episode = store.episode(draw.episode).unwrap();
        actual.observe(
            draw,
            episode.achieved(draw.transition),
            episode.achieved(draw.transition + 1),
        );
        let unchanged = states[draw.transition] == states[draw.transition + 1];
        if draw.provenance.origin == SampleOrigin::Original {
            original_walls += u64::from(unchanged);
            continue;
        }
        expected.relabeled_draws += 1;
        expected.unchanged_state += u64::from(unchanged);
        expected.unchanged_state_terminal +=
            u64::from(unchanged && goal(states[draw.transition + 1]) == draw.goal);
    }
    assert!(original_walls > 0);
    assert!(expected.unchanged_state_terminal > 0);
    assert!(expected.unchanged_state > expected.unchanged_state_terminal);
    assert_eq!(actual, expected);
}

#[test]
fn perceptual_twins_are_not_wall_bumps_when_the_coordinate_changes() {
    let mut episode = Episode::new(GoalStart {
        observation: Perception {
            symbols: [Symbol::Token(0)],
        },
        achieved: goal(0),
        desired: goal(9),
    });
    episode.push(
        0,
        GoalStep {
            observation: Perception {
                symbols: [Symbol::Token(0)],
            },
            achieved: goal(1),
            terminal_state: false,
            time_limit_reached: true,
        },
    );
    let mut store = TrajectoryStore::new(10);
    store.insert(episode).unwrap();
    let agent = agent(Preset::Acs2HerNontrivial);
    let mut sampler = Sampler::new(
        agent.learner().sampler().configuration(),
        ChaChaRandomSource::from_seed_and_stream(42, 7),
    );
    let objective = ObjectiveEvaluator(&ExactMatch {
        reward_on_reach: 1000.0,
    });
    let draws = sampler.draw::<1, 1, 2>(&store, 100, &objective).unwrap();
    let mut walls = WallCounters::default();
    for draw in &draws {
        let episode = store.episode(draw.episode).unwrap();
        walls.observe(
            draw,
            episode.achieved(draw.transition),
            episode.achieved(draw.transition + 1),
        );
    }
    assert!(walls.relabeled_draws > 0);
    assert_eq!(walls.unchanged_state, 0);
    assert_eq!(walls.unchanged_state_terminal, 0);
}
