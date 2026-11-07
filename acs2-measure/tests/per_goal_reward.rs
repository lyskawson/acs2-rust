use acs2_agents::episodic_replay::{EpisodicAgent, EpisodicReplay};
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::{
    classifier::Classifier, perception::Perception, population::Population, symbol::Symbol,
};
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{assert_protocol_evaluation_read_only, AgentKind, GoalAgent, Preset};

#[test]
fn both_per_goal_arms_preserve_values_streams_counters_and_the_complete_learner_during_evaluation()
{
    let distributions =
        MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    use acs2_measure::task::Task;
    let goals = <_ as Task<8, 2, 10>>::training_goals(distributions.protocol().training)
        .unwrap()
        .to_vec();
    assert_protocol_evaluation_read_only(
        &distributions.protocol(),
        42,
        &[10, 100],
        || {
            EpisodicGoalAgent(acs2_agents::acs2er::episodic::PerGoalRewardAgent::<
                8,
                2,
                10,
                _,
                _,
            >::new(
                preset.config(8),
                episodic_configuration(preset),
                &goals,
                ChaChaRandomSource::from_seed_and_stream(42, 1),
                ChaChaRandomSource::from_seed_and_stream(42, 7),
            ))
        },
        |agent| agent.0.learner().snapshot(),
    );
    assert_protocol_evaluation_read_only(
        &distributions.protocol(),
        42,
        &[10, 100],
        || {
            EpisodicGoalAgent(
                acs2_agents::acs2her::per_goal_reward::Agent::<8, 2, 10, _, _>::new(
                    preset.config(8),
                    episodic_configuration(preset),
                    &goals,
                    ChaChaRandomSource::from_seed_and_stream(42, 1),
                    ChaChaRandomSource::from_seed_and_stream(42, 7),
                ),
            )
        },
        |agent| agent.0.learner().snapshot(),
    );
}

fn verify_policy<
    A: EpisodicAgent<2, 1, 3, PolicyRandom = ChaChaRandomSource, SamplerRandom = ChaChaRandomSource>,
>(
    mut agent: A,
) {
    let config = agent.learner().config().clone();
    assert_eq!(config.epsilon, 0.8);
    let first = Goal::new([Symbol::Token(8)]);
    let second = Goal::new([Symbol::Token(9)]);
    let mut left = Classifier::general(Some(0), &config);
    left.q = 0.5;
    left.num = 3;
    left.effect.symbols[0] = Symbol::Token(1);
    left.r = 1e250;
    left.ir = -1e250;
    let mut right = left.clone();
    right.action = Some(1);
    right.num = 1;
    right.q = 1.0;
    right.r = -1e250;
    let seeded = EpisodicReplay::new(
        config,
        agent.learner().replay_configuration(),
        agent.learner().sampler().configuration(),
        agent.learner().copy_update(),
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    )
    .with_population(Population::from_classifiers(vec![left, right]))
    .with_per_goal_rewards(&[first, second]);
    *agent.learner_mut() = seeded;
    let rewards = agent.learner_mut().per_goal_rewards_mut().unwrap();
    rewards.entry_mut(0, &first).r = 10.0;
    rewards.entry_mut(1, &first).r = 8.0;
    rewards.entry_mut(0, &second).r = -10.0;
    rewards.entry_mut(1, &second).r = 20.0;
    let agent = EpisodicGoalAgent(agent);
    let before = agent.0.learner().snapshot();
    for seed in 42..92 {
        for (goal, expected_action, estimate, score, r) in [
            (first, 0, 5.0_f64, 15.0, 10.0),
            (second, 1, 20.0, 20.0, 20.0),
        ] {
            let state = GoalLayout::<2, 1, 3>::join(&Perception::new([Symbol::Token(0); 2]), &goal);
            let (action, value) =
                agent.eval_action(&state, &mut ChaChaRandomSource::from_seed(seed));
            assert_eq!(action, expected_action);
            assert_eq!(value.to_bits(), estimate.to_bits());
            let profile = agent.first_decision(0, &state, action).unwrap();
            let winners = profile["winners"].as_array().unwrap();
            assert_eq!(winners.len(), 1);
            assert_eq!(winners[0]["r"], r);
            assert_eq!(winners[0]["selection_score"], score);
            assert_eq!(winners[0]["fitness"], estimate);
        }
    }
    assert_eq!(agent.0.learner().snapshot(), before);
}

#[test]
fn both_new_policies_are_greedy_at_preset_epsilon_and_pair_profiles_use_the_pairs_goal() {
    let preset = Preset::thesis();
    let goals = [Goal::new([Symbol::Token(8)]), Goal::new([Symbol::Token(9)])];
    verify_policy(acs2_agents::acs2er::episodic::PerGoalRewardAgent::new(
        preset.config(2),
        episodic_configuration(preset),
        &goals,
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    ));
    verify_policy(acs2_agents::acs2her::per_goal_reward::Agent::new(
        preset.config(2),
        episodic_configuration(preset),
        &goals,
        ChaChaRandomSource::from_seed(42),
        ChaChaRandomSource::from_seed(43),
    ));
}

#[test]
fn per_goal_arm_names_and_shortened_proportion_are_exact() {
    for (name, kind) in [
        (
            "acs2er_episodic_per_goal_reward",
            AgentKind::Acs2ErEpisodicPerGoalReward,
        ),
        (
            "acs2her_shortened_per_goal_reward",
            AgentKind::Acs2HerShortenedPerGoalReward,
        ),
    ] {
        assert_eq!(AgentKind::from_run_name(name, 2.0 / 3.0).unwrap(), kind);
        assert_eq!(kind.name(), name);
        assert!(AgentKind::from_run_name(&(name.to_owned() + "_extra"), 2.0 / 3.0).is_err());
    }
    assert!(AgentKind::from_run_name("acs2her_shortened_per_goal_reward", 0.5).is_err());
}

#[test]
fn single_goal_pair_records_equal_the_base_records_in_binary64_bits() {
    let config = Preset::thesis().config(2);
    let mut first = Classifier::general(Some(0), &config);
    first.q = 0.7;
    first.num = 2;
    first.r = 11.0;
    first.effect.symbols[0] = Symbol::Token(1);
    let mut second = first.clone();
    second.action = Some(1);
    second.q = 0.9;
    second.r = 13.0;
    let population = Population::from_classifiers(vec![first, second]);
    let goal = Goal::new([Symbol::Token(9)]);
    let mut values = acs2_agents::per_goal_reward::PerGoalRewards::new(&[goal], &config, 2);
    for (index, classifier) in population.classifiers().iter().enumerate() {
        values.entry_mut(index, &goal).r = classifier.r;
    }
    let state = GoalLayout::<2, 1, 3>::join(&Perception::new([Symbol::Token(0); 2]), &goal);
    let base = acs2_measure::first_decision::profile::<2, 1, 3>(&population, &state, 1);
    let actual = acs2_measure::first_decision::profile_with_rewards::<2, 1, 3>(
        &population,
        &state,
        1,
        |reference| values.entry(reference, &goal).r,
    );
    assert_eq!(base.to_string(), actual.to_string());
}
