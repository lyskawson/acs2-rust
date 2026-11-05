use acs2_agents::acs2her::{self, random_rules::SKIP_STREAM};
use acs2_agents::episodic_replay::EpisodicAgent;
use acs2_core::support::rng::ChaChaRandomSource;
use acs2_envs::goal::maze::Coordinates;
use acs2_envs::maze::geometries::pyalcs::MAZE4;
use acs2_measure::agents::{episodic_configuration, EpisodicGoalAgent};
use acs2_measure::pair_protocol::MazePairDistributions;
use acs2_measure::runner::{assert_protocol_evaluation_read_only, AgentKind, Preset};

#[test]
fn random_rule_evaluation_preserves_the_skip_stream_counters_and_all_learner_state() {
    let distributions = MazePairDistributions::<Coordinates, 2>::registered("maze4", &MAZE4, 5, "coordinates");
    let preset = Preset::thesis();
    assert_protocol_evaluation_read_only(&distributions.protocol(), 42, &[10, 100], || {
        let mut agent = acs2her::Agent::<8, 2, 10, _, _>::new(preset.config(8), episodic_configuration(preset),
            acs2her::Preset::Acs2HerShortenedRandomRules,
            ChaChaRandomSource::from_seed_and_stream(42, 1), ChaChaRandomSource::from_seed_and_stream(42, 7));
        agent.learner_mut().set_random_rule_updates(acs2_measure::reward_dose::registered("maze4"), ChaChaRandomSource::from_seed_and_stream(42, SKIP_STREAM));
        EpisodicGoalAgent(agent)
    }, |agent| agent.0.learner().snapshot());
}

#[test]
fn random_rule_arm_has_an_exact_name_and_fixed_relabeling_proportion() {
    assert_eq!(AgentKind::from_run_name("acs2her_shortened_random_rules", 2.0 / 3.0).unwrap(), AgentKind::Acs2HerShortenedRandomRules);
    assert!(AgentKind::from_run_name("acs2her_shortened_random_rules_extra", 2.0 / 3.0).is_err());
    assert!(AgentKind::from_run_name("acs2her_shortened_random_rules", 0.5).is_err());
}
