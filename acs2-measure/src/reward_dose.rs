use acs2_agents::acs2her::random_rules::RandomRuleConfiguration;
use serde_json::Value;

pub fn registered(task: &str) -> RandomRuleConfiguration {
    let registration: Value = serde_json::from_str(include_str!(
        "../../experiments/reward-update-dose-control/probabilities.json"
    )).expect("registered reward-update dose");
    let maze = &registration["mazes"][task];
    let skipped = maze["skipped_without_goal"].as_u64().expect("maze with a registered reward-update dose");
    let updated = maze["with_goal"].as_u64().expect("registered integer dose denominator");
    RandomRuleConfiguration { skip_probability: skipped as f64 / (skipped + updated) as f64 }
}
