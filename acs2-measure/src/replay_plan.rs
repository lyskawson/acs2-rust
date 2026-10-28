use acs2_agents::episodic_replay::{ReplayOrder, ReplayPlan, ReplaySchedule, ORDER_STREAM};
use acs2_core::support::rng::ChaChaRandomSource;
use serde_json::{json, Value};

pub fn parse(schedule: &str, order: &str) -> ReplayPlan {
    let schedule = match schedule {
        "constant" => ReplaySchedule::Constant,
        "3to1at30000" => ReplaySchedule::ThreeToOne {
            switch_steps: 30_000,
        },
        "3to1at100000" => ReplaySchedule::ThreeToOne {
            switch_steps: 100_000,
        },
        "3to1at300000" => ReplaySchedule::ThreeToOne {
            switch_steps: 300_000,
        },
        _ => panic!("unregistered replay schedule"),
    };
    let order = match order {
        "independent" => ReplayOrder::Independent,
        "backward" => ReplayOrder::Backward,
        "random-pass" => ReplayOrder::RandomPass,
        _ => panic!("unregistered replay order"),
    };
    let plan = ReplayPlan { schedule, order };
    plan.validate();
    plan
}

pub fn order_rng(plan: ReplayPlan, seed: u64) -> Option<ChaChaRandomSource> {
    (plan.order == ReplayOrder::RandomPass)
        .then(|| ChaChaRandomSource::from_seed_and_stream(seed, ORDER_STREAM))
}

pub fn annotate(parameters: &mut Value, plan: ReplayPlan) {
    if plan == ReplayPlan::default() {
        return;
    }
    parameters["replay_schedule"] = match plan.schedule {
        ReplaySchedule::Constant => json!({"kind": "constant"}),
        ReplaySchedule::ThreeToOne { switch_steps } => json!({
            "kind": "three_to_one", "switch_steps": switch_steps,
            "early_rate": 3, "late_rate": 1, "clock": "episode_end_environment_steps",
            "early_inclusive": true,
        }),
    };
    parameters["replay_order"] = json!(match plan.order {
        ReplayOrder::Independent => "independent",
        ReplayOrder::Backward => "backward",
        ReplayOrder::RandomPass => "random_pass",
    });
    if plan.order != ReplayOrder::Independent {
        parameters["draws"] = json!("length_weighted_episode_passes");
        parameters["strategy"] = json!("episode");
        parameters["pass_goal"] = json!("one_goal_per_pass_admitted_at_start_then_per_transition");
        parameters["pass_cut"] = json!("backward_suffix_before_ordering_exact_episode_budget");
        if plan.order == ReplayOrder::RandomPass {
            parameters["order_stream"] = json!(ORDER_STREAM);
        }
    }
}
