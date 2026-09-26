pub mod maze;

use acs2_core::goal::ExactMatch;

pub const SPARSE_GOAL_REWARD: ExactMatch = ExactMatch {
    reward_on_reach: 1000.0,
};
