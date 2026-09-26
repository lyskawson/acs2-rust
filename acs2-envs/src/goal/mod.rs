pub mod bit_flipping;
pub mod hand_eye;
pub mod knowledge;
pub mod maze;
pub mod taxi;

use acs2_core::goal::ExactMatch;

pub const SPARSE_GOAL_REWARD: ExactMatch = ExactMatch {
    reward_on_reach: 1000.0,
};
