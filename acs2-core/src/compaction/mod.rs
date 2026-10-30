pub mod cracs;
pub mod rule;

pub use cracs::{
    compact, compact_population, is_subsumed, reliable_mean_specificity, CompactionConfiguration,
    CompactionOutcome, Decision, FilterReason,
};
pub use rule::CompactableRule;
