use acs2_core::learning::replay::ReplaySample;
use acs2_core::representation::population::Population;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::SampleOrigin;

use super::goal_rules::{learning_step_counted, RouteCounts};

pub const SKIP_STREAM: u64 = 0x7265_7761_7264_736b;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RandomRuleConfiguration {
    pub skip_probability: f64,
}

impl RandomRuleConfiguration {
    pub fn validate(self) {
        assert!(self.skip_probability.is_finite() && (0.0..=1.0).contains(&self.skip_probability));
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RandomRouteCounts {
    pub updates: RouteCounts,
    pub skipped_with_goal: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RandomRewardCounts {
    pub original: RandomRouteCounts,
    pub relabeled: RandomRouteCounts,
}

pub struct RandomRuleUpdates<P> {
    parameters: RandomRuleConfiguration,
    rng: P,
    counts: RandomRewardCounts,
}

impl<P: RandomSource> RandomRuleUpdates<P> {
    pub fn new(parameters: RandomRuleConfiguration, rng: P) -> Self {
        parameters.validate();
        Self { parameters, rng, counts: RandomRewardCounts::default() }
    }

    pub fn parameters(&self) -> RandomRuleConfiguration {
        self.parameters
    }

    pub fn counts(&self) -> RandomRewardCounts {
        self.counts
    }

    pub fn snapshot(&self) -> String {
        format!("{:?}|{:?}|{:?}", self.parameters, self.rng.capture_state(), self.counts)
    }

    pub(crate) fn learn<const S: usize, const M: usize>(
        &mut self,
        population: &mut Population<M>,
        config: &Configuration,
        sample: &ReplaySample<M>,
        time: u64,
        policy_rng: &mut dyn RandomSource,
        origin: SampleOrigin,
    ) {
        let relabeled = origin == SampleOrigin::Relabeled;
        let (value, skipped) = learning_step_counted::<S, M>(
            population, config, sample, time, policy_rng,
            |_| !relabeled || !self.rng.gen_bool(self.parameters.skip_probability),
        );
        let route = if relabeled { &mut self.counts.relabeled } else { &mut self.counts.original };
        route.updates.draws += value.draws;
        route.updates.with_goal += value.with_goal;
        route.updates.without_goal += value.without_goal;
        route.updates.skipped_without_goal += value.skipped_without_goal;
        route.updates.draws_with_goal += value.draws_with_goal;
        route.updates.action_set_rules += value.action_set_rules;
        route.skipped_with_goal += skipped;
    }
}

#[cfg(test)]
mod tests;
