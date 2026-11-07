use acs2_core::ports::goal::Goal;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;

use crate::episodic_replay::{EpisodicAgent, EpisodicConfiguration, EpisodicReplay};

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    learner: EpisodicReplay<S, G, M, R, P>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        goals: &[Goal<G>],
        rng: R,
        sampler_rng: P,
    ) -> Self {
        let base = super::Agent::<S, G, M, _, _>::new(
            config,
            replay,
            super::Preset::Acs2HerShortened {
                proportion: 2.0 / 3.0,
            },
            rng,
            sampler_rng,
        );
        Self {
            learner: base.learner.with_per_goal_rewards(goals),
        }
    }
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    EpisodicAgent<S, G, M> for Agent<S, G, M, R, P>
{
    type PolicyRandom = R;
    type SamplerRandom = P;
    fn name(&self) -> &'static str {
        "acs2her_shortened_per_goal_reward"
    }
    fn learner(&self) -> &EpisodicReplay<S, G, M, R, P> {
        &self.learner
    }
    fn learner_mut(&mut self) -> &mut EpisodicReplay<S, G, M, R, P> {
        &mut self.learner
    }
}
