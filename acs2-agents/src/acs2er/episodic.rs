use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{Admissibility, GoalStrategy, SamplerConfiguration, Selection};

use crate::episodic_replay::{CopyUpdate, EpisodicAgent, EpisodicConfiguration, EpisodicReplay};

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    learner: EpisodicReplay<S, G, M, R, P>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        rng: R,
        sampler_rng: P,
    ) -> Self {
        Self {
            learner: EpisodicReplay::new(
                config,
                replay,
                SamplerConfiguration {
                    selection: Selection {
                        strategy: GoalStrategy::Future,
                        admissibility: Admissibility::EveryTransition,
                        candidate_filter: false,
                    },
                    relabeled_proportion: 0.0,
                    truncation: replay.truncation,
                },
                CopyUpdate::Full,
                rng,
                sampler_rng,
            ),
        }
    }
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    EpisodicAgent<S, G, M> for Agent<S, G, M, R, P>
{
    type PolicyRandom = R;
    type SamplerRandom = P;
    fn name(&self) -> &'static str {
        "acs2er_episodic"
    }
    fn learner(&self) -> &EpisodicReplay<S, G, M, R, P> {
        &self.learner
    }
    fn learner_mut(&mut self) -> &mut EpisodicReplay<S, G, M, R, P> {
        &mut self.learner
    }
}
