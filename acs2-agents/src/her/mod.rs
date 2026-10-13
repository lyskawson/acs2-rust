use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{Admissibility, GoalStrategy, SamplerConfiguration, Selection};

use crate::episodic_replay::{CopyUpdate, EpisodicAgent, EpisodicConfiguration, EpisodicReplay};

#[cfg(test)]
mod validation;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Variant {
    Full,
    Values { proportion: f64 },
    ValuesShortened { proportion: f64 },
}

impl Variant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Full => "her_full",
            Self::Values { .. } => "her_values",
            Self::ValuesShortened { .. } => "her_values_shortened",
        }
    }
    pub fn proportion(self) -> f64 {
        match self {
            Self::Full => 2.0 / 3.0,
            Self::Values { proportion } | Self::ValuesShortened { proportion } => proportion,
        }
    }
}

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    variant: Variant,
    learner: EpisodicReplay<S, G, M, R, P>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        variant: Variant,
        rng: R,
        sampler_rng: P,
    ) -> Self {
        Self {
            variant,
            learner: EpisodicReplay::new(
                config,
                replay,
                SamplerConfiguration {
                    selection: Selection {
                        strategy: GoalStrategy::Future,
                        admissibility: if matches!(variant, Variant::ValuesShortened { .. }) {
                            Admissibility::CounterfactualEpisode
                        } else {
                            Admissibility::EveryTransition
                        },
                        candidate_filter: false,
                    },
                    relabeled_proportion: variant.proportion(),
                    truncation: replay.truncation,
                },
                if variant == Variant::Full {
                    CopyUpdate::Full
                } else {
                    CopyUpdate::ValuesOnly
                },
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
        self.variant.name()
    }
    fn learner(&self) -> &EpisodicReplay<S, G, M, R, P> {
        &self.learner
    }
    fn learner_mut(&mut self) -> &mut EpisodicReplay<S, G, M, R, P> {
        &mut self.learner
    }
}
