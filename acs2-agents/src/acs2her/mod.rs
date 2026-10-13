use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{Admissibility, GoalStrategy, SamplerConfiguration, Selection};

use crate::episodic_replay::{CopyUpdate, EpisodicAgent, EpisodicConfiguration, EpisodicReplay};

#[cfg(test)]
mod pyalcs_differential;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preset {
    Acs2Her,
    Acs2HerValues { proportion: f64 },
    Acs2HerValuesShortened { proportion: f64 },
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Acs2Her => "acs2her",
            Self::Acs2HerValues { .. } => "acs2her_values",
            Self::Acs2HerValuesShortened { .. } => "acs2her_values_shortened",
        }
    }
    pub fn proportion(self) -> f64 {
        match self {
            Self::Acs2Her => 2.0 / 3.0,
            Self::Acs2HerValues { proportion } | Self::Acs2HerValuesShortened { proportion } => proportion,
        }
    }
}

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    variant: Preset,
    learner: EpisodicReplay<S, G, M, R, P>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        variant: Preset,
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
                        admissibility: if matches!(variant, Preset::Acs2HerValuesShortened { .. }) {
                            Admissibility::CounterfactualEpisode
                        } else {
                            Admissibility::EveryTransition
                        },
                        candidate_filter: false,
                    },
                    relabeled_proportion: variant.proportion(),
                    truncation: replay.truncation,
                },
                if variant == Preset::Acs2Her {
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
