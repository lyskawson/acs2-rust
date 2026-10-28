use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_trajectory::{Admissibility, GoalStrategy, SamplerConfiguration, Selection};

use crate::episodic_replay::{CopyUpdate, EpisodicAgent, EpisodicConfiguration, EpisodicReplay};

#[cfg(test)]
mod pyalcs_differential;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Preset {
    Acs2Her { proportion: f64 },
    Acs2HerNontrivial,
    Acs2HerShortened { proportion: f64 },
    Acs2HerValues { proportion: f64 },
    Acs2HerValuesShortened { proportion: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HindsightConfiguration {
    pub copy_update: CopyUpdate,
    pub admissibility: Admissibility,
    pub relabeled_proportion: f64,
}

impl Preset {
    pub fn name(self) -> &'static str {
        match self {
            Self::Acs2Her { .. } => "acs2her",
            Self::Acs2HerNontrivial => "acs2her_nontrivial",
            Self::Acs2HerShortened { .. } => "acs2her_shortened",
            Self::Acs2HerValues { .. } => "acs2her_values",
            Self::Acs2HerValuesShortened { .. } => "acs2her_values_shortened",
        }
    }
    pub fn configuration(self) -> HindsightConfiguration {
        match self {
            Self::Acs2Her { proportion } => HindsightConfiguration {
                copy_update: CopyUpdate::Full,
                admissibility: Admissibility::EveryTransition,
                relabeled_proportion: proportion,
            },
            Self::Acs2HerNontrivial => HindsightConfiguration {
                copy_update: CopyUpdate::Full,
                admissibility: Admissibility::FromNonGoalState,
                relabeled_proportion: 2.0 / 3.0,
            },
            Self::Acs2HerShortened { proportion } => HindsightConfiguration {
                copy_update: CopyUpdate::Full,
                admissibility: Admissibility::CounterfactualEpisode,
                relabeled_proportion: proportion,
            },
            Self::Acs2HerValues { proportion } => HindsightConfiguration {
                copy_update: CopyUpdate::ValuesOnly,
                admissibility: Admissibility::EveryTransition,
                relabeled_proportion: proportion,
            },
            Self::Acs2HerValuesShortened { proportion } => HindsightConfiguration {
                copy_update: CopyUpdate::ValuesOnly,
                admissibility: Admissibility::CounterfactualEpisode,
                relabeled_proportion: proportion,
            },
        }
    }
}

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    preset: Preset,
    learner: EpisodicReplay<S, G, M, R, P>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        preset: Preset,
        rng: R,
        sampler_rng: P,
    ) -> Self {
        let parameters = preset.configuration();
        Self {
            preset,
            learner: EpisodicReplay::new(
                config,
                replay,
                SamplerConfiguration {
                    selection: Selection {
                        strategy: GoalStrategy::Future,
                        admissibility: parameters.admissibility,
                        candidate_filter: false,
                    },
                    relabeled_proportion: parameters.relabeled_proportion,
                    truncation: replay.truncation,
                },
                parameters.copy_update,
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
        self.preset.name()
    }
    fn learner(&self) -> &EpisodicReplay<S, G, M, R, P> {
        &self.learner
    }
    fn learner_mut(&mut self) -> &mut EpisodicReplay<S, G, M, R, P> {
        &mut self.learner
    }
}
