use acs2_agents::acs2her::{HindsightConfiguration, Preset};
use acs2_agents::episodic_replay::CopyUpdate;
use acs2_trajectory::Admissibility;

#[test]
fn full_hindsight_presets_change_only_requested_proportion() {
    for proportion in [0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 0.8, 0.731, 1.0] {
        for (preset, admissibility, name) in [
            (
                Preset::Acs2Her { proportion },
                Admissibility::EveryTransition,
                "acs2her",
            ),
            (
                Preset::Acs2HerShortened { proportion },
                Admissibility::CounterfactualEpisode,
                "acs2her_shortened",
            ),
        ] {
            assert_eq!(preset.name(), name);
            assert_eq!(
                preset.configuration(),
                HindsightConfiguration {
                    copy_update: CopyUpdate::Full,
                    admissibility,
                    relabeled_proportion: proportion,
                }
            );
            assert_eq!(
                preset.configuration().relabeled_proportion.to_bits(),
                proportion.to_bits()
            );
        }
        assert_eq!(
            Preset::Acs2HerNontrivial.configuration(),
            HindsightConfiguration {
                copy_update: CopyUpdate::Full,
                admissibility: Admissibility::FromNonGoalState,
                relabeled_proportion: 2.0 / 3.0,
            }
        );
    }
}
