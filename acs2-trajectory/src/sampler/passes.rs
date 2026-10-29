use super::*;
use crate::selection::Admissibility;

pub struct EpisodePass<const G: usize, const M: usize> {
    pub samples: Vec<DrawnSample<G, M>>,
    pub available_steps: usize,
    pub requested_relabel: bool,
    pub fallback: bool,
}

impl<R: RandomSource> Sampler<R> {
    pub fn draw_pass<const S: usize, const G: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        limit: usize,
        evaluator: &impl GoalEvaluator<S, G>,
    ) -> Result<EpisodePass<G, M>, SampleError> {
        assert!(limit > 0);
        if store.is_empty() {
            return Err(SampleError::EmptyStore);
        }
        let mut cost = CostByPurpose::default();
        let result = self.build_pass(store, limit, evaluator, &mut cost);
        match result {
            Ok(mut pass) => {
                pass.samples[0].cost.add(cost);
                for drawn in &pass.samples {
                    self.counters.record(drawn);
                }
                Ok(pass)
            }
            Err(error) => {
                self.counters.cost.add(cost);
                self.counters.failed_draws += 1;
                Err(error)
            }
        }
    }

    fn build_pass<const S: usize, const G: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        limit: usize,
        evaluator: &impl GoalEvaluator<S, G>,
        cost: &mut CostByPurpose,
    ) -> Result<EpisodePass<G, M>, SampleError> {
        let episode = store
            .episodes()
            .nth(self.rng.gen_range(store.episodes().count()))
            .expect("a nonempty store retains an episode");
        let config = self.configuration;
        let requested_relabel = config.selection.strategy != GoalStrategy::Original
            && (config.relabeled_proportion == 1.0
                || (config.relabeled_proportion > 0.0
                    && self.rng.gen_bool(config.relabeled_proportion)));
        let selection = Selection {
            strategy: GoalStrategy::Episode,
            ..config.selection
        };
        let selected = if requested_relabel {
            let distribution =
                goal_distribution(episode, 0, selection, store.candidates(), evaluator);
            cost.add(distribution.cost);
            distribution.draw(&mut self.rng)
        } else {
            None
        };
        let fallback = requested_relabel && selected.is_none();
        let (goal, origin, strategy) = match selected {
            Some(choice) => (choice.goal, SampleOrigin::Relabeled, GoalStrategy::Episode),
            None => (
                episode.start().desired,
                SampleOrigin::Original,
                GoalStrategy::Original,
            ),
        };
        let mut admitted = Vec::with_capacity(episode.len());
        for transition in 0..episode.len() {
            let (facts, queries) = goal_facts(episode, transition, &goal, evaluator);
            cost.add(if origin == SampleOrigin::Relabeled {
                config.selection.admissibility.attribute(queries)
            } else {
                queries.as_provenance()
            });
            if origin == SampleOrigin::Original || config.selection.admissibility.admits(facts) {
                admitted.push((transition, facts));
            } else if config.selection.admissibility == Admissibility::CounterfactualEpisode {
                break;
            }
        }
        assert!(
            !admitted.is_empty(),
            "an admitted pass or original fallback must be nonempty"
        );
        let available_steps = admitted.len();
        let mut samples = Vec::with_capacity(limit.min(available_steps));
        for (transition, facts) in admitted.into_iter().rev().take(limit) {
            let scored = build_sample(episode, transition, &goal, evaluator, config.truncation);
            cost.scoring
                .add(scoring_cost(&episode.steps()[transition].step));
            let scored = scored?;
            samples.push(DrawnSample {
                episode: episode.id(),
                transition,
                goal,
                scored,
                provenance: Provenance {
                    origin,
                    strategy,
                    requested_strategy: GoalStrategy::Episode,
                    already_reached: facts.already_reached,
                    after_counterfactual_end: facts.after_counterfactual_end,
                    done: scored.sample.done,
                    outside_candidates: !store.candidates().contains(&goal),
                    fallback,
                },
                cost: CostByPurpose::default(),
            });
        }
        Ok(EpisodePass {
            samples,
            available_steps,
            requested_relabel,
            fallback,
        })
    }
}
