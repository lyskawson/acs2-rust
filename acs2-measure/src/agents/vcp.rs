use std::mem::size_of;

use acs2_agents::acs2vcp::{self, priority::DistributionSummary};
use acs2_core::representation::{classifier::Classifier, perception::Perception, symbol::Symbol};
use acs2_core::support::{
    knowledge::{anticipation_fraction, Transition},
    rng::RandomSource,
};
use serde_json::{json, Value};

use super::EpisodeAdapter;
use crate::runner::{GoalAgent, MeasuredEnvironment};

pub struct VcpGoalAgent<const S: usize, const G: usize, const M: usize, R, P, A> {
    pub agent: acs2vcp::Agent<S, G, M, R, P, A>,
    pub preset: acs2vcp::Preset,
}

fn summary(value: &DistributionSummary) -> Value {
    let count = value.count as f64;
    let mean = (value.count > 0).then(|| value.sum / count);
    json!({"count": value.count, "sum": value.sum, "sum_squares": value.sum_squares,
        "mean": mean, "minimum": (value.count > 0).then_some(value.minimum),
        "maximum": (value.count > 0).then_some(value.maximum),
        "standard_deviation": mean.map(|mean| (value.sum_squares / count - mean * mean).max(0.0).sqrt()),
        "histogram": value.histogram, "histogram_bins": "sixteen_equal_bins_over_zero_to_variance_maximum_last_includes_upper_bound"})
}

fn marks<const M: usize>(
    population: &acs2_core::representation::population::Population<M>,
) -> usize {
    population
        .classifiers()
        .iter()
        .map(|classifier| {
            classifier
                .mark
                .attributes
                .iter()
                .map(|attribute| attribute.len())
                .sum::<usize>()
        })
        .sum()
}

impl<
        const S: usize,
        const G: usize,
        const M: usize,
        R: RandomSource,
        P: RandomSource,
        A: RandomSource,
    > GoalAgent<S, G, M> for VcpGoalAgent<S, G, M, R, P, A>
{
    fn name(&self) -> &'static str {
        self.preset.name()
    }
    fn train_episode<E: MeasuredEnvironment<S, G, M>>(&mut self, env: &mut E, time: u64) {
        self.agent
            .train_episode(&mut EpisodeAdapter::<_, M>(env), time);
    }
    fn declared_policy(&self) -> &'static str {
        "mean_of_heads_greedy_change_anticipating_success"
    }
    fn evaluation_heads(&self) -> usize {
        self.agent.heads().len()
    }
    fn eval_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        assert_eq!(
            self.evaluation_heads(),
            1,
            "ensemble evaluation must name its head"
        );
        self.agent.greedy_action(0, state, rng)
    }
    fn eval_head_action(
        &self,
        head: usize,
        state: &Perception<M>,
        rng: &mut dyn RandomSource,
    ) -> (usize, f64) {
        self.agent.greedy_action(head, state, rng)
    }
    fn online_updates(&self) -> u64 {
        0
    }
    fn replay_updates(&self) -> u64 {
        let counts = self.agent.counts();
        counts.full + counts.values_only
    }
    fn learning_update_counts(&self) -> Option<Value> {
        let counts = self.agent.counts();
        Some(
            json!({"full": counts.full, "values_only": counts.values_only, "skipped_warmup_steps": counts.skipped_warmup_steps}),
        )
    }
    fn replay_samples(&self) -> usize {
        0
    }
    fn replay_logical_bytes(&self) -> usize {
        0
    }
    fn trajectory_logical_bytes(&self) -> usize {
        self.agent.store().logical_bytes()
    }
    fn population_classifiers(&self) -> usize {
        self.agent
            .heads()
            .iter()
            .map(|head| head.population().len())
            .sum()
    }
    fn population_numerosity(&self) -> u32 {
        self.agent
            .heads()
            .iter()
            .map(|head| head.population().numerosity())
            .sum()
    }
    fn population_logical_bytes(&self) -> usize {
        self.population_classifiers() * size_of::<Classifier<M>>()
    }
    fn population_mark_entries(&self) -> usize {
        self.agent
            .heads()
            .iter()
            .map(|head| marks(head.population()))
            .sum()
    }
    fn knowledge(&self, transitions: &[Transition<M>]) -> Option<f64> {
        let sum: f64 = self
            .agent
            .heads()
            .iter()
            .map(|head| {
                anticipation_fraction(
                    head.population(),
                    self.agent.config().theta_r,
                    transitions
                        .iter()
                        .map(|t| Transition::new(t.p0, t.action, t.p1)),
                )
            })
            .sum();
        Some(sum / self.evaluation_heads() as f64)
    }
    fn ensemble_diagnostics(&self, transitions: Option<&[Transition<M>]>) -> Option<Value> {
        let mut total_head_bytes = 0;
        let heads: Vec<_> = self.agent.heads().iter().enumerate().map(|(index, head)| {
            let population = head.population();
            let mark_entries = marks(population);
            let population_bytes = population.len() * size_of::<Classifier<M>>();
            let known_bytes = population_bytes + mark_entries * size_of::<Symbol>();
            let head_bytes = known_bytes + size_of::<acs2vcp::Head<M, R>>();
            total_head_bytes += head_bytes;
            let counts = head.counts();
            let knowledge = transitions.map(|transitions| anticipation_fraction(population, self.agent.config().theta_r,
                transitions.iter().map(|t| Transition::new(t.p0, t.action, t.p1))));
            json!({"head": index, "population_classifiers": population.len(), "population_numerosity": population.numerosity(),
                "population_logical_bytes": population_bytes, "population_mark_entries": mark_entries,
                "population_known_bytes_lower_bound": known_bytes, "head_known_bytes_lower_bound": head_bytes,
                "knowledge": knowledge, "episodes": head.episodes(), "environment_steps": counts.environment_steps,
                "full": counts.full, "values_only": counts.values_only, "skipped_warmup_steps": counts.skipped_warmup_steps,
                "random_stream": head.random_source().capture_state().map(|state| state.stream)})
        }).collect();
        Some(
            json!({"heads": heads, "head_count": heads.len(), "heads_known_bytes_lower_bound": total_head_bytes,
            "mean_population_classifiers": self.population_classifiers() as f64 / heads.len() as f64,
            "mean_population_numerosity": f64::from(self.population_numerosity()) / heads.len() as f64,
            "agent_known_bytes_lower_bound": total_head_bytes + self.trajectory_logical_bytes()
                + size_of::<acs2vcp::Agent<S, G, M, R, P, A>>() - size_of::<acs2_trajectory::TrajectoryStore<S, G>>(),
            "pending_sample_bytes": 0, "batch_peak_bytes": self.agent.priority_counts().batch_peak_bytes,
            "memory_scope": "logical_payload_and_inline_headers_excludes_unused_capacity_and_allocator_overhead"}),
        )
    }
    fn replay_diagnostics(&self) -> Option<Value> {
        let agent = &self.agent;
        let mut accepted = agent.accepted_counters().clone();
        accepted.cost = agent.sampler().counters().cost;
        let mut diagnostics = crate::trajectory::replay_diagnostics(&accepted);
        let priority = agent.priority_counts();
        let proposals = priority.proposals as f64;
        diagnostics["realized_relabeled_share"] =
            json!((priority.accepted > 0)
                .then(|| accepted.relabeled.draws as f64 / priority.accepted as f64));
        diagnostics["stored_steps"] = json!(agent.store().len());
        diagnostics["stored_episodes"] = json!(agent.store().episodes().count());
        diagnostics["proposal_routes"] =
            crate::trajectory::replay_diagnostics(agent.sampler().counters());
        diagnostics["priority"] = json!({"proposals": priority.proposals, "accepted_draws": priority.accepted,
            "proposals_per_draw": (priority.accepted > 0).then(|| proposals / priority.accepted as f64),
            "acceptance_share": (priority.proposals > 0).then(|| priority.accepted as f64 / proposals),
            "mean_acceptance_probability": (priority.proposals > 0).then(|| priority.acceptance_sum / proposals),
            "variance_maximum": agent.parameters().variance_maximum(), "variance_maximum_definition": "reward_scale_bound",
            "proposal_variance": summary(&priority.variance), "accepted_variance": summary(&priority.accepted_variance),
            "value_queries": priority.value_queries, "acceptance_coins": priority.acceptance_coins,
            "candidate_acceptance_sums": priority.candidate_acceptance_sums,
            "candidate_temperatures": (1..=acs2vcp::priority::MAX_TEMPERATURE).collect::<Vec<_>>(),
            "match_formation_identity": "environment_steps + 3 * full + 2 * values_only + priority_value_queries"});
        Some(diagnostics)
    }
    fn agent_parameters(&self) -> Value {
        let parameters = self.agent.parameters();
        let replay = self.agent.replay_configuration();
        json!({"replay": true, "schedule": "after_complete_episode", "heads": parameters.heads,
            "temperature": parameters.temperature, "prioritized": parameters.temperature > 0,
            "replay_capacity": replay.capacity, "replay_warmup": replay.warmup,
            "replay_updates_per_step": replay.updates_per_step, "strategy": "future",
            "admissibility": format!("{:?}", parameters.hindsight.admissibility), "candidate_filter": false,
            "copy_update": format!("{:?}", parameters.hindsight.copy_update),
            "relabeled_proportion": parameters.hindsight.relabeled_proportion, "truncation": format!("{:?}", replay.truncation),
            "sampler_truncation": format!("{:?}", self.agent.sampler().configuration().truncation),
            "sampler_stream": acs2_trajectory::SAMPLER_STREAM, "acceptance_stream": acs2vcp::ACCEPTANCE_STREAM,
            "extra_head_stream_start": acs2vcp::EXTRA_HEAD_STREAM, "priority_epsilon": acs2vcp::priority::EPSILON,
            "variance_maximum": parameters.variance_maximum(), "variance_maximum_definition": "reward_scale_bound",
            "draws": "proposal_reweighted_by_fresh_consistency_priority", "importance_weights": "none",
            "priority_clock": "all_draws_before_any_learning_in_episode", "learning_clock": "episode_end_environment_steps",
            "acting_head": "episode_index_modulo_head_count", "learning_head": "acting_head"})
    }
}
