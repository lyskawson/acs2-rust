use std::collections::{BTreeMap, BTreeSet};
use std::mem::size_of;

use acs2_core::checkpoint::AgentState;
use acs2_core::ports::goal::Goal;
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_core::trial::TruncationMode;
use acs2_trajectory::{DrawRoute, EpisodeId, GoalEvaluator, SampleOrigin, TrajectoryStore};

use crate::acs2her::per_goal_reward;
use crate::episodic_replay::{
    EpisodicAgent, EpisodicConfiguration, EpisodicReplay, ReplayEnvironment,
};

use utility::{first_reach, sample_candidates, ObservationMatches, Scoring};

mod utility;

#[cfg(test)]
mod tests;

pub const ROUTE_STREAM: u64 = 10;
pub const CANDIDATE_STREAM: u64 = 11;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Measure {
    MeanValue,
    TrialUpdate,
    RandomGoal,
}

impl Measure {
    pub fn name(self) -> &'static str {
        match self {
            Self::MeanValue => "acs2tuher_mean_value",
            Self::TrialUpdate => "acs2tuher_trial_update",
            Self::RandomGoal => "acs2tuher_random_goal",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UtilityConfiguration {
    pub measure: Measure,
    pub candidates: usize,
    pub refresh_steps: Option<u64>,
}

impl UtilityConfiguration {
    pub fn pilot(measure: Measure) -> Self {
        Self {
            measure,
            candidates: 8,
            refresh_steps: Some(2_000),
        }
    }

    pub fn validate(self) {
        assert!(
            matches!(self.candidates, 4 | 8),
            "TU candidates must be 4 or 8"
        );
        assert!(
            matches!(self.refresh_steps, None | Some(2_000 | 5_000)),
            "TU refresh must be none, 2000 or 5000 steps"
        );
    }
}

pub struct UtilityRandom<P> {
    pub routes: P,
    pub candidates: P,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cost {
    pub cpu_nanoseconds: u64,
    pub scoring_cpu_nanoseconds: u64,
    pub insertion_cpu_nanoseconds: u64,
    pub insertion_scoring_cpu_nanoseconds: u64,
    pub insertions: u64,
    pub refreshes: u64,
    pub candidates_sampled: u64,
    pub candidates_scored: u64,
    pub candidates_at_four: u64,
    pub insertion_candidates: u64,
    pub insertion_candidates_at_four: u64,
    pub match_sets: u64,
    pub observation_match_sets: u64,
    pub value_reads: u64,
    pub trial_value_updates: u64,
    pub trial_reward_entries: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ByRoute {
    pub original: u64,
    pub hindsight: u64,
    pub trajectory_utility: u64,
}

impl ByRoute {
    fn record<const G: usize>(&mut self, route: DrawRoute<G>, count: u64) {
        match route {
            DrawRoute::Original => self.original += count,
            DrawRoute::Hindsight => self.hindsight += count,
            DrawRoute::TrajectoryUtility(_) => self.trajectory_utility += count,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measurements {
    pub configuration: UtilityConfiguration,
    pub cost: Cost,
    pub buffer_entries: usize,
    pub buffer_logical_bytes: usize,
    pub draw_requests: ByRoute,
    pub hindsight_fallbacks: u64,
    pub tu_fallbacks: u64,
    pub reward_updates: ByRoute,
    pub assigned: u64,
    pub assigned_goals_reached: u64,
    pub distinct_assigned_goals: usize,
    pub minimum_updates_per_goal: u64,
    pub median_updates_per_goal: f64,
    pub maximum_updates_per_goal: u64,
    pub mean_post_update_level: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
struct Assignment<const G: usize> {
    goal: Option<Goal<G>>,
    evaluated_at: u64,
}

struct Utility<const G: usize, P> {
    configuration: UtilityConfiguration,
    rng: UtilityRandom<P>,
    assignments: BTreeMap<EpisodeId, Assignment<G>>,
    cost: Cost,
    draw_requests: ByRoute,
    hindsight_fallbacks: u64,
    tu_fallbacks: u64,
    reward_updates: ByRoute,
    updates_per_goal: BTreeMap<Goal<G>, u64>,
    assigned: u64,
    assigned_goals_reached: u64,
    distinct_assigned_goals: BTreeSet<Goal<G>>,
    post_update_level_sum: f64,
}

impl<const G: usize, P: RandomSource> Utility<G, P> {
    fn new(configuration: UtilityConfiguration, rng: UtilityRandom<P>, goals: &[Goal<G>]) -> Self {
        configuration.validate();
        Self {
            configuration,
            rng,
            assignments: BTreeMap::new(),
            cost: Cost::default(),
            draw_requests: ByRoute::default(),
            hindsight_fallbacks: 0,
            tu_fallbacks: 0,
            reward_updates: ByRoute::default(),
            updates_per_goal: goals.iter().map(|&goal| (goal, 0)).collect(),
            assigned: 0,
            assigned_goals_reached: 0,
            distinct_assigned_goals: BTreeSet::new(),
            post_update_level_sum: 0.0,
        }
    }

    fn on_insertion<const S: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        episode: EpisodeId,
        evaluator: &impl GoalEvaluator<S, G>,
        scoring: &Scoring<G, M>,
        time: u64,
    ) {
        let first = store
            .episodes()
            .next()
            .expect("newly retained episode")
            .id();
        while self
            .assignments
            .first_key_value()
            .is_some_and(|(&id, _)| id < first)
        {
            self.assignments.pop_first();
        }
        self.evaluate(store, episode, evaluator, scoring, time, true);
    }

    fn assigned_goal<const S: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        episode: EpisodeId,
        evaluator: &impl GoalEvaluator<S, G>,
        scoring: &Scoring<G, M>,
        time: u64,
    ) -> Option<Goal<G>> {
        let last = self.assignments[&episode];
        if self
            .configuration
            .refresh_steps
            .is_some_and(|steps| time - last.evaluated_at > steps)
        {
            self.evaluate(store, episode, evaluator, scoring, time, false);
        }
        self.assignments[&episode].goal
    }

    fn evaluate<const S: usize, const M: usize>(
        &mut self,
        store: &TrajectoryStore<S, G>,
        id: EpisodeId,
        evaluator: &impl GoalEvaluator<S, G>,
        scoring: &Scoring<G, M>,
        time: u64,
        insertion: bool,
    ) {
        let started = thread_cpu_nanoseconds();
        let episode = store.episode(id).expect("retained episode");
        let candidates = sample_candidates(
            episode,
            store.candidates(),
            self.configuration.candidates,
            evaluator,
            &mut self.rng.candidates,
        );
        let sampled = candidates.len() as u64;
        self.cost.candidates_sampled += sampled;
        self.cost.candidates_at_four += sampled.min(4);
        if insertion {
            self.cost.insertions += 1;
            self.cost.insertion_candidates += sampled;
            self.cost.insertion_candidates_at_four += sampled.min(4);
        } else {
            self.cost.refreshes += 1;
        }
        let measure = self.configuration.measure;
        let mut scoring_elapsed = 0;
        let selected = if measure == Measure::RandomGoal || candidates.is_empty() {
            candidates
                .first()
                .map(|&goal| (goal, first_reach(episode, &goal, evaluator).is_some()))
        } else {
            let reaches = candidates
                .iter()
                .map(|goal| first_reach(episode, goal, evaluator))
                .collect::<Vec<_>>();
            let states = match measure {
                Measure::MeanValue => reaches
                    .iter()
                    .map(|reach| reach.unwrap_or(episode.len()))
                    .max()
                    .unwrap(),
                Measure::TrialUpdate => episode.len() + 1,
                Measure::RandomGoal => unreachable!(),
            };
            let observations =
                ObservationMatches::new(episode, states, scoring.population, &mut self.cost);
            let scoring_started = thread_cpu_nanoseconds();
            let mut best = None;
            for (goal, reach) in candidates.into_iter().zip(reaches) {
                let score = scoring.score(
                    measure,
                    episode,
                    (goal, reach.unwrap_or(episode.len())),
                    &observations,
                    evaluator,
                    &mut self.cost,
                );
                self.post_update_level_sum += score.post_update_level.unwrap_or(0.0);
                if best.is_none_or(|(previous, _)| score.utility > previous) {
                    best = Some((score.utility, (goal, reach.is_some())));
                }
            }
            scoring_elapsed = thread_cpu_nanoseconds() - scoring_started;
            best.map(|(_, choice)| choice)
        };
        if let Some((goal, reached)) = selected {
            self.assigned += 1;
            self.assigned_goals_reached += u64::from(reached);
            self.distinct_assigned_goals.insert(goal);
        }
        self.assignments.insert(
            id,
            Assignment {
                goal: selected.map(|(goal, _)| goal),
                evaluated_at: time,
            },
        );
        let elapsed = thread_cpu_nanoseconds() - started;
        self.cost.cpu_nanoseconds += elapsed;
        self.cost.scoring_cpu_nanoseconds += scoring_elapsed;
        if insertion {
            self.cost.insertion_cpu_nanoseconds += elapsed;
            self.cost.insertion_scoring_cpu_nanoseconds += scoring_elapsed;
        }
    }

    fn measurements(&self) -> Measurements {
        let mut updates = self.updates_per_goal.values().copied().collect::<Vec<_>>();
        updates.sort_unstable();
        let middle = updates.len() / 2;
        Measurements {
            configuration: self.configuration,
            cost: self.cost,
            buffer_entries: self.assignments.len(),
            buffer_logical_bytes: size_of::<BTreeMap<EpisodeId, Assignment<G>>>()
                + self.assignments.len() * size_of::<(EpisodeId, Assignment<G>)>(),
            draw_requests: self.draw_requests,
            hindsight_fallbacks: self.hindsight_fallbacks,
            tu_fallbacks: self.tu_fallbacks,
            reward_updates: self.reward_updates,
            assigned: self.assigned,
            assigned_goals_reached: self.assigned_goals_reached,
            distinct_assigned_goals: self.distinct_assigned_goals.len(),
            minimum_updates_per_goal: updates[0],
            median_updates_per_goal: (updates[(updates.len() - 1) / 2] as f64
                + updates[middle] as f64)
                / 2.0,
            maximum_updates_per_goal: updates[updates.len() - 1],
            mean_post_update_level: (self.configuration.measure == Measure::TrialUpdate
                && self.cost.candidates_scored > 0)
                .then(|| self.post_update_level_sum / self.cost.candidates_scored as f64),
        }
    }
}

fn thread_cpu_nanoseconds() -> u64 {
    let mut time = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut time) };
    assert_eq!(result, 0, "thread CPU time is unavailable");
    time.tv_sec as u64 * 1_000_000_000 + time.tv_nsec as u64
}

pub struct Agent<const S: usize, const G: usize, const M: usize, R, P> {
    base: per_goal_reward::Agent<S, G, M, R, P>,
    utility: Option<Utility<G, P>>,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    Agent<S, G, M, R, P>
{
    pub fn new(
        config: Configuration,
        replay: EpisodicConfiguration,
        goals: &[Goal<G>],
        policy_rng: R,
        sampler_rng: P,
        utility: Option<(UtilityConfiguration, UtilityRandom<P>)>,
    ) -> Self {
        assert!(!config.do_ga, "ACS2-TUHER requires GA disabled");
        if utility.is_some() {
            assert_eq!(replay.truncation, TruncationMode::Bootstrap);
        }
        Self {
            base: per_goal_reward::Agent::new(config, replay, goals, policy_rng, sampler_rng),
            utility: utility.map(|(config, rng)| Utility::new(config, rng, goals)),
        }
    }

    pub fn restore(&mut self, _state: AgentState<M>) {
        panic!("ACS2-TUHER does not support restore");
    }

    pub fn snapshot(&self) -> String {
        let base = self.base.learner().snapshot();
        let Some(utility) = &self.utility else {
            return base;
        };
        let mut measurements = utility.measurements();
        measurements.cost.cpu_nanoseconds = 0;
        measurements.cost.scoring_cpu_nanoseconds = 0;
        measurements.cost.insertion_cpu_nanoseconds = 0;
        measurements.cost.insertion_scoring_cpu_nanoseconds = 0;
        format!(
            "{base}|{measurements:?}|{:?}|{:?}|{:?}|{:?}|{:?}",
            utility.assignments,
            utility.distinct_assigned_goals,
            utility.updates_per_goal,
            utility.rng.routes.capture_state(),
            utility.rng.candidates.capture_state()
        )
    }
}

fn scoring<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>(
    learner: &EpisodicReplay<S, G, M, R, P>,
) -> Scoring<'_, G, M> {
    Scoring {
        population: learner.population(),
        rewards: learner.per_goal_rewards().expect("per-goal rewards"),
        config: learner.config(),
    }
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource, P: RandomSource>
    EpisodicAgent<S, G, M> for Agent<S, G, M, R, P>
{
    type PolicyRandom = R;
    type SamplerRandom = P;

    fn name(&self) -> &'static str {
        self.utility.as_ref().map_or(self.base.name(), |utility| {
            utility.configuration.measure.name()
        })
    }

    fn learner(&self) -> &EpisodicReplay<S, G, M, R, P> {
        self.base.learner()
    }

    fn learner_mut(&mut self) -> &mut EpisodicReplay<S, G, M, R, P> {
        self.base.learner_mut()
    }

    fn train_episode(&mut self, env: &mut impl ReplayEnvironment<S, G, M>, time: u64) {
        let Some(utility) = &mut self.utility else {
            self.base.train_episode(env, time);
            return;
        };
        let learner = self.base.learner_mut();
        let (inserted, steps) = learner.collect_episode(env);
        let now = time + steps as u64;
        utility.on_insertion(learner.store(), inserted, env, &scoring(learner), now);
        if !learner.ready_for_replay(steps) {
            return;
        }
        let draws = steps
            .checked_mul(learner.replay_configuration().updates_per_step)
            .expect("replay draw count");
        for _ in 0..draws {
            let transition = learner.select_transition();
            let route = match utility.rng.routes.gen_range(3) {
                0 => DrawRoute::Original,
                1 => DrawRoute::Hindsight,
                _ => DrawRoute::TrajectoryUtility(utility.assigned_goal(
                    learner.store(),
                    transition.0,
                    env,
                    &scoring(learner),
                    now,
                )),
            };
            let sample = learner.draw_routed(transition, env, route);
            utility.draw_requests.record(route, 1);
            if sample.provenance.fallback {
                match route {
                    DrawRoute::Original => unreachable!("the original route cannot fall back"),
                    DrawRoute::Hindsight => utility.hindsight_fallbacks += 1,
                    DrawRoute::TrajectoryUtility(_) => utility.tu_fallbacks += 1,
                }
            }
            let before = scoring(learner).rewards.updates();
            learner.observe_and_learn(&sample, now);
            let after = scoring(learner).rewards.updates();
            let (origin, writes) = match sample.provenance.origin {
                SampleOrigin::Original => (DrawRoute::Original, after.original - before.original),
                SampleOrigin::Relabeled => (route, after.relabeled - before.relabeled),
            };
            utility.reward_updates.record(origin, writes);
            *utility
                .updates_per_goal
                .get_mut(&sample.goal)
                .expect("registered write goal") += writes;
        }
    }

    fn trajectory_utility(&self) -> Option<Measurements> {
        self.utility.as_ref().map(Utility::measurements)
    }
}
