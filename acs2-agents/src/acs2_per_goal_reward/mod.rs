use crate::per_goal_reward::{learning, PerGoalRewards};
use acs2_core::checkpoint::{AgentState, Checkpointed};
use acs2_core::learning::replay::ReplaySample;
use acs2_core::ports::environment::Environment;
use acs2_core::ports::goal::{Goal, GoalLayout};
use acs2_core::representation::perception::Perception;
use acs2_core::representation::population::{ClassifierRef, Population};
use acs2_core::support::config::Configuration;
use acs2_core::support::rng::RandomSource;
use acs2_core::trial::{TrialMetrics, TruncationMode};
use acs2_trajectory::SampleOrigin;

pub struct Agent<const S: usize, const G: usize, const M: usize, R: RandomSource> {
    population: Population<M>,
    config: Configuration,
    rng: R,
    rewards: PerGoalRewards<G>,
    truncation: TruncationMode,
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource> Agent<S, G, M, R> {
    pub fn new(config: Configuration, goals: &[Goal<G>], rng: R) -> Self {
        let rewards = PerGoalRewards::new(goals, &config, 0);
        Self {
            population: Population::new(),
            config,
            rng,
            rewards,
            truncation: TruncationMode::default(),
        }
    }

    pub fn with_population(mut self, population: Population<M>) -> Self {
        self.rewards
            .initialize_general(population.len(), &self.config);
        self.population = population;
        self
    }

    pub fn with_truncation_mode(mut self, mode: TruncationMode) -> Self {
        self.truncation = mode;
        self
    }

    pub fn population(&self) -> &Population<M> {
        &self.population
    }
    pub fn config(&self) -> &Configuration {
        &self.config
    }
    pub fn rewards(&self) -> &PerGoalRewards<G> {
        &self.rewards
    }
    pub fn rewards_mut(&mut self) -> &mut PerGoalRewards<G> {
        &mut self.rewards
    }
    pub fn snapshot(&self) -> String {
        format!(
            "{:?}|{:?}|{:?}",
            self.population.classifiers(),
            self.rewards,
            self.rng.capture_state()
        )
    }

    pub fn greedy_action(&self, state: &Perception<M>, rng: &mut dyn RandomSource) -> (usize, f64) {
        let goal = GoalLayout::<S, G, M>::split(state).1;
        let matched = self.population.form_match_set(state);
        let action = self.rewards.best_action(
            &self.population,
            &matched,
            &goal,
            self.config.number_of_possible_actions,
            rng,
        );
        let action_set = self.population.form_action_set(&matched, action);
        (
            action,
            self.rewards.bootstrap(&self.population, &action_set, &goal),
        )
    }

    fn learn(
        &mut self,
        sample: &ReplaySample<M>,
        action_set: Vec<ClassifierRef>,
        time: u64,
    ) -> Vec<ClassifierRef> {
        let goal = GoalLayout::<S, G, M>::split(&sample.state).1;
        self.rewards.goal_index(&goal);
        assert_eq!(goal, GoalLayout::<S, G, M>::split(&sample.next_state).1);
        let mut sets = learning::AlpSets {
            action_set,
            next_matched: if sample.done {
                Vec::new()
            } else {
                self.population.form_match_set(&sample.next_state)
            },
        };
        learning::apply_alp(
            &mut self.rewards,
            &mut self.population,
            &mut sets,
            sample,
            time,
            &self.config,
            &mut self.rng,
        );
        let matched = if sample.done {
            Vec::new()
        } else {
            self.population.form_match_set(&sample.next_state)
        };
        let bootstrap = if sample.done {
            0.0
        } else {
            self.rewards.bootstrap(&self.population, &matched, &goal)
        };
        self.rewards.update_values(
            &self.config,
            &sets.action_set,
            &goal,
            sample.reward,
            bootstrap,
            SampleOrigin::Original,
        );
        matched
    }

    pub fn train_episode<E: Environment<M>>(&mut self, env: &mut E, time: u64) -> TrialMetrics {
        let mut state = env.reset();
        let mut steps = 0;
        let mut total_reward = 0.0;
        let mut previous: Option<(ReplaySample<M>, Vec<ClassifierRef>)> = None;
        loop {
            let matched = if let Some((mut sample, action_set)) = previous.take() {
                sample.next_state = state;
                self.learn(&sample, action_set, time + u64::from(steps))
            } else {
                self.population.form_match_set(&state)
            };
            let goal = GoalLayout::<S, G, M>::split(&state).1;
            let action = if self.rng.gen_bool(self.config.epsilon) {
                self.rng.gen_range(self.config.number_of_possible_actions)
            } else {
                self.rewards.best_action(
                    &self.population,
                    &matched,
                    &goal,
                    self.config.number_of_possible_actions,
                    &mut self.rng,
                )
            };
            let action_set = self.population.form_action_set(&matched, action);
            let outcome = env.step(action);
            total_reward += outcome.reward;
            steps += 1;
            let sample = ReplaySample {
                state,
                action,
                next_state: outcome.observation,
                reward: outcome.reward,
                done: self
                    .truncation
                    .is_terminal(outcome.terminated, outcome.truncated),
            };
            state = outcome.observation;
            if outcome.terminated || outcome.truncated {
                self.learn(&sample, action_set, time + u64::from(steps));
                break;
            }
            previous = Some((sample, action_set));
        }
        TrialMetrics {
            steps,
            reward: total_reward,
        }
    }
}

impl<const S: usize, const G: usize, const M: usize, R: RandomSource> Checkpointed<M>
    for Agent<S, G, M, R>
{
    fn capture(&self) -> AgentState<M> {
        panic!("ACS2 per-goal rewards do not support checkpoints")
    }
    fn restore(&mut self, _state: AgentState<M>) {
        panic!("ACS2 per-goal rewards do not support restore")
    }
}

#[cfg(test)]
mod tests;
