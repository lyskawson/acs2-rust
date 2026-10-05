use std::collections::VecDeque;
use std::mem::size_of;

use acs2_core::goal::{Goal, GoalStart, GoalStep};
use acs2_core::perception::Perception;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct EpisodeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StoredStep<const S: usize, const G: usize> {
    pub action: usize,
    pub step: GoalStep<S, G>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Episode<const S: usize, const G: usize> {
    start: GoalStart<S, G>,
    steps: Vec<StoredStep<S, G>>,
}

impl<const S: usize, const G: usize> Episode<S, G> {
    pub fn new(start: GoalStart<S, G>) -> Self {
        Self {
            start,
            steps: Vec::new(),
        }
    }

    pub fn push(&mut self, action: usize, step: GoalStep<S, G>) {
        assert!(
            self.steps
                .last()
                .is_none_or(|last| !last.step.terminal_state && !last.step.time_limit_reached),
            "a raw terminal or time-limit step must be last"
        );
        self.steps.push(StoredStep { action, step });
    }

    pub fn start(&self) -> &GoalStart<S, G> {
        &self.start
    }

    pub fn steps(&self) -> &[StoredStep<S, G>] {
        &self.steps
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredEpisode<const S: usize, const G: usize> {
    id: EpisodeId,
    episode: Episode<S, G>,
}

impl<const S: usize, const G: usize> StoredEpisode<S, G> {
    pub fn id(&self) -> EpisodeId {
        self.id
    }
    pub fn start(&self) -> &GoalStart<S, G> {
        self.episode.start()
    }
    pub fn steps(&self) -> &[StoredStep<S, G>] {
        self.episode.steps()
    }
    pub fn len(&self) -> usize {
        self.steps().len()
    }
    pub fn is_empty(&self) -> bool {
        self.steps().is_empty()
    }

    pub fn achieved(&self, state: usize) -> &Goal<G> {
        if state == 0 {
            &self.start().achieved
        } else {
            &self.steps()[state - 1].step.achieved
        }
    }

    pub fn observation(&self, state: usize) -> &Perception<S> {
        if state == 0 {
            &self.start().observation
        } else {
            &self.steps()[state - 1].step.observation
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreError {
    EmptyEpisode,
    EpisodeTooLong { steps: usize, capacity: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrajectoryStore<const S: usize, const G: usize> {
    capacity: usize,
    steps: usize,
    next_id: u64,
    episodes: VecDeque<StoredEpisode<S, G>>,
    candidates: Vec<Goal<G>>,
}

impl<const S: usize, const G: usize> TrajectoryStore<S, G> {
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "trajectory capacity must be positive");
        Self {
            capacity,
            steps: 0,
            next_id: 0,
            episodes: VecDeque::new(),
            candidates: Vec::new(),
        }
    }

    pub fn begin_episode(&mut self, start: GoalStart<S, G>) -> Episode<S, G> {
        self.observe_start(&start);
        Episode::new(start)
    }

    fn observe_start(&mut self, start: &GoalStart<S, G>) {
        if let Err(index) = self.candidates.binary_search(&start.desired) {
            self.candidates.insert(index, start.desired);
        }
    }

    pub fn insert(&mut self, episode: Episode<S, G>) -> Result<EpisodeId, StoreError> {
        let len = episode.steps.len();
        if len == 0 {
            return Err(StoreError::EmptyEpisode);
        }
        self.observe_start(episode.start());
        if len > self.capacity {
            return Err(StoreError::EpisodeTooLong {
                steps: len,
                capacity: self.capacity,
            });
        }
        let id = EpisodeId(self.next_id);
        self.next_id = self
            .next_id
            .checked_add(1)
            .expect("episode identifiers exhausted");
        while self.steps > self.capacity - len {
            self.steps -= self.episodes.pop_front().expect("stored step count").len();
        }
        self.steps += len;
        self.episodes.push_back(StoredEpisode { id, episode });
        Ok(id)
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn len(&self) -> usize {
        self.steps
    }
    pub fn is_empty(&self) -> bool {
        self.steps == 0
    }
    pub fn candidates(&self) -> &[Goal<G>] {
        &self.candidates
    }
    pub fn episodes(&self) -> impl Iterator<Item = &StoredEpisode<S, G>> {
        self.episodes.iter()
    }
    pub fn episode(&self, id: EpisodeId) -> Option<&StoredEpisode<S, G>> {
        self.episodes.iter().find(|episode| episode.id == id)
    }

    pub fn transition(&self, mut index: usize) -> (&StoredEpisode<S, G>, usize) {
        assert!(index < self.steps, "stored transition index out of bounds");
        for episode in &self.episodes {
            if index < episode.len() {
                return (episode, index);
            }
            index -= episode.len();
        }
        unreachable!("stored step count")
    }

    pub fn logical_bytes(&self) -> usize {
        size_of::<Self>()
            + self.episodes.len() * size_of::<StoredEpisode<S, G>>()
            + self.steps * size_of::<StoredStep<S, G>>()
            + self.candidates.len() * size_of::<Goal<G>>()
    }
}
