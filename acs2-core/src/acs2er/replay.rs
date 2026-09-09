use std::collections::VecDeque;

use crate::perception::Perception;
use crate::rng::RandomSource;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct ReplaySample<const N: usize> {
    pub state: Perception<N>,
    pub action: usize,
    pub reward: f64,
    pub next_state: Perception<N>,
    pub done: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReplayConfiguration {
    pub buffer_size: usize,
    pub min_samples: usize,
    pub samples_number: usize,
}

impl ReplayConfiguration {
    pub fn default_protocol() -> Self {
        Self {
            buffer_size: 10_000,
            min_samples: 1_000,
            samples_number: 3,
        }
    }
}

impl Default for ReplayConfiguration {
    fn default() -> Self {
        Self::default_protocol()
    }
}

pub struct ReplayMemory<const N: usize> {
    samples: VecDeque<ReplaySample<N>>,
    max_size: usize,
}

impl<const N: usize> ReplayMemory<N> {
    pub fn new(max_size: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(max_size.min(1_024)),
            max_size,
        }
    }

    pub fn max_size(&self) -> usize {
        self.max_size
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn get(&self, index: usize) -> ReplaySample<N> {
        self.samples[index]
    }

    pub fn update(&mut self, sample: ReplaySample<N>) {
        if self.samples.len() >= self.max_size {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn sample_indices(&self, count: usize, rng: &mut dyn RandomSource) -> Vec<usize> {
        let len = self.samples.len();
        let wanted = count.min(len);
        let mut chosen: Vec<usize> = Vec::with_capacity(wanted);
        while chosen.len() < wanted {
            let candidate = rng.gen_range(len);
            if !chosen.contains(&candidate) {
                chosen.push(candidate);
            }
        }
        chosen
    }
}
