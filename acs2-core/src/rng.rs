use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RngState {
    pub seed: [u8; 32],
    pub stream: u64,
    pub word_pos: u128,
}

pub trait RandomSource {
    fn gen_bool(&mut self, probability: f64) -> bool;
    fn gen_range(&mut self, bound: usize) -> usize;
    fn gen_unit(&mut self) -> f64;

    fn capture_state(&self) -> Option<RngState> {
        None
    }

    fn restore_state(&mut self, _state: &RngState) -> bool {
        false
    }
}

pub fn shuffle<T>(items: &mut [T], rng: &mut dyn RandomSource) {
    for index in (1..items.len()).rev() {
        let swap_with = rng.gen_range(index + 1);
        items.swap(index, swap_with);
    }
}

pub struct ChaChaRandomSource {
    inner: ChaCha8Rng,
}

impl ChaChaRandomSource {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    pub fn from_state(state: &RngState) -> Self {
        let mut inner = ChaCha8Rng::from_seed(state.seed);
        inner.set_stream(state.stream);
        inner.set_word_pos(state.word_pos);
        Self { inner }
    }
}

impl RandomSource for ChaChaRandomSource {
    fn gen_bool(&mut self, probability: f64) -> bool {
        self.inner.gen_bool(probability)
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        self.inner.gen_range(0..bound)
    }

    fn gen_unit(&mut self) -> f64 {
        self.inner.gen::<f64>()
    }

    fn capture_state(&self) -> Option<RngState> {
        Some(RngState {
            seed: self.inner.get_seed(),
            stream: self.inner.get_stream(),
            word_pos: self.inner.get_word_pos(),
        })
    }

    fn restore_state(&mut self, state: &RngState) -> bool {
        self.inner = ChaCha8Rng::from_seed(state.seed);
        self.inner.set_stream(state.stream);
        self.inner.set_word_pos(state.word_pos);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_captured_state_resumes_the_same_stream() {
        let mut original = ChaChaRandomSource::from_seed(42);
        for _ in 0..37 {
            original.gen_range(11);
        }
        let state = original.capture_state().expect("chacha exposes its state");
        let expected: Vec<usize> = (0..64).map(|_| original.gen_range(11)).collect();

        let mut restored = ChaChaRandomSource::from_state(&state);
        let replayed: Vec<usize> = (0..64).map(|_| restored.gen_range(11)).collect();
        assert_eq!(replayed, expected);

        let mut in_place = ChaChaRandomSource::from_seed(7);
        assert!(in_place.restore_state(&state));
        let overwritten: Vec<usize> = (0..64).map(|_| in_place.gen_range(11)).collect();
        assert_eq!(overwritten, expected);
    }

    #[test]
    fn a_capture_taken_mid_block_keeps_the_word_offset() {
        let mut original = ChaChaRandomSource::from_seed(3);
        original.gen_bool(0.5);
        let state = original.capture_state().unwrap();
        let expected: Vec<f64> = (0..8).map(|_| original.gen_unit()).collect();
        let mut restored = ChaChaRandomSource::from_state(&state);
        let replayed: Vec<f64> = (0..8).map(|_| restored.gen_unit()).collect();
        assert_eq!(replayed, expected);
    }

    #[test]
    fn every_word_offset_in_a_buffer_round_trips() {
        for consumed in 0..80usize {
            let mut original = ChaChaRandomSource::from_seed(11);
            for _ in 0..consumed {
                original.gen_unit();
            }
            let state = original.capture_state().unwrap();
            let expected: Vec<u64> = (0..4).map(|_| original.gen_range(usize::MAX) as u64).collect();
            let mut restored = ChaChaRandomSource::from_state(&state);
            let replayed: Vec<u64> = (0..4).map(|_| restored.gen_range(usize::MAX) as u64).collect();
            assert_eq!(replayed, expected, "diverged after {consumed} words");
        }
    }

    #[test]
    fn a_capture_carries_the_stream_as_well_as_the_position() {
        let mut original = ChaChaRandomSource::from_seed(5);
        original.inner.set_stream(0x0123_4567_89ab_cdef);
        original.gen_unit();
        let state = original.capture_state().unwrap();
        assert_eq!(state.stream, 0x0123_4567_89ab_cdef);
        let expected: Vec<f64> = (0..4).map(|_| original.gen_unit()).collect();

        let mut restored = ChaChaRandomSource::from_state(&state);
        assert_eq!(restored.capture_state().unwrap().stream, state.stream);
        assert_eq!((0..4).map(|_| restored.gen_unit()).collect::<Vec<_>>(), expected);

        let mut other_stream = ChaChaRandomSource::from_seed(5);
        other_stream.gen_unit();
        assert_ne!(
            (0..4).map(|_| other_stream.gen_unit()).collect::<Vec<_>>(),
            expected,
            "the stream must actually change the output, or this test proves nothing"
        );
    }
}
