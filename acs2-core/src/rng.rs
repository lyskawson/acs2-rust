use rand_chacha::ChaCha8Rng;

pub trait RandomSource {
    fn gen_bool(&mut self, probability: f64) -> bool;
    fn gen_range(&mut self, bound: usize) -> usize;
}

pub fn shuffle<T>(items: &mut [T], rng: &mut dyn RandomSource) {
    todo!()
}

pub struct ChaChaRandomSource {
    inner: ChaCha8Rng,
}

impl ChaChaRandomSource {
    pub fn from_seed(seed: u64) -> Self {
        todo!()
    }
}

impl RandomSource for ChaChaRandomSource {
    fn gen_bool(&mut self, probability: f64) -> bool {
        todo!()
    }

    fn gen_range(&mut self, bound: usize) -> usize {
        todo!()
    }
}
