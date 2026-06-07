use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

pub trait RandomSource {
    fn gen_bool(&mut self, probability: f64) -> bool;
    fn gen_range(&mut self, bound: usize) -> usize;
    fn gen_unit(&mut self) -> f64;
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
}
