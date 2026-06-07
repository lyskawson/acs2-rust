use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Perception<const N: usize> {
    pub symbols: [Symbol; N],
}

impl<const N: usize> Perception<N> {
    pub fn new(symbols: [Symbol; N]) -> Self {
        todo!()
    }

    pub fn get(&self, index: usize) -> Symbol {
        todo!()
    }
}
