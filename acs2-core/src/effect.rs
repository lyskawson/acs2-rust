use crate::perception::Perception;
use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Effect<const N: usize> {
    pub symbols: [Symbol; N],
}

impl<const N: usize> Effect<N> {
    pub fn all_wildcard() -> Self {
        todo!()
    }

    pub fn anticipates_correctly(&self, p0: &Perception<N>, p1: &Perception<N>) -> bool {
        todo!()
    }

    pub fn is_specializable(&self, p0: &Perception<N>, p1: &Perception<N>) -> bool {
        todo!()
    }

    pub fn specify_change(&self) -> bool {
        todo!()
    }

    pub fn subsumes(&self, other: &Effect<N>) -> bool {
        todo!()
    }

    pub fn get(&self, index: usize) -> Symbol {
        todo!()
    }

    pub fn set(&mut self, index: usize, symbol: Symbol) {
        todo!()
    }
}
