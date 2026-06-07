use crate::perception::Perception;
use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Condition<const N: usize> {
    pub symbols: [Symbol; N],
}

impl<const N: usize> Condition<N> {
    pub fn all_wildcard() -> Self {
        todo!()
    }

    pub fn does_match(&self, perception: &Perception<N>) -> bool {
        todo!()
    }

    pub fn subsumes(&self, other: &Condition<N>) -> bool {
        todo!()
    }

    pub fn specificity(&self) -> usize {
        todo!()
    }

    pub fn specialize_with(&mut self, other: &Condition<N>) {
        todo!()
    }

    pub fn generalize(&mut self, position: usize) {
        todo!()
    }

    pub fn get(&self, index: usize) -> Symbol {
        todo!()
    }

    pub fn set(&mut self, index: usize, symbol: Symbol) {
        todo!()
    }
}
