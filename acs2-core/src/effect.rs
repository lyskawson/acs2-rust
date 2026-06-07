use crate::perception::Perception;
use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Effect<const N: usize> {
    pub symbols: [Symbol; N],
}

fn item_anticipates_change(item: Symbol, before: Symbol, after: Symbol) -> bool {
    if item.is_wildcard() {
        before == after
    } else {
        before != after && item == after
    }
}

impl<const N: usize> Effect<N> {
    pub fn all_wildcard() -> Self {
        Self {
            symbols: [Symbol::Wildcard; N],
        }
    }

    pub fn anticipates_correctly(&self, p0: &Perception<N>, p1: &Perception<N>) -> bool {
        (0..N).all(|index| item_anticipates_change(self.symbols[index], p0.symbols[index], p1.symbols[index]))
    }

    pub fn is_specializable(&self, p0: &Perception<N>, p1: &Perception<N>) -> bool {
        (0..N).all(|index| {
            let item = self.symbols[index];
            item.is_wildcard()
                || (item == p1.symbols[index] && p0.symbols[index] != p1.symbols[index])
        })
    }

    pub fn specify_change(&self) -> bool {
        self.symbols.iter().any(|symbol| !symbol.is_wildcard())
    }

    pub fn subsumes(&self, other: &Effect<N>) -> bool {
        self.symbols == other.symbols
    }

    pub fn get(&self, index: usize) -> Symbol {
        self.symbols[index]
    }

    pub fn set(&mut self, index: usize, symbol: Symbol) {
        self.symbols[index] = symbol;
    }
}
