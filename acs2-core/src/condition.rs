use crate::perception::Perception;
use crate::rng::RandomSource;
use crate::symbol::Symbol;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Condition<const N: usize> {
    pub symbols: [Symbol; N],
}

fn symbols_compatible(left: Symbol, right: Symbol) -> bool {
    left.is_wildcard() || right.is_wildcard() || left == right
}

impl<const N: usize> Condition<N> {
    pub fn all_wildcard() -> Self {
        Self {
            symbols: [Symbol::Wildcard; N],
        }
    }

    pub fn does_match(&self, perception: &Perception<N>) -> bool {
        (0..N).all(|index| symbols_compatible(self.symbols[index], perception.symbols[index]))
    }

    pub fn subsumes(&self, other: &Condition<N>) -> bool {
        (0..N).all(|index| symbols_compatible(self.symbols[index], other.symbols[index]))
    }

    pub fn specificity(&self) -> usize {
        self.symbols.iter().filter(|symbol| !symbol.is_wildcard()).count()
    }

    pub fn specialize_with(&mut self, other: &Condition<N>) {
        for index in 0..N {
            if !other.symbols[index].is_wildcard() {
                self.symbols[index] = other.symbols[index];
            }
        }
    }

    pub fn generalize(&mut self, position: usize) {
        self.symbols[position] = Symbol::Wildcard;
    }

    pub fn generalize_specific_attribute_randomly(&mut self, rng: &mut dyn RandomSource) {
        let specific: Vec<usize> = (0..N)
            .filter(|&index| !self.symbols[index].is_wildcard())
            .collect();
        if !specific.is_empty() {
            let chosen = specific[rng.gen_range(specific.len())];
            self.generalize(chosen);
        }
    }

    pub fn get(&self, index: usize) -> Symbol {
        self.symbols[index]
    }

    pub fn set(&mut self, index: usize, symbol: Symbol) {
        self.symbols[index] = symbol;
    }
}
