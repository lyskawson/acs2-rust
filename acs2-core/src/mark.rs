use std::collections::BTreeSet;

use crate::condition::Condition;
use crate::perception::Perception;
use crate::rng::RandomSource;
use crate::symbol::Symbol;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Mark<const N: usize> {
    pub attributes: [BTreeSet<Symbol>; N],
}

impl<const N: usize> Mark<N> {
    pub fn new() -> Self {
        Self {
            attributes: core::array::from_fn(|_| BTreeSet::new()),
        }
    }

    pub fn is_marked(&self) -> bool {
        self.attributes.iter().any(|attribute| !attribute.is_empty())
    }

    pub fn set_using_condition(
        &mut self,
        condition: &Condition<N>,
        perception: &Perception<N>,
    ) -> bool {
        if self.is_marked() {
            return self.complement(perception);
        }
        let mut changed = false;
        for index in 0..N {
            if condition.get(index).is_wildcard() {
                self.attributes[index].insert(perception.get(index));
                changed = true;
            }
        }
        changed
    }

    pub fn complement(&mut self, perception: &Perception<N>) -> bool {
        let mut changed = false;
        for index in 0..N {
            if !self.attributes[index].is_empty() {
                self.attributes[index].insert(perception.get(index));
                changed = true;
            }
        }
        changed
    }

    pub fn get_differences(&self, p0: &Perception<N>, rng: &mut dyn RandomSource) -> Condition<N> {
        let mut diff = Condition::all_wildcard();

        let mut nr1 = 0usize;
        let mut nr2 = 0usize;
        for index in 0..N {
            let attribute = &self.attributes[index];
            if !attribute.is_empty() && !attribute.contains(&p0.symbols[index]) {
                nr1 += 1;
            } else if attribute.len() > 1 {
                nr2 += 1;
            }
        }

        if nr1 > 0 {
            let candidates: Vec<usize> = (0..N)
                .filter(|&index| {
                    let attribute = &self.attributes[index];
                    !attribute.is_empty() && !attribute.contains(&p0.symbols[index])
                })
                .collect();
            let chosen = candidates[rng.gen_range(candidates.len())];
            diff.set(chosen, p0.symbols[chosen]);
        } else if nr2 > 0 {
            for index in 0..N {
                if self.attributes[index].len() > 1 {
                    diff.set(index, p0.symbols[index]);
                }
            }
        }

        diff
    }
}
