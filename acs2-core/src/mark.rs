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
        todo!()
    }

    pub fn is_marked(&self) -> bool {
        todo!()
    }

    pub fn set_using_condition(
        &mut self,
        condition: &Condition<N>,
        perception: &Perception<N>,
    ) -> bool {
        todo!()
    }

    pub fn complement(&mut self, perception: &Perception<N>) -> bool {
        todo!()
    }

    pub fn get_differences(
        &self,
        p0: &Perception<N>,
        rng: &mut dyn RandomSource,
    ) -> Condition<N> {
        todo!()
    }
}
