use crate::representation::classifier::Classifier;

pub trait CompactableRule {
    fn condition_includes(&self, other: &Self) -> bool;
    fn same_action(&self, other: &Self) -> bool;
    fn effect_includes(&self, other: &Self) -> bool;
    fn quality(&self) -> f64;
    fn experience(&self) -> u32;
    fn mark_permits(&self, other: &Self) -> bool;
    fn action_sequence_len(&self) -> usize;
    fn anticipates_change(&self) -> bool;
}

impl<const N: usize> CompactableRule for Classifier<N> {
    fn condition_includes(&self, other: &Self) -> bool {
        self.condition.includes(&other.condition)
    }

    fn same_action(&self, other: &Self) -> bool {
        self.action == other.action
    }

    fn effect_includes(&self, other: &Self) -> bool {
        self.effect.subsumes(&other.effect)
    }

    fn quality(&self) -> f64 {
        self.q
    }

    fn experience(&self) -> u32 {
        self.exp
    }

    fn mark_permits(&self, other: &Self) -> bool {
        !self.is_marked() || self.mark == other.mark
    }

    fn action_sequence_len(&self) -> usize {
        1
    }

    fn anticipates_change(&self) -> bool {
        self.does_anticipate_change()
    }
}
