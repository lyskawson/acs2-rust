use crate::perception::Perception;

pub type Info = ();

pub struct StepOutcome<const N: usize> {
    pub observation: Perception<N>,
    pub reward: f64,
    pub terminated: bool,
    pub truncated: bool,
    pub info: Info,
}

pub trait Environment<const N: usize> {
    fn reset(&mut self) -> Perception<N>;
    fn step(&mut self, action: usize) -> StepOutcome<N>;
}
