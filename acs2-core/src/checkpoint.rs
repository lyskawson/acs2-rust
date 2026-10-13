pub mod codec;

use crate::learning::replay::ReplaySample;
use crate::representation::classifier::Classifier;
use crate::support::rng::RngState;

#[derive(Clone, Debug)]
pub struct AgentState<const N: usize> {
    pub population: Vec<Classifier<N>>,
    pub rng: RngState,
    pub replay: Option<Vec<ReplaySample<N>>>,
}

pub trait Checkpointed<const N: usize> {
    fn capture(&self) -> AgentState<N>;
    fn restore(&mut self, state: AgentState<N>);
}
