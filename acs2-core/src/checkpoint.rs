use crate::acs2er::ReplaySample;
use crate::classifier::Classifier;
use crate::rng::RngState;

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
