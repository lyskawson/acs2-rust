use crate::classifier::Classifier;
use crate::perception::Perception;

pub type ClassifierRef = usize;

pub struct Population<const N: usize> {
    classifiers: Vec<Classifier<N>>,
}

impl<const N: usize> Population<N> {
    pub fn new() -> Self {
        todo!()
    }

    pub fn from_classifiers(classifiers: Vec<Classifier<N>>) -> Self {
        todo!()
    }

    pub fn len(&self) -> usize {
        todo!()
    }

    pub fn is_empty(&self) -> bool {
        todo!()
    }

    pub fn numerosity(&self) -> u32 {
        todo!()
    }

    pub fn reliable_count(&self, theta_r: f64) -> usize {
        todo!()
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &Classifier<N>> + '_> {
        todo!()
    }

    pub fn get(&self, reference: ClassifierRef) -> &Classifier<N> {
        todo!()
    }

    pub fn get_mut(&mut self, reference: ClassifierRef) -> &mut Classifier<N> {
        todo!()
    }

    pub fn view(&self, references: &[ClassifierRef]) -> Vec<&Classifier<N>> {
        todo!()
    }

    pub fn form_match_set(&self, state: &Perception<N>) -> Vec<ClassifierRef> {
        todo!()
    }

    pub fn form_action_set(
        &self,
        match_set: &[ClassifierRef],
        action: usize,
    ) -> Vec<ClassifierRef> {
        todo!()
    }

    pub fn get_maximum_fitness(&self, match_set: &[ClassifierRef]) -> f64 {
        todo!()
    }

    pub fn insert(&mut self, classifier: Classifier<N>) -> ClassifierRef {
        todo!()
    }

    pub fn remove(&mut self, reference: ClassifierRef) {
        todo!()
    }
}
