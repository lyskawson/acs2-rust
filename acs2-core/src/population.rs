use crate::classifier::Classifier;
use crate::perception::Perception;

pub type ClassifierRef = usize;

pub struct Population<const N: usize> {
    classifiers: Vec<Classifier<N>>,
}

impl<const N: usize> Population<N> {
    pub fn new() -> Self {
        Self {
            classifiers: Vec::new(),
        }
    }

    pub fn from_classifiers(classifiers: Vec<Classifier<N>>) -> Self {
        Self { classifiers }
    }

    pub fn len(&self) -> usize {
        self.classifiers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.classifiers.is_empty()
    }

    pub fn numerosity(&self) -> u32 {
        self.classifiers.iter().map(|classifier| classifier.num).sum()
    }

    pub fn reliable_count(&self, theta_r: f64) -> usize {
        self.classifiers
            .iter()
            .filter(|classifier| classifier.is_reliable(theta_r))
            .count()
    }

    pub fn iter(&self) -> Box<dyn Iterator<Item = &Classifier<N>> + '_> {
        Box::new(self.classifiers.iter())
    }

    pub fn get(&self, reference: ClassifierRef) -> &Classifier<N> {
        &self.classifiers[reference]
    }

    pub fn get_mut(&mut self, reference: ClassifierRef) -> &mut Classifier<N> {
        &mut self.classifiers[reference]
    }

    pub fn view(&self, references: &[ClassifierRef]) -> Vec<&Classifier<N>> {
        references
            .iter()
            .map(|&reference| &self.classifiers[reference])
            .collect()
    }

    pub fn form_match_set(&self, state: &Perception<N>) -> Vec<ClassifierRef> {
        (0..self.classifiers.len())
            .filter(|&reference| self.classifiers[reference].does_match(state))
            .collect()
    }

    pub fn form_action_set(&self, match_set: &[ClassifierRef], action: usize) -> Vec<ClassifierRef> {
        match_set
            .iter()
            .copied()
            .filter(|&reference| self.classifiers[reference].action == Some(action))
            .collect()
    }

    pub fn get_maximum_fitness(&self, match_set: &[ClassifierRef]) -> f64 {
        match_set
            .iter()
            .map(|&reference| &self.classifiers[reference])
            .filter(|classifier| classifier.does_anticipate_change())
            .map(|classifier| classifier.fitness())
            .fold(0.0, f64::max)
    }

    pub fn insert(&mut self, classifier: Classifier<N>) -> ClassifierRef {
        self.classifiers.push(classifier);
        self.classifiers.len() - 1
    }

    pub fn remove(&mut self, reference: ClassifierRef) {
        self.classifiers.remove(reference);
    }

    pub fn remove_many(&mut self, removed_sorted: &[ClassifierRef]) {
        for &index in removed_sorted.iter().rev() {
            self.classifiers.remove(index);
        }
    }
}

pub fn remap_after_removal(refs: &mut Vec<ClassifierRef>, removed_sorted: &[ClassifierRef]) {
    refs.retain(|reference| removed_sorted.binary_search(reference).is_err());
    for reference in refs.iter_mut() {
        let shift = removed_sorted.iter().take_while(|&&removed| removed < *reference).count();
        *reference -= shift;
    }
}
