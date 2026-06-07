use crate::classifier::Classifier;

pub fn is_subsumer<const N: usize>(cl: &Classifier<N>, theta_exp: u32, theta_r: f64) -> bool {
    cl.exp > theta_exp && cl.is_reliable(theta_r) && !cl.is_marked()
}

pub fn does_subsume<const N: usize>(
    cl: &Classifier<N>,
    other: &Classifier<N>,
    theta_exp: u32,
    theta_r: f64,
) -> bool {
    is_subsumer(cl, theta_exp, theta_r)
        && cl.is_more_general(other)
        && cl.condition.subsumes(&other.condition)
        && cl.action == other.action
        && cl.effect.subsumes(&other.effect)
}
