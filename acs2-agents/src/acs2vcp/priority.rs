use acs2_core::learning::replay::ReplaySample;
use acs2_core::learning::rl::{BootstrapEstimator, MaxFitnessBootstrap};
use acs2_core::representation::population::Population;

pub const EPSILON: f64 = 1e-6;
pub const MAX_TEMPERATURE: u32 = 24;

pub fn integer_power(base: f64, exponent: u32) -> f64 {
    let mut result = 1.0;
    for _ in 0..exponent {
        result *= base;
    }
    result
}

pub fn population_variance(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / values.len() as f64
}

pub fn action_value<const M: usize>(population: &Population<M>, sample: &ReplaySample<M>) -> f64 {
    let matched = population.form_match_set(&sample.state);
    let action_set = population.form_action_set(&matched, sample.action);
    MaxFitnessBootstrap.estimate(population, &action_set)
}

pub fn acceptance(variance: f64, maximum: f64, temperature: u32) -> f64 {
    assert!(maximum.is_finite() && maximum >= 0.0);
    assert!(variance.is_finite() && variance >= 0.0 && variance <= maximum);
    assert!(temperature <= MAX_TEMPERATURE);
    if temperature == 0 || variance == 0.0 {
        return 1.0;
    }
    let priority = integer_power(maximum - variance + EPSILON, temperature);
    let envelope = integer_power(maximum + EPSILON, temperature);
    let probability = priority / envelope;
    assert!(
        probability.is_finite() && probability > 0.0 && probability <= 1.0,
        "VCP cannot draw: priority acceptance is not representable"
    );
    probability
}

#[derive(Clone, Debug, PartialEq)]
pub struct DistributionSummary {
    pub count: u64,
    pub sum: f64,
    pub sum_squares: f64,
    pub minimum: f64,
    pub maximum: f64,
    pub histogram: [u64; 16],
}

impl Default for DistributionSummary {
    fn default() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            minimum: f64::INFINITY,
            maximum: 0.0,
            histogram: [0; 16],
        }
    }
}

impl DistributionSummary {
    pub fn record(&mut self, value: f64, maximum: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        let bin = ((value / maximum * 16.0) as usize).min(15);
        self.histogram[bin] += 1;
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PriorityCounts {
    pub proposals: u64,
    pub accepted: u64,
    pub value_queries: u64,
    pub acceptance_coins: u64,
    pub acceptance_sum: f64,
    pub variance: DistributionSummary,
    pub accepted_variance: DistributionSummary,
    pub candidate_acceptance_sums: [f64; MAX_TEMPERATURE as usize],
    pub batch_peak_bytes: usize,
}
