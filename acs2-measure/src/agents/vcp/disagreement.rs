use acs2_agents::acs2vcp::priority::DistributionSummary;
use serde_json::{json, Value};

pub const POSITIVE_UPPER_BOUNDS: [f64; 25] = [
    1e-12, 1e-9, 1e-6, 1e-3, 0.01, 0.1, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0,
    1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0, 65536.0, 131072.0, 250000.0,
];

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LogSummary {
    pub exact_zeros: u64,
    pub positive_bins: [u64; POSITIVE_UPPER_BOUNDS.len()],
}

impl LogSummary {
    pub fn record(&mut self, value: f64) {
        assert!(value.is_finite() && (0.0..=250000.0).contains(&value));
        if value == 0.0 {
            self.exact_zeros += 1;
        } else {
            let index = POSITIVE_UPPER_BOUNDS.partition_point(|bound| *bound < value);
            self.positive_bins[index] += 1;
        }
    }

    pub fn json(&self) -> Value {
        let count = self.exact_zeros + self.positive_bins.iter().sum::<u64>();
        json!({"count": count, "exact_zero_count": self.exact_zeros,
            "exact_zero_share": (count > 0).then(|| self.exact_zeros as f64 / count as f64),
            "positive_bin_counts": self.positive_bins,
            "positive_bin_upper_bounds": POSITIVE_UPPER_BOUNDS,
            "bin_rule": "exact_zero_separate_positive_lower_exclusive_upper_inclusive"})
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TrainingDisagreement {
    pub proposals: LogSummary,
    pub accepted: LogSummary,
    pub avoided_value_queries: u64,
    pub avoided_classifier_tests: u64,
}

impl TrainingDisagreement {
    pub fn record(&mut self, variance: f64, accepted: bool) {
        self.proposals.record(variance);
        if accepted {
            self.accepted.record(variance);
        }
    }
}

pub fn evaluation_summary(mut values: Vec<f64>, maximum: f64, heads: usize) -> Value {
    let mut moments = DistributionSummary::default();
    let mut logarithmic = LogSummary::default();
    for &value in &values {
        moments.record(value, maximum);
        logarithmic.record(value);
    }
    values.sort_by(f64::total_cmp);
    let quantiles: serde_json::Map<_, _> = [
        ("p25", 25),
        ("p50", 50),
        ("p75", 75),
        ("p90", 90),
        ("p99", 99),
    ]
    .into_iter()
    .map(|(name, percentile)| {
        let rank = (values.len() * percentile).div_ceil(100).max(1);
        (name.to_owned(), json!(values.get(rank - 1)))
    })
    .collect();
    let mut result = super::summary(&moments);
    result["logarithmic"] = logarithmic.json();
    result["exact_zero_share"] = logarithmic.json()["exact_zero_share"].clone();
    result["quantiles"] = Value::Object(quantiles);
    result["quantile_rule"] = json!("nearest_rank_ceil_n_times_percentile");
    result["variance_maximum"] = json!(maximum);
    result["value_queries"] = json!(values.len() * heads);
    result["cost_scope"] = json!("evaluation_only_training_counters_paused");
    result["reference_set"] = json!("non_identity_knowledge_transitions_from_evaluated_block_start_goal_pairs_equal_transition_weights");
    result["temporary_variance_payload_bytes"] = json!(values.len() * std::mem::size_of::<f64>());
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logarithmic_summary_separates_exact_zero_and_resolves_small_variances() {
        let mut summary = LogSummary::default();
        for value in [
            0.0, 1e-13, 1e-12, 0.5, 55.0, 520.0, 5000.0, 24000.0, 250000.0,
        ] {
            summary.record(value);
        }
        let json = summary.json();
        assert_eq!(json["count"], 9);
        assert_eq!(json["exact_zero_count"], 1);
        assert_eq!(json["exact_zero_share"], 1.0 / 9.0);
        assert_eq!(summary.positive_bins[0], 2);
        assert_eq!(summary.positive_bins[6], 1);
        assert_eq!(summary.positive_bins[12], 1);
        assert_eq!(summary.positive_bins[16], 1);
        assert_eq!(summary.positive_bins[19], 1);
        assert_eq!(summary.positive_bins[21], 1);
        assert_eq!(summary.positive_bins[24], 1);
    }

    #[test]
    fn evaluation_quantiles_use_all_transitions_and_keep_the_population_variance_scale() {
        let result = evaluation_summary(vec![100.0, 0.0, 1.0, 4.0, 0.0], 250000.0, 4);
        assert_eq!(result["mean"], 21.0);
        assert_eq!(result["exact_zero_share"], 0.4);
        assert_eq!(
            result["quantiles"],
            json!({"p25": 0.0, "p50": 1.0, "p75": 4.0, "p90": 100.0, "p99": 100.0})
        );
        assert_eq!(result["maximum"], 100.0);
        assert_eq!(result["value_queries"], 20);
        assert_eq!(result["histogram"][0], 5);
    }
}
