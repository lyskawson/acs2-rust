use std::collections::{BTreeMap, BTreeSet};

use acs2_measure::agent_names::{canonical_row, canonical_run_name, ARCHIVED_RUN_NAMES};
use acs2_measure::runner::AgentKind;
use serde_json::json;

#[test]
fn archived_agent_names_are_complete_and_one_to_one() {
    let expected = BTreeMap::from([
        ("acs2", "acs2"),
        ("acs2er", "acs2er"),
        ("acs2er_u1", "acs2er_one_update"),
        ("er_episodic", "acs2er_episodic"),
        ("her_full", "acs2her"),
        ("her_values", "acs2her_values"),
        ("her_values_shortened", "acs2her_values_shortened"),
    ]);
    assert_eq!(*ARCHIVED_RUN_NAMES, expected);
    let names: BTreeSet<_> = ARCHIVED_RUN_NAMES.values().collect();
    assert_eq!(names.len(), ARCHIVED_RUN_NAMES.len());
    for (old, new) in expected {
        assert_eq!(canonical_run_name(old), new);
        assert_eq!(canonical_run_name(new), new);
        assert_eq!(AgentKind::from_run_name(new, 1.0 / 3.0).unwrap().name(), new);
    }
}

#[test]
fn cli_rejects_archived_names_with_their_replacements() {
    for (old, new) in ARCHIVED_RUN_NAMES.iter().filter(|(old, new)| old != new) {
        let output = std::process::Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
            .args(["--agents", old])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8(output.stderr).unwrap();
        assert!(error.contains(&format!("agent {old} was renamed to {new}; use {new}")), "{error}");
    }
}

#[test]
fn archived_row_normalization_changes_only_the_agent_name() {
    let original = json!({"agent": "her_values", "arm": "her_values_p2of3", "value": 0.1});
    let mapped = canonical_row(&original);
    assert_eq!(mapped, json!({"agent": "acs2her_values", "arm": "her_values_p2of3", "value": 0.1}));
    assert_eq!(original["agent"], "her_values");
}
