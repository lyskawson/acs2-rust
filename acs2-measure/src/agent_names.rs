use std::collections::BTreeMap;
use std::sync::LazyLock;

use serde_json::Value;

pub static ARCHIVED_RUN_NAMES: LazyLock<BTreeMap<&'static str, &'static str>> = LazyLock::new(|| {
    serde_json::from_str(include_str!("../../experiments/grid/agent_names.json"))
        .expect("archived agent run names")
});

pub fn canonical_run_name(name: &str) -> &str {
    ARCHIVED_RUN_NAMES.get(name).copied().unwrap_or(name)
}

pub fn canonical_row(row: &Value) -> Value {
    let mut result = row.clone();
    if let Some(name) = row["agent"].as_str() {
        result["agent"] = Value::String(canonical_run_name(name).to_owned());
    }
    result
}
