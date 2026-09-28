use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn command(out: &PathBuf) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .args([
            "--task",
            "maze4",
            "--cap",
            "5",
            "--pool",
            "2:5,5:5,6:3,6:4",
            "--encoding",
            "coordinates",
            "--agents",
            "acs2",
            "--seeds",
            "42",
            "--targets",
            "20",
            "--out",
        ])
        .arg(out)
        .output()
        .expect("runner process")
}

#[test]
fn one_row_contains_the_configuration_agent_seed_point_and_references() {
    let out = std::env::temp_dir().join(format!("acs2-measure-{}.jsonl", std::process::id()));
    let status = command(&out);
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let contents = fs::read_to_string(&out).unwrap();
    fs::remove_file(&out).unwrap();
    let lines: Vec<_> = contents.lines().collect();
    assert_eq!(lines.len(), 1);
    let row: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(row["task"], "maze4");
    assert_eq!(row["agent"], "acs2");
    assert_eq!(row["seed"], 42);
    assert_eq!(row["nominal_step"], 20);
    assert_eq!(row["evaluation_pairs"], 104);
    assert!(row["commit"]
        .as_str()
        .is_some_and(|commit| commit.len() == 40));
    assert!(row["preset"]["beta"].is_number());
    assert!(row["agent_parameters"].is_object());
    assert!((row["random_floor"].as_f64().unwrap() - 334385.0 / 3407872.0).abs() < 1e-12);
}

#[test]
fn output_inside_the_checkout_is_rejected_before_file_creation() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("guard-{}.jsonl", std::process::id()));
    assert!(!out.exists());
    let status = command(&out);
    assert!(!status.status.success());
    assert!(!out.exists());
}
