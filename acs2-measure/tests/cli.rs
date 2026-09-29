use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

#[allow(dead_code)]
#[path = "../build.rs"]
mod build_script;

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
    assert_eq!(row["schema"], 2);
    assert!(row["starts"].is_null());
    assert!(row["host"].as_str().is_some_and(|value| !value.is_empty()));
    assert!(row["cpu_model"]
        .as_str()
        .is_some_and(|value| !value.is_empty()));
    assert!(matches!(
        row["source_state"].as_str(),
        Some("clean" | "dirty")
    ));
    assert!(row["wall_seconds_train"].is_number());
    assert!(row["wall_seconds_eval"].is_number());
    assert!((row["random_floor"].as_f64().unwrap() - 334385.0 / 3407872.0).abs() < 1e-12);
}

#[test]
fn build_provenance_is_independent_of_the_run_directory() {
    let directory = std::env::temp_dir().join(format!("acs2-foreign-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let out = directory.join("result.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .current_dir(&directory)
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
            "1",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let row: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&out).unwrap().trim()).unwrap();
    let expected = Command::new("git")
        .arg("-C")
        .arg(env!("CARGO_MANIFEST_DIR"))
        .args(["rev-parse", "HEAD"])
        .output()
        .unwrap();
    assert_eq!(
        row["commit"].as_str().unwrap(),
        String::from_utf8(expected.stdout).unwrap().trim()
    );
    assert_ne!(row["commit"], "");
    fs::remove_file(out).unwrap();
    fs::remove_dir(directory).unwrap();
}

#[test]
fn output_guard_works_when_the_build_source_path_does_not_exist() {
    let directory = std::env::temp_dir().join(format!("acs2-output-probe-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let source = directory.join("probe.rs");
    let binary = directory.join("probe");
    let out = directory.join("out.jsonl");
    let module = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/output.rs");
    fs::write(
        &source,
        format!(
            "#[path = {:?}] mod output;\nfn main() {{ output::output_path(std::path::Path::new({:?})); }}\n",
            module.display().to_string(),
            out.display().to_string()
        ),
    )
    .unwrap();
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".to_owned());
    let built = Command::new(rustc)
        .args(["--edition=2021", "-o"])
        .arg(&binary)
        .arg(&source)
        .env("CARGO_MANIFEST_DIR", directory.join("absent-source"))
        .output()
        .unwrap();
    assert!(
        built.status.success(),
        "{}",
        String::from_utf8_lossy(&built.stderr)
    );
    let ran = Command::new(binary).output().unwrap();
    assert!(
        ran.status.success(),
        "{}",
        String::from_utf8_lossy(&ran.stderr)
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn per_start_records_are_opt_in() {
    let out = std::env::temp_dir().join(format!("acs2-starts-{}.jsonl", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
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
            "1",
            "--per-start",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let row: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&out).unwrap().trim()).unwrap();
    assert_eq!(row["starts"].as_array().unwrap().len(), 104);
    fs::remove_file(out).unwrap();
}

#[test]
fn handeye_pool_cli_maps_row_col_to_x_col_y_row() {
    let out = std::env::temp_dir().join(format!("acs2-handeye-axes-{}.jsonl", std::process::id()));
    let status = Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
        .args([
            "--task",
            "handeye3",
            "--cap",
            "1",
            "--pool",
            "1:2",
            "--encoding",
            "coordinates",
            "--agents",
            "acs2",
            "--seeds",
            "42",
            "--targets",
            "1",
            "--out",
        ])
        .arg(&out)
        .output()
        .unwrap();
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let row: serde_json::Value =
        serde_json::from_str(fs::read_to_string(&out).unwrap().trim()).unwrap();
    assert_eq!(row["goal_pool"], "[Goal { symbols: [Token(2), Token(1)] }]");
    fs::remove_file(out).unwrap();
}

#[test]
fn build_provenance_marks_dirty_and_non_git_sources() {
    let directory = std::env::temp_dir().join(format!("acs2-provenance-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    assert_eq!(
        build_script::source_provenance(&directory),
        ("not_git_checkout".to_owned(), "not_git_checkout")
    );
    let status = Command::new("git")
        .current_dir(&directory)
        .args(["init", "-q"])
        .status()
        .unwrap();
    assert!(status.success());
    fs::write(directory.join("source.rs"), "fn main() {}\n").unwrap();
    for args in [
        vec!["add", "source.rs"],
        vec![
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Initial",
        ],
    ] {
        assert!(Command::new("git")
            .current_dir(&directory)
            .args(args)
            .status()
            .unwrap()
            .success());
    }
    let (commit, state) = build_script::source_provenance(&directory);
    assert_eq!(commit.len(), 40);
    assert_eq!(state, "clean");
    fs::write(
        directory.join("source.rs"),
        "fn main() { println!(\"changed\"); }\n",
    )
    .unwrap();
    assert_eq!(
        build_script::source_provenance(&directory),
        (commit, "dirty")
    );
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn build_provenance_watches_the_real_git_head() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let paths = build_script::git_watch_paths(&manifest);
    let branch = Command::new("git")
        .arg("-C")
        .arg(&manifest)
        .args(["symbolic-ref", "HEAD"])
        .output()
        .unwrap();
    let mut names = vec![
        "HEAD".to_owned(),
        "index".to_owned(),
        "packed-refs".to_owned(),
    ];
    if branch.status.success() {
        names.push(String::from_utf8(branch.stdout).unwrap().trim().to_owned());
    }
    let expected: Vec<_> = names
        .iter()
        .map(|name| {
            let output = Command::new("git")
                .arg("-C")
                .arg(&manifest)
                .args(["rev-parse", "--git-path", name])
                .output()
                .unwrap();
            assert!(output.status.success());
            let path = PathBuf::from(String::from_utf8(output.stdout).unwrap().trim());
            if path.is_absolute() {
                path
            } else {
                manifest.join(path)
            }
        })
        .collect();
    assert_eq!(paths, expected);
    assert!(paths[0].is_file());
}

#[test]
fn killed_cli_keeps_each_completed_evaluation_point() {
    let out = std::env::temp_dir().join(format!("acs2-killed-{}.jsonl", std::process::id()));
    let mut child = Command::new(env!("CARGO_BIN_EXE_acs2-measure"))
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
            "1,2,1000000000",
            "--out",
        ])
        .arg(&out)
        .spawn()
        .unwrap();
    let started = Instant::now();
    let completed_prefix = loop {
        if let Ok(contents) = fs::read_to_string(&out) {
            if contents.ends_with('\n') && contents.lines().count() >= 2 {
                break Some(contents);
            }
        }
        if started.elapsed() > Duration::from_secs(3) || child.try_wait().unwrap().is_some() {
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let _ = child.kill();
    child.wait().unwrap();
    let contents = fs::read_to_string(&out).unwrap_or_default();
    let _ = fs::remove_file(out);
    assert!(
        completed_prefix.is_some(),
        "completed rows were not flushed before interruption"
    );
    let rows: Vec<serde_json::Value> = contents
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["nominal_step"], 1);
    assert_eq!(rows[1]["nominal_step"], 2);
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
