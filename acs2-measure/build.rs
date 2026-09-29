use std::path::{Path, PathBuf};
use std::process::Command;

fn git(manifest: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest)
        .args(args)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|text| text.trim().to_owned())
}

pub fn source_provenance(manifest: &Path) -> (String, &'static str) {
    if !manifest.ancestors().any(|path| path.join(".git").exists()) {
        return ("not_git_checkout".to_owned(), "not_git_checkout");
    }
    let commit = git(manifest, &["rev-parse", "HEAD"])
        .filter(|value| !value.is_empty())
        .expect("read source commit");
    let status = git(
        manifest,
        &["status", "--porcelain=v1", "--untracked-files=normal"],
    )
    .expect("read source status");
    (commit, if status.is_empty() { "clean" } else { "dirty" })
}

pub fn git_watch_paths(manifest: &Path) -> Vec<PathBuf> {
    let mut names = vec![
        "HEAD".to_owned(),
        "index".to_owned(),
        "packed-refs".to_owned(),
    ];
    if let Some(branch) = git(manifest, &["symbolic-ref", "HEAD"]) {
        names.push(branch);
    }
    names
        .iter()
        .filter_map(|name| git(manifest, &["rev-parse", "--git-path", name]))
        .map(|path| {
            let path = PathBuf::from(path);
            if path.is_absolute() {
                path
            } else {
                manifest.join(path)
            }
        })
        .collect()
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let (commit, source_state) = source_provenance(Path::new(&manifest));
    println!("cargo:rustc-env=ACS2_BUILD_COMMIT={commit}");
    println!("cargo:rustc-env=ACS2_BUILD_SOURCE_STATE={source_state}");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=../acs2-core/src");
    println!("cargo:rerun-if-changed=../acs2-envs/src");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=../.git");
    for path in git_watch_paths(Path::new(&manifest)) {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}
