use std::path::{Path, PathBuf};

pub fn output_path(out: &Path) -> PathBuf {
    let absolute = if out.is_absolute() {
        out.to_path_buf()
    } else {
        std::env::current_dir().unwrap().join(out)
    };
    let parent = absolute.parent().expect("output parent");
    std::fs::create_dir_all(parent).expect("create output directory");
    let parent = parent.canonicalize().expect("output parent exists");
    assert!(
        !parent
            .ancestors()
            .any(|ancestor| ancestor.join(".git").exists()),
        "results belong outside the checkout"
    );
    parent.join(absolute.file_name().expect("output file name"))
}
