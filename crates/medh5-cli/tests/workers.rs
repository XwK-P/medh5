//! `--workers` through the native binary: the worker processes are this
//! binary's own `__worker`, so only a run of the built executable starts them.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const MEDH5: &str = env!("CARGO_BIN_EXE_medh5");

/// Valid and invalid cases, so the batch has both outcomes in it.
const CASES: [&str; 6] = [
    "core-minimal",
    "seg-labelmap",
    "training-index",
    "E001-missing-version",
    "E102-non-orthonormal",
    "E201-no-images",
];

fn corpus(dir: &Path) -> Vec<PathBuf> {
    let mut build = Command::new(MEDH5);
    build.args(["conformance", "build"]).arg(dir);
    for case in CASES {
        build.args(["--case", case]);
    }
    assert!(build.output().unwrap().status.success());
    let mut files: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "medh5"))
        .collect();
    files.sort();
    assert_eq!(files.len(), CASES.len());
    files
}

fn run(words: &[&str], workers: &str, files: &[PathBuf]) -> Output {
    Command::new(MEDH5).args(words).args(["--workers", workers]).args(files).output().unwrap()
}

/// What a run printed and how it exited, with `dir` written as `DIR`.
fn printed(out: &Output, dir: &Path) -> (Option<i32>, String, String) {
    let text = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&*dir.to_string_lossy(), "DIR");
    (out.status.code(), text(&out.stdout), text(&out.stderr))
}

#[test]
fn reading_commands_print_what_one_process_prints() {
    let dir = tempfile::tempdir().unwrap();
    let files = corpus(dir.path());
    for words in [&["validate"][..], &["validate", "--json"], &["verify"], &["verify", "--json"]] {
        let one = printed(&run(words, "1", &files), dir.path());
        assert!(!one.1.is_empty(), "{words:?} printed nothing");
        assert_eq!(one, printed(&run(words, "3", &files), dir.path()));
    }
}

#[test]
fn a_writing_command_prints_what_one_process_prints() {
    let built = tempfile::tempdir().unwrap();
    let files = corpus(built.path());
    let runs: Vec<_> = ["1", "3"]
        .into_iter()
        .map(|workers| {
            let dir = tempfile::tempdir().unwrap();
            let mut copies: Vec<PathBuf> = files
                .iter()
                .map(|f| {
                    let copy = dir.path().join(f.file_name().unwrap());
                    fs::copy(f, &copy).unwrap();
                    copy
                })
                .collect();
            // A file named twice is written by one worker, in order.
            let named_twice = copies.iter().find(|c| c.ends_with("seg-labelmap.medh5")).unwrap().clone();
            copies.push(named_twice);
            printed(&run(&["index", "build"], workers, &copies), dir.path())
        })
        .collect();
    assert!(runs[0].1.contains("index current"), "{}", runs[0].1);
    assert_eq!(runs[0], runs[1]);
}
