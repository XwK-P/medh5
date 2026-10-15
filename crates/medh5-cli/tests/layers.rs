//! Only the engine replaces a file (D23, 1.0 §14.4).  `std::fs::write` wrote a
//! report in place, through a symbolic link and not atomically, where the
//! Python command line replaced it: every file the command line writes goes
//! through `medh5::h5::file::write_atomic`, and `std::fs::write` appears in
//! its sources only in test code.

use std::path::Path;

/// The sites still writing in place, by file: none since 2.1.
const ALLOWED: [&str; 0] = [];

/// `source` without its `#[cfg(test)]` code: what ships.
fn shipped(source: &str) -> &str {
    source.split("#[cfg(test)]").next().unwrap_or(source)
}

#[test]
fn s14_4_only_the_engine_replaces_files() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut found: Vec<String> = std::fs::read_dir(&src)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "rs"))
        .filter(|path| shipped(&std::fs::read_to_string(path).unwrap()).contains("fs::write("))
        .map(|path| path.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    found.sort();
    let outside: Vec<&String> = found.iter().filter(|f| !ALLOWED.contains(&f.as_str())).collect();
    assert!(outside.is_empty(), "write through medh5::h5::file::write_atomic: {outside:?}");
    let moved: Vec<&str> = ALLOWED.iter().copied().filter(|a| !found.iter().any(|f| f == a)).collect();
    assert!(moved.is_empty(), "moved; remove from ALLOWED: {moved:?}");
}

/// The guard finds a write in shipped code, as 2.0.2's `migrate --report`
/// made one, and none in test code.
#[test]
fn s14_4_the_guard_sees_shipped_code_only() {
    assert!(shipped("fn f() { std::fs::write(p, b)?; }").contains("fs::write("));
    assert!(!shipped("#[cfg(test)]\nmod tests { fn t() { std::fs::write(p, b).unwrap(); } }").contains("fs::write("));
}
