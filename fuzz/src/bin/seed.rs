//! Seed `open_validate`'s corpus with the conformance corpus's small files.
//!
//!     cargo run --release --bin seed -- corpus/open_validate
//!
//! Each case the engine builds, of 256 KiB or less: the larger ones slow
//! every run more than they add.

use std::path::PathBuf;

const LARGEST: u64 = 256 * 1024;

fn main() {
    let Some(out) = std::env::args_os().nth(1).map(PathBuf::from) else {
        eprintln!("usage: seed DIR");
        std::process::exit(2);
    };
    let built = std::env::temp_dir().join(format!("medh5-fuzz-seed-{}", std::process::id()));
    medh5::conformance::build_corpus(&built, None).expect("the corpus builds");
    std::fs::create_dir_all(&out).expect("the corpus directory is writable");
    let mut kept = 0;
    for entry in std::fs::read_dir(&built).expect("the built corpus is readable") {
        let path = entry.expect("a directory entry").path();
        let hdf5 = path.extension().is_some_and(|e| e == "medh5" || e == "medh5c");
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(u64::MAX);
        if hdf5 && size <= LARGEST {
            std::fs::copy(&path, out.join(path.file_name().expect("a file name"))).expect("the seed is written");
            kept += 1;
        }
    }
    let _ = std::fs::remove_dir_all(&built);
    println!("{kept} seeds in {}", out.display());
}
