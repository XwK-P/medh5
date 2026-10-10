//! Validate files at a level and print one JSON report per line.
//!
//! `cargo run --example validate_files -- LEVEL FILE...`

use std::path::Path;

use medh5::json::{dumps, Style};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let level = &args[1];
    for path in &args[2..] {
        match medh5::validate::validate_file(Path::new(path), level, None) {
            Ok(report) => println!("{}", dumps(&report.to_json(), Style::PYTHON.sorted())),
            Err(e) => println!("{{\"path\": {:?}, \"crash\": {:?}}}", path, e.to_string()),
        }
    }
}
