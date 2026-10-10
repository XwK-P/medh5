//! Print the summary, recomputed `content_id` and verification of samples.
//!
//! `cargo run --example summary -- FILE...` --- one JSON object per line.

use std::path::Path;

use medh5::json::{dumps, Style};
use serde_json::{json, Value};

fn describe(path: &Path) -> Value {
    let sample = match medh5::sample::open_sample(path) {
        Ok(s) => s,
        Err(e) => return json!({"path": path.file_name().unwrap().to_string_lossy(), "error": e.to_string()}),
    };
    let summary = sample.summary().unwrap_or_else(|e| json!({"error": e.to_string()}));
    let content_id = sample.compute_content_id().map(Value::from).unwrap_or_else(|e| json!({"error": e.to_string()}));
    let verify = sample.verify(None).map(|v| v.summary()).unwrap_or_else(|e| json!({"error": e.to_string()}));
    json!({
        "path": path.file_name().unwrap().to_string_lossy(),
        "summary": summary,
        "content_id": content_id,
        "verify": verify,
    })
}

fn main() {
    for arg in std::env::args().skip(1) {
        let value = describe(Path::new(&arg));
        println!("{}", dumps(&value, Style::PYTHON.sorted()));
    }
}
