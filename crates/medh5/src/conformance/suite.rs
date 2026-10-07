//! Publishing the corpus, and scoring an implementation that is not this one.
//!
//! [`run_corpus`](super::run_corpus) answers "does *this* validator agree with
//! the spec".  The question a format has to answer is different: **does
//! yours?**  So the corpus ships as a directory anyone can download --- files,
//! expected diagnostics, the code table, the JSON Schema and checksums --- and
//! any implementation, in any language, is scored by handing back a list of
//! what it reported per file:
//!
//! ```text
//! [{"file": "core-minimal.medh5", "errors": ["E101"], "warnings": []}, ...]
//! ```
//!
//! `medh5 validate --json` already emits a superset of it (`path` plus a
//! `diagnostics` list), so the reference implementation scores itself through
//! the same door as everybody else --- which is the only way the door stays
//! honest.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

use super::{build_corpus, cases, selected, Case, CaseResult};
use crate::digest::hash_hex;
use crate::json::{pretty, py_str};
use crate::pyval::py_path;
use crate::{codes, Error, Result, FORMAT_VERSION, VERSION};

/// The file name the `/meta` JSON Schema is published under.
pub const SCHEMA: &str = "medh5-sample-1.0.schema.json";
/// The checksum file covering every other published file.
pub const CHECKSUMS: &str = "SHA256SUMS";

/// Write the distributable suite and return its directory.
///
/// Everything an independent implementation needs is in one place: it should
/// not have to install this package to be measured against the spec.
pub fn publish(outdir: &Path, names: Option<&[String]>) -> Result<PathBuf> {
    build_corpus(outdir, names)?;
    let chosen: Vec<&Case> = selected(names);
    let table = json!({
        "format": FORMAT_VERSION,
        "codes": codes::all()
            .iter()
            .map(|c| json!({"code": c.code, "severity": c.severity, "domain": c.domain, "summary": c.summary}))
            .collect::<Vec<Value>>(),
    });
    fs::write(outdir.join("codes.json"), pretty(&table) + "\n")?;
    fs::write(outdir.join(SCHEMA), crate::document::schema_text())?;
    fs::write(outdir.join("README.md"), readme(&chosen))?;
    write_checksums(outdir)?;
    Ok(outdir.to_path_buf())
}

/// Every file under `root`, as `/`-joined relative paths in path order.
fn published_files(root: &Path) -> Result<Vec<String>> {
    fn walk(dir: &Path, prefix: &[String], out: &mut Vec<Vec<String>>) -> Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let mut parts = prefix.to_vec();
            parts.push(entry.file_name().to_string_lossy().into_owned());
            let kind = entry.file_type()?;
            if kind.is_dir() {
                walk(&entry.path(), &parts, out)?;
            } else if entry.path().is_file() {
                out.push(parts);
            }
        }
        Ok(())
    }
    let mut found = Vec::new();
    walk(root, &[], &mut found)?;
    // `sorted(root.rglob("*"))` orders by path components.
    found.sort();
    Ok(found.into_iter().map(|parts| parts.join("/")).collect())
}

/// Checksum every published file except the checksum file itself.
fn write_checksums(root: &Path) -> Result<PathBuf> {
    let mut lines = Vec::new();
    for name in published_files(root)? {
        if name.rsplit('/').next() == Some(CHECKSUMS) {
            continue;
        }
        let digest = hash_hex("sha256", &fs::read(root.join(&name))?)?;
        lines.push(format!("{digest}  {name}"));
    }
    let target = root.join(CHECKSUMS);
    fs::write(&target, lines.join("\n") + "\n")?;
    Ok(target)
}

/// Names of published files whose bytes no longer match `SHA256SUMS`.
pub fn check_checksums(root: &Path) -> Result<Vec<String>> {
    let listing = root.join(CHECKSUMS);
    let text = fs::read_to_string(&listing).map_err(|e| crate::error::os_error(&e, py_path(&listing)))?;
    let mut bad = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        let (digest, name) = line.split_once("  ").unwrap_or((line, ""));
        let path = root.join(name);
        let matches = path.exists()
            && hash_hex("sha256", &fs::read(&path).map_err(|e| crate::error::os_error(&e, py_path(&path)))?)? == digest;
        if !matches {
            bad.push(name.to_string());
        }
    }
    Ok(bad)
}

/// The suite's `expected.json`.
pub fn load_manifest(root: &Path) -> Result<Value> {
    let path = root.join("expected.json");
    if !path.exists() {
        return Err(Error::invalid(format!(
            "{} not found --- build the suite first with `medh5 conformance publish`",
            py_path(&path)
        )));
    }
    let text = fs::read_to_string(&path).map_err(|e| crate::error::os_error(&e, py_path(&path)))?;
    crate::json::loads(&text).map_err(|e| Error::Value(crate::json::decode_error(&text, &e)))
}

/// Errors and warnings from either accepted submission shape.
///
/// A foreign implementation sends `errors`/`warnings` lists.  `medh5 validate
/// --json` sends `diagnostics` with a severity on each.  Both are read here so
/// nobody has to reshape a report to be scored.
fn submitted_codes(entry: &Value) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut errors = BTreeSet::new();
    let mut warnings = BTreeSet::new();
    if let Some(diagnostics) = entry.get("diagnostics") {
        for diagnostic in diagnostics.as_array().into_iter().flatten() {
            let code = diagnostic.get("code").map(py_str).unwrap_or_default();
            if code.is_empty() {
                continue;
            }
            let severity = diagnostic.get("severity").map(py_str).unwrap_or_else(|| "error".into());
            if severity == "warning" {
                warnings.insert(code);
            } else {
                errors.insert(code);
            }
        }
        return (errors, warnings);
    }
    let list = |key: &str| -> BTreeSet<String> {
        entry.get(key).and_then(Value::as_array).map(|a| a.iter().map(py_str).collect()).unwrap_or_default()
    };
    (list("errors"), list("warnings"))
}

/// The case a submitted result is about, keyed by file name.
fn submitted_name(entry: &Value) -> String {
    let raw = ["file", "path", "name"]
        .iter()
        .filter_map(|k| entry.get(*k))
        .find(|v| crate::json::py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    Path::new(&raw).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn strings_of(record: &Value, key: &str) -> Vec<String> {
    record.get(key).and_then(Value::as_array).map(|a| a.iter().map(py_str).collect()).unwrap_or_default()
}

/// Score a foreign validator's results against the published expectations.
///
/// A case with no submitted result is a failure, not a skip: silence about a
/// file you were given is the same as failing to diagnose it.
pub fn score(root: &Path, submitted: &[Value]) -> Result<Vec<CaseResult>> {
    let manifest = load_manifest(root)?;
    let by_file: HashMap<String, &Value> = submitted.iter().map(|e| (submitted_name(e), e)).collect();
    let mut results = Vec::new();
    for record in manifest.get("cases").and_then(Value::as_array).into_iter().flatten() {
        let name = record.get("name").map(py_str).unwrap_or_default();
        let case = cases().iter().find(|c| c.name == name).cloned().unwrap_or_else(|| Case::from_record(record));
        let file = record.get("file").map(py_str).unwrap_or_default();
        let mut result = CaseResult::new(case, py_path(&root.join(&file)));
        let Some(entry) = by_file.get(&file) else {
            result.error = Some("no result submitted for this case".into());
            results.push(result);
            continue;
        };
        let (errors, warnings) = submitted_codes(entry);
        let expected: BTreeSet<String> =
            strings_of(record, "expect_errors").into_iter().chain(strings_of(record, "expect_warnings")).collect();
        let got: BTreeSet<String> = errors.union(&warnings).cloned().collect();
        result.got_errors = errors.into_iter().collect();
        result.got_warnings = warnings.into_iter().collect();
        result.missing = expected.difference(&got).cloned().collect();
        result.unexpected = got.difference(&expected).cloned().collect();
        results.push(result);
    }
    Ok(results)
}

/// The `--json` summary of a run or a score.
pub fn summarize(results: &[CaseResult]) -> Value {
    let failures: Vec<&CaseResult> = results.iter().filter(|r| !r.ok()).collect();
    json!({
        "cases": results.len(),
        "passed": results.len() - failures.len(),
        "failed": failures.len(),
        "ok": failures.is_empty(),
        "failures": failures.iter().map(|r| r.to_json()).collect::<Vec<Value>>(),
    })
}

fn readme(cases: &[&Case]) -> String {
    let valid = cases.iter().filter(|c| c.valid()).count();
    let shards = cases.iter().filter(|c| c.suffix == ".medh5c").count();
    let total = cases.len();
    let kinds = format!("the cases: {} samples and {shards} collections", total - shards);
    let covered: BTreeSet<&str> =
        cases.iter().flat_map(|c| c.errors.iter().chain(&c.warnings)).map(String::as_str).collect();
    let invalid = total - valid;
    let n_covered = covered.len();
    format!(
        r#"# MEDH5 {FORMAT_VERSION} conformance suite

Generated by medh5 {VERSION}. {total} cases: {valid} valid files a
conforming implementation must accept, {invalid} invalid ones it must
reject with specific diagnostic codes. {n_covered} of the codes in the
specification's §15.2 table appear here.

## What is in this directory

| File | What it is |
|---|---|
| `*.medh5`, `*.medh5c` | {kinds} |
| `expected.json` | per case, the exact codes a conforming validator must emit |
| `codes.json` | the §15.2 diagnostic code table as data |
| `{SCHEMA}` | the JSON Schema for the `/meta` document |
| `{CHECKSUMS}` | sha256 of every file above |

## Running it

Validate every case **at the level its manifest entry declares**, and write one
JSON array:

```json
[
  {{"file": "core-minimal.medh5", "errors": [], "warnings": []}},
  {{"file": "E102-not-orthonormal.medh5", "errors": ["E102"], "warnings": []}}
]
```

Then score it:

```
medh5 conformance score . results.json
```

`medh5 validate --json` emits a superset of that shape (a `diagnostics` list
with a severity on each), so the reference implementation is scored through
exactly the same door as everybody else:

```python
import json, subprocess
manifest = json.load(open("expected.json"))
results = []
for case in manifest["cases"]:
    out = subprocess.run(
        ["medh5", "validate", case["file"], "--level", case["level"], "--json"],
        capture_output=True, text=True,
    ).stdout
    report = json.loads(out)[0]
    results.append({{"file": case["file"], "diagnostics": report["diagnostics"]}})
json.dump(results, open("results.json", "w"))
```

## How it is scored

For each case, the set of codes you report must **equal** the expected set: a
missing code is a defect you failed to catch, an extra code is a valid file you
rejected. Both fail. A case you report nothing about fails too --- silence about
a file you were handed is not a pass.

Diagnostic *messages* are yours to write; only the codes are normative.

## Three things worth knowing before you start

**Validate at the declared `level`, not deeper.** `structural` < `semantic` <
`integrity`. Shallower misses the defect the case exists to test. *Deeper is
not safe either*: most invalid cases were made by editing a valid file, so
their stored digests cover the pre-edit bytes and an integrity pass adds a
`content_id` mismatch the case never claimed. Those cases are marked
`"mutated": true`, and the mismatch is an artifact of how they are built, not
something to diagnose.

**A `.medh5c` case is a collection** (spec §2.1) --- it contains samples rather
than being one. `"file_suffix"` in the manifest tells you which.

**Verify the bytes first.** `{CHECKSUMS}` covers every published file; `medh5
conformance score` warns when a case has drifted, because a score over files
that are not the published files is not a score.

The specification is `docs/spec/medh5-{FORMAT_VERSION}.md` in the medh5
repository, and every case names the clause it tests in `expected.json`.
"#
    )
}
