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
//! The scorer also takes the shape `medh5 validate --json` reports a file in
//! --- a `diagnostics` list, each entry with its `code` and `severity`, in
//! place of `errors` and `warnings`, and `path` in place of `file` --- so the
//! reference implementation scores itself through the same door as everybody
//! else, which is the only way the door stays honest.

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
/// The clinical profile's schema (format 1.1): `clinical/meta` and the
/// logical-record bundle.
pub const CLINICAL_SCHEMA: &str = crate::clinical::schema::SCHEMA_FILE;
/// The task-and-cache contract's schemas, published beside the format's so a
/// third-party training layer has them in the same place.
pub const COMPANION_SCHEMAS: [&str; 2] = [crate::companion::task::SCHEMA_FILE, crate::companion::cache::SCHEMA_FILE];
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
        "revision": codes::table_revision(),
        "codes": codes::all()
            .iter()
            .map(|c| {
                json!({
                    "code": c.code,
                    "severity": c.severity,
                    "domain": c.domain,
                    "summary": c.summary,
                    "since": codes::since(&c.code),
                    "revision": codes::revision(&c.code),
                })
            })
            .collect::<Vec<Value>>(),
    });
    fs::write(outdir.join("codes.json"), pretty(&table) + "\n")?;
    fs::write(outdir.join(SCHEMA), crate::document::schema_text())?;
    fs::write(outdir.join(CLINICAL_SCHEMA), crate::clinical::schema::schema_text())?;
    fs::write(outdir.join(COMPANION_SCHEMAS[0]), crate::companion::task::schema_text())?;
    fs::write(outdir.join(COMPANION_SCHEMAS[1]), crate::companion::cache::schema_text())?;
    // The task-and-cache fixtures read the corpus's clinical samples: they
    // are published whenever those were.
    if super::companion::SOURCES.iter().all(|f| outdir.join(f).exists()) {
        super::companion::write_fixtures(outdir)?;
    }
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
    crate::json::loads(&text).map_err(|e| Error::Value(format!("not valid JSON: {e}")))
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

/// What a score holds an implementation to: every case unless bounded.
///
/// An implementation of format 1.0 has no answer to a 1.1 case, and one
/// written against the first revision of the code table (§15.2) none for a
/// code a later revision allocated: unbounded, both fail cases they were
/// never told about.  A case above either bound is not scored, and a
/// submitted code of a later revision than `codes_revision` is ignored.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScoreOptions {
    /// Score only the cases addressed to this format version or an earlier
    /// one (`"1.0"`, `"1.1"`).
    pub format_version: Option<String>,
    /// Score only the cases whose codes this revision of the code table
    /// holds.
    pub codes_revision: Option<u32>,
}

impl ScoreOptions {
    /// Bound the score by format version.
    #[must_use]
    pub fn format_version(mut self, version: &str) -> ScoreOptions {
        self.format_version = Some(version.to_string());
        self
    }

    /// Bound the score by code-table revision.
    #[must_use]
    pub fn codes_revision(mut self, revision: u32) -> ScoreOptions {
        self.codes_revision = Some(revision);
        self
    }
}

/// The `since` and `revision` of every code, from the suite's own
/// `codes.json` where it has one --- a suite published by a later medh5 holds
/// codes this one does not --- and from this build's table otherwise.
fn published_codes(root: &Path) -> HashMap<String, (String, u32)> {
    let mut table: HashMap<String, (String, u32)> = codes::all()
        .iter()
        .map(|c| {
            let since = codes::since(&c.code).unwrap_or("1.0").to_string();
            (c.code.clone(), (since, codes::revision(&c.code).unwrap_or(1)))
        })
        .collect();
    let published = fs::read_to_string(root.join("codes.json")).ok().and_then(|t| crate::json::loads(&t).ok());
    for entry in published.iter().flat_map(|t| t.get("codes").and_then(Value::as_array)).flatten() {
        let (Some(code), Some(since)) =
            (entry.get("code").and_then(Value::as_str), entry.get("since").and_then(Value::as_str))
        else {
            continue;
        };
        let revision = entry.get("revision").and_then(Value::as_u64).map_or(1, |r| r as u32);
        table.insert(code.to_string(), (since.to_string(), revision));
    }
    table
}

/// Score a foreign validator's results against the published expectations.
///
/// A case with no submitted result is a failure, not a skip: silence about a
/// file you were given is the same as failing to diagnose it.
pub fn score(root: &Path, submitted: &[Value]) -> Result<Vec<CaseResult>> {
    score_with(root, submitted, &ScoreOptions::default())
}

/// [`score`], bounded by `options`: the cases above a bound are left out of
/// the results, and the codes of a later revision out of what was submitted.
pub fn score_with(root: &Path, submitted: &[Value], options: &ScoreOptions) -> Result<Vec<CaseResult>> {
    if let Some(version) = &options.format_version {
        if super::version_key(version).is_none() {
            return Err(Error::Value(format!(
                "format version {} is not a version: give it as major.minor, such as 1.0",
                crate::json::repr_str(version)
            )));
        }
    }
    let manifest = load_manifest(root)?;
    let table = published_codes(root);
    let by_file: HashMap<String, &Value> = submitted.iter().map(|e| (submitted_name(e), e)).collect();
    let mut results = Vec::new();
    for record in manifest.get("cases").and_then(Value::as_array).into_iter().flatten() {
        if !within(record, options, &table) {
            continue;
        }
        let name = record.get("name").map(py_str).unwrap_or_default();
        let case = cases().iter().find(|c| c.name == name).cloned().unwrap_or_else(|| Case::from_record(record));
        let file = record.get("file").map(py_str).unwrap_or_default();
        let mut result = CaseResult::new(case, py_path(&root.join(&file)));
        let Some(entry) = by_file.get(&file) else {
            result.error = Some("no result submitted for this case".into());
            results.push(result);
            continue;
        };
        let (mut errors, mut warnings) = submitted_codes(entry);
        if let Some(bound) = options.codes_revision {
            let later = |code: &String| table.get(code).is_some_and(|(_, revision)| *revision > bound);
            errors.retain(|c| !later(c));
            warnings.retain(|c| !later(c));
        }
        // Against the published expectations, severity by severity.
        let (missing, unexpected) = super::differences(
            &strings_of(record, "expect_errors"),
            &strings_of(record, "expect_warnings"),
            &errors,
            &warnings,
        );
        result.got_errors = errors.into_iter().collect();
        result.got_warnings = warnings.into_iter().collect();
        result.missing = missing;
        result.unexpected = unexpected;
        results.push(result);
    }
    Ok(results)
}

/// Whether a manifest record falls within the bounds of `options`.
fn within(record: &Value, options: &ScoreOptions, table: &HashMap<String, (String, u32)>) -> bool {
    let expected: Vec<String> =
        strings_of(record, "expect_errors").into_iter().chain(strings_of(record, "expect_warnings")).collect();
    if let Some(bound) = &options.format_version {
        // The record's own statement, and for a suite published before
        // records stated one, what its codes and clause say.
        let version = match record.get("format_version").and_then(Value::as_str) {
            Some(stated) => stated.to_string(),
            None => {
                let clause = record.get("clause").map(py_str).unwrap_or_default();
                let floor = if clause.contains("(1.1)") { "1.1" } else { "1.0" };
                super::latest_version(floor, expected.iter().filter_map(|c| table.get(c).map(|(s, _)| s.as_str())))
            }
        };
        if super::version_key(&version) > super::version_key(bound) {
            return false;
        }
    }
    if let Some(bound) = options.codes_revision {
        if expected.iter().any(|c| table.get(c).is_some_and(|(_, revision)| *revision > bound)) {
            return false;
        }
    }
    true
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
    let n_codes = codes::all().len();
    let revision = codes::table_revision();
    let (task_schema, cache_schema) = (COMPANION_SCHEMAS[0], COMPANION_SCHEMAS[1]);
    format!(
        r#"# MEDH5 {FORMAT_VERSION} conformance suite

Generated by medh5 {VERSION}. {total} cases: {valid} valid files a
conforming implementation must accept, {invalid} invalid ones it must
reject with specific diagnostic codes. {n_covered} of the {n_codes} codes in
the specification's tables (1.0 §15.2 and 1.1 §11.2) appear here.

## What is in this directory

| File | What it is |
|---|---|
| `*.medh5`, `*.medh5c` | {kinds} |
| `expected.json` | per case, the exact codes a conforming validator must emit |
| `codes.json` | the diagnostic code table as data (1.0 §15.2 and 1.1 §11.2), at revision {revision}: each code's `since` version and the `revision` that allocated it |
| `{SCHEMA}` | the JSON Schema for the `/meta` document |
| `{CLINICAL_SCHEMA}` | the JSON Schema for `clinical/meta` and clinical records (format 1.1) |
| `{task_schema}`, `{cache_schema}` | the task-and-cache contract's schemas (not part of the corpus) |
| `{CHECKSUMS}` | sha256 of every file above |

## Running it

Validate every case **at the level its manifest entry declares**, and write one
JSON array:

```json
[
  {{"file": "core-minimal.medh5", "errors": [], "warnings": []}},
  {{"file": "E102-non-orthonormal.medh5", "errors": ["E102"], "warnings": []}}
]
```

Then score it:

```
medh5 conformance score . results.json
```

The scorer also takes the shape `medh5 validate --json` reports a file in: a
`diagnostics` list, each entry with its `code` and `severity`, in place of
`errors` and `warnings` (and `path` in place of `file`). So the reference
implementation is scored through exactly the same door as everybody else:

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

For each case, the errors you report must **equal** the expected errors, and
the warnings the expected warnings: a missing code is a defect you failed to
catch, an extra code is a valid file you rejected. Both fail. Severity is
scored: an expected error you report as a warning is missing as an error, and
a file you report only warnings for is a file you accepted. A case you report
nothing about fails too --- silence about a file you were handed is not a pass.

Diagnostic *messages* are yours to write; only the codes are normative.

**An implementation of an earlier version is scored on what it was told
about.** Each case states its `format_version` (the latest version among the
codes it expects; not always the one its file declares), and the manifest the
`codes_revision` of the code table it was built against. `--format-version
1.0` scores only the cases addressed to 1.0, and `--codes-revision 1` only
those whose codes revision 1 holds, ignoring any later code you report; the
cases left out are counted as skipped.

## Three things worth knowing before you start

**Validate at the declared `level`, not deeper.** `structural` < `semantic` <
`integrity`. Shallower misses the defect the case exists to test. *Deeper is
not safe either*: most invalid cases were made by editing a valid file, so
their stored digests cover the pre-edit bytes and an integrity pass adds a
`content_id` mismatch the case never claimed. Those cases are marked
`"mutated": true`, and the mismatch is an artifact of how they are built, not
something to diagnose.

**A `.medh5c` case is a collection** (spec §2.2) --- it contains samples rather
than being one. `"file_suffix"` in the manifest tells you which.

**Verify the bytes first.** `{CHECKSUMS}` covers every published file, and
`medh5 conformance score` fails the score when one has drifted, whatever the
cases say, naming it (under `changed` with `--json`). A score over files that
are not the published files is not a score.

The specification is `docs/spec/medh5-1.0.md` with `docs/spec/medh5-1.1.md`
in the medh5 repository, and every case names the clause it tests in
`expected.json`.

## The task-and-cache fixtures

`companion/` holds task manifests (`medh5.task/1`) over this suite's own
clinical samples, each pinned to the content those samples have here, and
`companion/expected.json`: for each, the level it is checked at ---
`validate` (the manifest alone, no file opened) or `preflight` (its sources
opened too) --- the exact set of `T` codes checking it must find, and for a
valid task each row's status, selected event versions, slot images and target
label. They score an implementation of the separately versioned task-and-cache
contract (`docs/spec/task-cache-1.md`), not the format. Resolve each source
`uri` against the manifest's directory. With the reference implementation:

```
medh5 task validate companion/<fixture>.task.json --json
medh5 task preflight companion/<fixture>.task.json --json
```
"#
    )
}
