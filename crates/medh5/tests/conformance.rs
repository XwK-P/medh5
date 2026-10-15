//! The conformance corpus, end to end: every case is written by this engine's
//! writer (and mutated where the case says so), then validated by this
//! engine's validator at the case's declared level, and must report exactly
//! the codes the case expects (spec §15).

use medh5::conformance::{build_corpus, cases, check_checksums, load_manifest, publish, run_corpus, score, summarize};
use serde_json::{json, Value};

#[test]
fn every_case_reports_exactly_its_expected_codes() {
    let dir = tempfile::tempdir().unwrap();
    let results = run_corpus(dir.path(), None).unwrap();
    assert_eq!(results.len(), cases().len());
    let failures: Vec<String> = results
        .iter()
        .filter(|r| !r.ok())
        .map(|r| {
            format!("{}: missing {:?}, unexpected {:?}, error {:?}", r.case.name, r.missing, r.unexpected, r.error)
        })
        .collect();
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn the_corpus_directory_holds_exactly_the_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let names = vec!["core-minimal".to_string(), "collection-two-samples".to_string()];
    build_corpus(dir.path(), Some(&names)).unwrap();
    let mut listed: Vec<String> =
        std::fs::read_dir(dir.path()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    listed.sort();
    assert_eq!(listed, ["collection-two-samples.medh5c", "core-minimal.medh5", "expected.json"]);
}

#[test]
fn a_published_suite_scores_a_foreign_report() {
    let dir = tempfile::tempdir().unwrap();
    let names = vec!["core-minimal".to_string(), "E001-missing-version".to_string()];
    publish(dir.path(), Some(&names)).unwrap();
    assert!(check_checksums(dir.path()).unwrap().is_empty());
    assert_eq!(load_manifest(dir.path()).unwrap()["cases"].as_array().unwrap().len(), 2);

    let submitted = vec![
        json!({"file": "core-minimal.medh5", "errors": [], "warnings": []}),
        json!({"path": "/elsewhere/E001-missing-version.medh5",
               "diagnostics": [{"code": "E001", "severity": "error"}]}),
    ];
    let results = score(dir.path(), &submitted).unwrap();
    assert!(results.iter().all(|r| r.ok()), "{:?}", summarize(&results));

    // Silence about a case is a failure, and a drifted file is named.
    let results = score(dir.path(), &submitted[..1]).unwrap();
    let summary: Value = summarize(&results);
    assert_eq!(summary["failed"], 1);
    std::fs::write(dir.path().join("core-minimal.medh5"), b"not the published bytes").unwrap();
    assert_eq!(check_checksums(dir.path()).unwrap(), ["core-minimal.medh5"]);
}

/// Severity is scored: an expected error reported as a warning is a defect
/// not reported, and a warning reported as an error a file rejected.  The
/// codes alone were compared, so a validator that reported every code as a
/// warning --- and so rejected nothing --- scored every case.
#[test]
fn s15_2_an_error_reported_as_a_warning_fails_its_case() {
    let dir = tempfile::tempdir().unwrap();
    let names = vec!["E001-missing-version".to_string(), "W903-no-deidentification".to_string()];
    publish(dir.path(), Some(&names)).unwrap();
    let report = |e001: &str, w903: &str| {
        vec![
            json!({"file": "E001-missing-version.medh5", "diagnostics": [{"code": "E001", "severity": e001}]}),
            json!({"file": "W903-no-deidentification.medh5", "diagnostics": [{"code": "W903", "severity": w903}]}),
        ]
    };
    let scored = |submitted: Vec<Value>| -> Vec<(String, bool, Vec<String>, Vec<String>)> {
        score(dir.path(), &submitted)
            .unwrap()
            .into_iter()
            .map(|r| (r.case.name.clone(), r.ok(), r.missing.clone(), r.unexpected.clone()))
            .collect()
    };
    assert!(scored(report("error", "warning")).iter().all(|(_, ok, _, _)| *ok));
    let swapped = scored(report("warning", "error"));
    assert_eq!(
        swapped,
        [
            ("E001-missing-version".to_string(), false, vec!["E001".to_string()], vec!["E001".to_string()]),
            ("W903-no-deidentification".to_string(), false, vec!["W903".to_string()], vec!["W903".to_string()]),
        ]
    );
    // The foreign shape is scored the same way.
    let lists = vec![
        json!({"file": "E001-missing-version.medh5", "errors": [], "warnings": ["E001"]}),
        json!({"file": "W903-no-deidentification.medh5", "errors": [], "warnings": ["W903"]}),
    ];
    let results = score(dir.path(), &lists).unwrap();
    assert_eq!(results.iter().map(|r| r.ok()).collect::<Vec<_>>(), [false, true]);
}

#[test]
fn the_task_and_cache_fixtures_hold() {
    use medh5::conformance::companion::{run_fixtures, write_fixtures, SOURCES};
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = SOURCES.iter().map(|f| f.trim_end_matches(".medh5").to_string()).collect();
    build_corpus(dir.path(), Some(&names)).unwrap();
    write_fixtures(dir.path()).unwrap();
    // Every code of the contract has a fixture, the caches' with the tasks'
    // (task-cache-1 §9): a code added without one fails here.
    let expected = medh5::conformance::companion::load_expected(dir.path()).unwrap();
    let fixtures = expected["fixtures"].as_array().unwrap();
    let codes: std::collections::BTreeSet<&str> =
        fixtures.iter().flat_map(|f| f["findings"].as_array().unwrap()).filter_map(Value::as_str).collect();
    let table: std::collections::BTreeSet<&str> = medh5::companion::CODES.iter().map(|(code, _)| *code).collect();
    assert_eq!(codes, table);
    let results = run_fixtures(dir.path()).unwrap();
    assert_eq!(results.len(), fixtures.len());
    let failures: Vec<_> = results.iter().filter(|r| !r.ok()).collect();
    assert!(failures.is_empty(), "{failures:#?}");

    // A published suite carries them, under its checksums.
    let suite = tempfile::tempdir().unwrap();
    publish(suite.path(), Some(&names)).unwrap();
    assert!(suite.path().join("companion/expected.json").exists());
    assert!(suite.path().join("companion/caches/valid-patient.medh5cache").exists());
    assert!(check_checksums(suite.path()).unwrap().is_empty());
}

/// task-cache-1 §8: the scorer reads another implementation's reports, and a
/// cache report that misses its finding, or a row it does not state, fails.
#[test]
fn s8_the_companion_scorer_scores_reports() {
    use medh5::conformance::companion::{load_expected, score, SOURCES};
    let dir = tempfile::tempdir().unwrap();
    let names: Vec<String> = SOURCES.iter().map(|f| f.trim_end_matches(".medh5").to_string()).collect();
    publish(dir.path(), Some(&names)).unwrap();
    let expected = load_expected(dir.path()).unwrap();
    // The fixtures' own findings, as a minimal report.
    let reports: Vec<Value> = expected["fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|f| f["rows"].as_object().is_none_or(|rows| rows.is_empty()))
        .map(|f| json!({"file": f["file"], "findings": f["findings"]}))
        .collect();
    let results = score(dir.path(), &reports).unwrap();
    let failed: Vec<&str> = results.iter().filter(|r| !r.ok()).map(|r| r.case.name.as_str()).collect();
    // Only the fixtures stating rows, which this report leaves out.
    assert_eq!(failed, ["valid-two-subjects", "valid-self-asserted-mask"]);
    let mut wrong = reports.clone();
    let t407 = wrong.iter_mut().find(|r| r["file"] == "caches/T407-another-versions-text.medh5cache").unwrap();
    t407["findings"] = json!([]);
    let results = score(dir.path(), &wrong).unwrap();
    let t407 = results.iter().find(|r| r.case.name == "T407-another-versions-text").unwrap();
    assert_eq!(t407.missing, ["T407"]);
}

/// A suite built twice is the same bytes, so a mirror can check a published
/// suite by building it again and comparing `SHA256SUMS`.  No clause of the
/// specification asks for this: group timestamps and `created` are not format
/// content, and no digest or `content_id` covers them.  It is what lets a
/// third party trust a copy of the suite without trusting the copy.
#[test]
fn a_corpus_published_twice_is_the_same_bytes() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    publish(first.path(), None).unwrap();
    // HDF5 stamps times in whole seconds: let one pass between the builds.
    std::thread::sleep(std::time::Duration::from_millis(1100));
    publish(second.path(), None).unwrap();
    let sums = |dir: &std::path::Path| std::fs::read_to_string(dir.join("SHA256SUMS")).unwrap();
    let (a, b) = (sums(first.path()), sums(second.path()));
    let differing: Vec<&str> =
        a.lines().zip(b.lines()).filter(|(x, y)| x != y).map(|(x, _)| x.split_once("  ").map_or(x, |p| p.1)).collect();
    assert!(differing.is_empty(), "{} files differ: {differing:?}", differing.len());
    assert_eq!(a, b);
}

/// Every group the engine creates goes through `h5::ops::new_group`, which
/// turns HDF5's object timestamps off: a `create_group` stamps the time, and
/// two writes of the same content differ in bytes.  Test code may still call
/// it.
#[test]
fn no_shipped_code_creates_a_timed_group() {
    fn walk(dir: &std::path::Path, found: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                walk(&path, found);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                let shipped = text.split("#[cfg(test)]").next().unwrap_or(&text);
                for (n, line) in shipped.lines().enumerate() {
                    if line.contains(".create_group(") {
                        found.push(format!("{}:{}", path.display(), n + 1));
                    }
                }
            }
        }
    }
    let mut found = Vec::new();
    walk(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut found);
    assert!(found.is_empty(), "create through medh5::h5::ops::new_group: {found:#?}");
}
