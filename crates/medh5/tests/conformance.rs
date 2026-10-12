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
    let written = write_fixtures(dir.path()).unwrap();
    assert!(written.len() >= 14);
    let codes: std::collections::BTreeSet<String> = written.iter().flat_map(|f| f.findings.iter().cloned()).collect();
    for code in ["T101", "T102", "T103", "T201", "T202", "T203", "T204", "T301", "T302", "T303", "T304", "T305", "T306"]
    {
        assert!(codes.contains(code), "no fixture for {code}");
    }
    let results = run_fixtures(dir.path()).unwrap();
    let failures: Vec<_> = results.iter().filter(|r| !r.ok()).collect();
    assert!(failures.is_empty(), "{failures:#?}");

    // A published suite carries them, under its checksums.
    let suite = tempfile::tempdir().unwrap();
    publish(suite.path(), Some(&names)).unwrap();
    assert!(suite.path().join("companion/expected.json").exists());
    assert!(check_checksums(suite.path()).unwrap().is_empty());
}
