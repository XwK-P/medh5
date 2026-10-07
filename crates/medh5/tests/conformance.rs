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
        .map(|r| format!("{}: missing {:?}, unexpected {:?}, error {:?}", r.case.name, r.missing, r.unexpected, r.error))
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
