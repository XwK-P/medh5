//! De-identification through the public API (§11.4): what a batch keeps
//! when one of its files is refused.

use std::path::Path;

use medh5::conformance::clinical::worked_example;
use medh5::curation::scrub::{scrub_each, ScrubOptions};
use medh5::sample::{create, open_sample, GridOptions, ImageOptions};

fn plain(path: &Path, subject: &str) {
    let ct = medh5::array::NdArray::from_vec(&[4, 8, 8], vec![0i16; 4 * 8 * 8]).unwrap();
    let mut w = create(path, Some("s"), Some(subject), "balanced", &[]).unwrap();
    let mut timepoint = serde_json::Map::new();
    timepoint.insert("study_uid".into(), "1.2.840.113619.2.55.3".into());
    w.add_timepoint("tp0", timepoint).unwrap();
    w.add_grid("ct", &[4, 8, 8], &[2.0, 0.8, 0.8], GridOptions::default()).unwrap();
    w.add_image("CT", &ct, "ct", "CT", ImageOptions::default()).unwrap();
    w.commit(true).unwrap();
}

/// `scrub_paths` collected an iterator of results, so the first refused
/// file ended the call and the reports of the files already rewritten ---
/// their `uid_map` the only key to the pseudonyms written --- were lost.
#[test]
fn s11_4_scrub_each_keeps_reports_after_a_refusal() {
    let dir = tempfile::tempdir().unwrap();
    let (first, clinical, last) = (dir.path().join("a.medh5"), dir.path().join("c.medh5"), dir.path().join("b.medh5"));
    plain(&first, "subj-1");
    worked_example(&clinical).unwrap();
    plain(&last, "subj-2");
    let untouched = open_sample(&clinical).unwrap().content_id().unwrap();
    let options = ScrubOptions::default().with_salt("pep").with_withhold_shift(true);
    let results = scrub_each(&[&first, &clinical, &last], true, &options).unwrap();
    assert_eq!(results.len(), 3);
    assert!(results[1].is_err(), "the clinical sample's text is not attested");
    for (result, path) in [(&results[0], &first), (&results[2], &last)] {
        let report = result.as_ref().unwrap();
        assert!(report.applied && report.ok());
        assert!(report.uid_map.contains_key("1.2.840.113619.2.55.3"), "{:?}", report.uid_map);
        assert!(open_sample(path).unwrap().document().unwrap().deidentification.is_some());
    }
    assert_eq!(open_sample(&clinical).unwrap().content_id().unwrap(), untouched);
    // An option no scan acts on is refused before any file is read.
    assert!(scrub_each(&[&first], false, &options).is_err());
}
