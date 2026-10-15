//! The clinical profile under `medh5 scrub` (1.1 §3, §6, §7.3, §10; D15).
//!
//! **What is read.**  The clock, the id columns, the short text columns
//! (`value_text`, `missing_reason`, the clock's `origin_description`) and each
//! document, one at a time: a document's bytes are read when its turn comes
//! and dropped after, so no column is decoded whole.  The name, date, UID and
//! record-number rules run over every text, and report where they matched in
//! UTF-8 bytes, which is what a span measures.
//!
//! **What is not done.**  Text is never rewritten: a span into it would point
//! somewhere else.  Whoever de-identified it attests that, and the attestation
//! holds the digests of the bytes it covers, so it lapses when they change.
//! Until then the text blocks `--apply` (rule `clinical`, the name 2.0 gave
//! the whole profile).
//!
//! **What `--apply` writes.**  A derivation, not an amendment (D15): the
//! source's event versions cannot be changed under their ids (§7.3), so the
//! records are written again as new versions of a new sample --- times moved
//! by the one shift `/meta`'s dates take, a `utc` clock become `shifted_utc`,
//! and every id re-minted from the salt.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Map, Value};

use super::{id_tokens, is_dicom_date, is_locale_date, iso_dates, regex, uid_tokens, ScrubReport};
use crate::clinical::model::{Bounds, ClinicalRecords, Clock, Descriptor, Event, Link, DAY};
use crate::clinical::Clinical;
use crate::h5::ops;
use crate::json::repr_str;
use crate::sample::Sample;
use crate::{Error, Result};

/// Text no one has attested as de-identified --- the rule 2.0 gave the whole
/// profile, kept so a gate on it still holds.
pub const CLINICAL: &str = "clinical";
/// A `utc` clock: every time on it is the subject's own.
pub const CLINICAL_CLOCK: &str = "clinical_clock";
/// An id of the profile's that may name the subject or the source system.
pub const CLINICAL_ID: &str = "clinical_id";
/// A name, date, UID or record number in the profile's text.
pub const CLINICAL_TEXT: &str = "clinical_text";

/// What a `deidentify` activity attesting clinical text names as its scope.
pub const TEXT_SCOPE: &str = "clinical_text";
/// What a re-minted id starts with.  Ids are `[A-Za-z0-9_.-]` (1.1 §4), so
/// not the `pseudo:` of a UID's pseudonym.
pub const PSEUDO_ID: &str = "pseudo.";
/// Where an attestation keeps the digest of the short text columns.
const SHORT_TEXTS: &str = "clinical/events";

/// The attestation key of a document's text.
fn document_key(document_id: &str) -> String {
    format!("clinical/documents/{document_id}")
}

/// §13.1's digest of a text's UTF-8 bytes.
fn text_digest(text: &str) -> String {
    crate::digest::digest_bytes(text.as_bytes(), "sha256").expect("sha256 is always available")
}

/// The short texts as one value: distinct and sorted, so a re-minted event
/// id or a reordered table does not change it.
fn short_digest(texts: &BTreeSet<String>) -> Option<String> {
    (!texts.is_empty()).then(|| text_digest(&crate::json::canonical(&json!(texts))))
}

/// What a sample's clinical profile is to the scrub.
pub(super) enum Profile {
    Absent,
    /// A group this tool cannot read as the profile: undeclared, or tables
    /// that do not read.  Reported, and never scrubbed.
    Unread(String),
    Examined(Examined),
}

/// What [`examine`] found that `apply` decides on.
pub(super) struct Examined {
    pub utc: bool,
    /// Whether an id was reported (`clinical_id`).
    pub ids: bool,
    /// The digests of the text as it is now, by attestation key.
    pub digests: BTreeMap<String, String>,
    /// The tool whose attestation still holds, by attestation key.
    pub attested_by: BTreeMap<String, String>,
    /// Text no attestation covers, by attestation key.
    pub unattested: Vec<String>,
    /// Documents whose bytes do not read: never attestable.
    pub unreadable: Vec<String>,
    /// Where a name, a UID or one of the sample's ids is in the text.
    pub blocking: Vec<String>,
}

/// Each attested key's digest, and the tool that attested it: the latest
/// attestation of a key replaces an earlier one.
fn attestations(doc: &Map<String, Value>) -> BTreeMap<String, (String, String)> {
    let mut out = BTreeMap::new();
    let activities = doc.get("provenance").and_then(|p| p.get("activities")).and_then(Value::as_array);
    for params in activities
        .into_iter()
        .flatten()
        .filter(|a| a.get("type").and_then(Value::as_str) == Some("deidentify"))
        .filter_map(|a| a.get("params").and_then(Value::as_object))
        .filter(|p| p.get("scope").and_then(Value::as_str) == Some(TEXT_SCOPE))
    {
        let tool = params.get("tool").and_then(Value::as_str).unwrap_or_default();
        for (key, digest) in params.get("digests").and_then(Value::as_object).into_iter().flatten() {
            if let Some(digest) = digest.as_str() {
                out.insert(key.clone(), (digest.to_string(), tool.to_string()));
            }
        }
    }
    out
}

/// Where in a text the rules matched, as byte spans.
struct Hits {
    names: Vec<(usize, usize)>,
    dates: Vec<(usize, usize)>,
    uids: Vec<(usize, usize)>,
    ids: Vec<(usize, usize)>,
}

/// A name in DICOM's `Family^Given` form anywhere in a text.
fn name_spans(text: &str) -> Vec<(usize, usize)> {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex(&CELL, r"[\p{L}][\p{L}'-]*\^[\p{L}][\p{L}'-]*").find_iter(text).map(|m| (m.start(), m.end())).collect()
}

/// A date in a locale's order (`31.01.2020`, `01/31/2020`) anywhere in a text.
fn locale_spans(text: &str) -> Vec<(usize, usize)> {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex(&CELL, r"[0-9]{1,2}[./][0-9]{1,2}[./][0-9]{4}")
        .find_iter(text)
        .filter(|m| {
            let digit_before = text[..m.start()].chars().next_back().is_some_and(|c| c.is_ascii_digit());
            let digit_after = text[m.end()..].chars().next().is_some_and(|c| c.is_ascii_digit());
            !digit_before && !digit_after && is_locale_date(m.as_str())
        })
        .map(|m| (m.start(), m.end()))
        .collect()
}

fn hits(text: &str, ids: &[String]) -> Hits {
    let mut dates: Vec<(usize, usize)> = iso_dates(text).iter().map(|d| (d.start, d.end)).collect();
    dates.extend(locale_spans(text));
    if dates.is_empty() && is_dicom_date(text) {
        dates.push((0, text.len()));
    }
    dates.sort_unstable();
    let mut found_ids: Vec<(usize, usize)> = ids.iter().flat_map(|id| id_tokens(text, id)).collect();
    found_ids.sort_unstable();
    Hits { names: name_spans(text), dates, uids: uid_tokens(text), ids: found_ids }
}

/// Whether an id says more than an opaque key would.
fn telling(id: &str, ids: &[String]) -> bool {
    let h = hits(id, ids);
    !(h.names.is_empty() && h.dates.is_empty() && h.uids.is_empty() && h.ids.is_empty())
}

fn spans(found: &[(usize, usize)]) -> String {
    let shown: Vec<String> = found.iter().take(5).map(|(a, b)| format!("[{a}, {b})")).collect();
    let more = found.len().saturating_sub(5);
    format!("{}{}", shown.join(", "), if more > 0 { format!(", and {more} more") } else { String::new() })
}

/// Report what the rules find in one text; whether a hit blocks `apply`.
fn report_text(report: &mut ScrubReport, text: &str, r#where: &str, ids: &[String]) -> bool {
    let h = hits(text, ids);
    let rules = [
        ("name(s) in DICOM's Family^Given form", &h.names, "a person's name", true),
        ("UID(s)", &h.uids, "a real UID ties the record to the source system", true),
        ("of the sample's ids", &h.ids, "the record number the sample's ids replace", true),
        (
            "date(s)",
            &h.dates,
            "this tool cannot tell a date the de-identifying tool shifted from one it missed, so a person must check \
             it",
            false,
        ),
    ];
    let mut blocks = false;
    for (what, found, why, blocking) in rules {
        let Some(&(a, b)) = found.first() else { continue };
        let detail = format!(
            "{} {what} in the clinical text, at byte(s) {} --- {why}{}",
            found.len(),
            spans(found),
            if blocking { "; --apply refuses while it is there" } else { "" }
        );
        report.add(CLINICAL_TEXT, r#where, &detail, Some(&text[a..b]), false, false);
        blocks |= blocking;
    }
    blocks
}

/// Find what identifies the subject in a sample's clinical profile, and add
/// it to `report`.  `ids` are the sample's ids as the scrub looks for them.
pub(super) fn examine(
    sample: &Sample,
    doc: &Map<String, Value>,
    ids: &[String],
    report: &mut ScrubReport,
) -> Result<Profile> {
    let undeclared = !sample.profiles()?.contains(crate::clinical::PROFILE);
    let clinical = match sample.clinical() {
        _ if undeclared && !ops::exists(&sample.root, crate::clinical::GROUP) => return Ok(Profile::Absent),
        _ if undeclared => Err("a `clinical` group the sample does not declare".to_string()),
        Ok(Some(clinical)) => Ok(clinical),
        Ok(None) => return Ok(Profile::Absent),
        Err(e) => Err(format!("clinical tables this tool could not read ({e})")),
    };
    let clinical: &Clinical = match clinical {
        Ok(c) => c,
        Err(what) => {
            report.add(
                CLINICAL,
                "clinical",
                &format!(
                    "{what}: not examined by this tool --- a person must review it, or share the imaging projection \
                     (`medh5 clinical strip`)"
                ),
                None,
                false,
                false,
            );
            report.not_checked.push(format!("the clinical profile: {what}"));
            return Ok(Profile::Unread(what));
        }
    };
    let clock = &clinical.descriptor.clock;
    let utc = clock.reference == "utc";
    if utc {
        report.add(
            CLINICAL_CLOCK,
            "clinical/meta.clock",
            "a utc clock: every time on it is the subject's own --- --apply moves every time by the date shift and \
             makes the clock shifted_utc (1.1 §3); give --date-shift-days or --date-shift-range",
            Some(&clock.id),
            true,
            true,
        );
    }
    let mut reported_ids = false;
    let mut id_finding = |report: &mut ScrubReport, r#where: &str, found: Vec<&str>, detail: &str| {
        if let Some(first) = found.first() {
            report.add(
                CLINICAL_ID,
                r#where,
                &format!(
                    "{} {detail} --- --apply --pseudonymise-ids --salt re-mints every id of the profile from the \
                     salt, consistently",
                    found.len()
                ),
                Some(first),
                false,
                true,
            );
            reported_ids = true;
        }
    };
    if !clock.id.starts_with(PSEUDO_ID) {
        id_finding(
            report,
            "clinical/meta.clock.id",
            vec![&clock.id],
            "clock id, shared by every fragment of the subject: one salt re-mints it alike in each, so they still \
             share a clock (task-cache-1 T304)",
        );
    }
    let encounters: BTreeSet<&str> = clinical
        .events
        .iter()
        .filter_map(|e| e.encounter_id.as_deref())
        .filter(|id| !id.starts_with(PSEUDO_ID))
        .collect();
    id_finding(
        report,
        "clinical/events.encounter_id",
        encounters.into_iter().collect(),
        "encounter id(s), the source's",
    );
    let telling_of = |found: &mut dyn Iterator<Item = &str>| -> Vec<String> {
        let distinct: BTreeSet<&str> = found.filter(|id| !id.starts_with(PSEUDO_ID) && telling(id, ids)).collect();
        distinct.into_iter().map(str::to_string).collect()
    };
    for (column, found) in [
        ("events.event_id", telling_of(&mut clinical.events.iter().map(|e| e.event_id.as_str()))),
        ("events.record_id", telling_of(&mut clinical.events.iter().map(|e| e.record_id.as_str()))),
        ("documents.document_id", telling_of(&mut clinical.documents().iter().map(|d| d.document_id.as_str()))),
    ] {
        let found: Vec<&str> = found.iter().map(String::as_str).collect();
        id_finding(
            report,
            &format!("clinical/{column}"),
            found,
            "id(s) holding a name, a date, a UID or the sample's ids",
        );
    }

    let attested = attestations(doc);
    let mut found = Examined {
        utc,
        ids: reported_ids,
        digests: BTreeMap::new(),
        attested_by: BTreeMap::new(),
        unattested: Vec::new(),
        unreadable: Vec::new(),
        blocking: Vec::new(),
    };
    // The short texts, one event at a time.
    let mut short = BTreeSet::new();
    if let Some(origin) = &clock.origin_description {
        if report_text(report, origin, "clinical/meta.clock.origin_description", ids) {
            found.blocking.push("clinical/meta.clock.origin_description".into());
        }
        short.insert(origin.clone());
    }
    for event in &clinical.events {
        for (field, value) in [("value_text", &event.value_text), ("missing_reason", &event.missing_reason)] {
            let Some(text) = value else { continue };
            let r#where = format!("clinical/events/{}.{field}", event.event_id);
            if report_text(report, text, &r#where, ids) {
                found.blocking.push(r#where);
            }
            short.insert(text.clone());
        }
    }
    let mut texts: Vec<(String, Option<String>)> = vec![(SHORT_TEXTS.to_string(), short_digest(&short))];
    // Each document's bytes, read in turn and dropped.
    for info in clinical.documents() {
        let key = document_key(&info.document_id);
        match clinical.text(&info.document_id) {
            Ok(text) => {
                if report_text(report, &text, &key, ids) {
                    found.blocking.push(key.clone());
                }
                texts.push((key, Some(text_digest(&text))));
            }
            Err(e) => {
                report.add(
                    "unreadable",
                    &key,
                    &format!("a document whose text does not read ({e}): no one can attest it, so a person must"),
                    None,
                    false,
                    false,
                );
                found.unreadable.push(key);
            }
        }
    }
    for (key, digest) in texts {
        let Some(digest) = digest else { continue };
        match attested.get(&key) {
            Some((was, tool)) if *was == digest => {
                found.attested_by.insert(key.clone(), tool.clone());
            }
            _ => {
                let what = if key == SHORT_TEXTS {
                    "the short text columns (value_text, missing_reason, the clock's origin_description)"
                } else {
                    "a document"
                };
                report.add(
                    CLINICAL,
                    &key,
                    &format!(
                        "free text not attested as de-identified: {what} this tool checks only for names, dates, UIDs \
                         and record numbers --- once a person or a tool has de-identified it, --apply --attest-text \
                         TOOL records that with the digest of these bytes (D15)"
                    ),
                    None,
                    false,
                    false,
                );
                found.unattested.push(key.clone());
            }
        }
        found.digests.insert(key, digest);
    }
    let tools: BTreeSet<&String> = found.attested_by.values().collect();
    if !tools.is_empty() {
        report.not_checked.push(format!(
            "clinical text beyond its names, dates, UIDs and record numbers: attested as de-identified by {}",
            tools.iter().map(|t| repr_str(t)).collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(Profile::Examined(found))
}

/// The refusal for a profile `apply` cannot write as de-identified, if any.
pub(super) fn refusal(found: &Examined, attesting: bool, shown: &str) -> Option<String> {
    let listed = |keys: &[String]| {
        let shown: Vec<String> = keys.iter().take(3).map(|k| repr_str(k)).collect();
        let more = keys.len().saturating_sub(3);
        format!("{}{}", shown.join(", "), if more > 0 { format!(" and {more} more") } else { String::new() })
    };
    if !found.unreadable.is_empty() {
        return Some(format!(
            "{shown} holds clinical documents whose text does not read ({}): no one can attest them, and this tool \
             does not write what it cannot read --- repair them, or share the imaging projection (`medh5 clinical \
             strip`)",
            listed(&found.unreadable)
        ));
    }
    if !found.blocking.is_empty() {
        return Some(format!(
            "{shown} holds a name, a UID or one of the sample's ids in its clinical text ({}): this tool does not \
             rewrite text, whose spans would then point elsewhere --- de-identify it and write the result as a new \
             sample, or share the imaging projection (`medh5 clinical strip`)",
            listed(&found.blocking)
        ));
    }
    if !attesting && !found.unattested.is_empty() {
        return Some(format!(
            "{shown} holds clinical text no one has attested as de-identified ({}): this tool checks it only for \
             names, dates, UIDs and record numbers, and an attestation that it was scrubbed would claim more --- once \
             it is de-identified, pass --attest-text TOOL (attest_text), or share the imaging projection (`medh5 \
             clinical strip`)",
            listed(&found.unattested)
        ));
    }
    None
}

/// The `deidentify` activity's params attesting the text by its digests.
pub(super) fn attestation(tool: &str, digests: &BTreeMap<String, String>) -> Map<String, Value> {
    let mut params = Map::new();
    params.insert("scope".into(), json!(TEXT_SCOPE));
    params.insert("tool".into(), json!(tool));
    params.insert("digests".into(), json!(digests));
    params
}

/// One id [`derive`] replaced.
pub(super) struct Reminted {
    pub kind: &'static str,
    pub old: String,
    pub new: String,
}

/// An id of the profile's re-minted from the salt and the subject: the same
/// subject, old id and salt give the same new one, in every fragment, and two
/// subjects that share an id (a clock named `ehr`, each shifted by its own
/// amount) do not share its new one.  `subject` is the subject's salted
/// pseudonym.
pub(super) fn remint(kind: &str, old: &str, salt: &str, subject: &str) -> String {
    let mac = super::hmac_sha256(salt.as_bytes(), format!("{subject}\n{kind}:{old}").as_bytes());
    format!("{PSEUDO_ID}{kind}.{}", &hex::encode(mac)[..24])
}

/// The records written again as a new sample's (D15): every bound moved by
/// `shift_us`, a `utc` clock become `shifted_utc`, and every id re-minted,
/// with each link and each attestation key following.  The old and new ids,
/// for the report's key.
pub(super) fn derive(
    records: ClinicalRecords,
    shift_us: i64,
    salt: &str,
    subject: &str,
) -> Result<(ClinicalRecords, Vec<Reminted>)> {
    let mut replaced: Vec<Reminted> = Vec::new();
    let mut minted: HashMap<(&'static str, String), String> = HashMap::new();
    let mut mint = |kind: &'static str, old: &str, replaced: &mut Vec<Reminted>| -> String {
        minted
            .entry((kind, old.to_string()))
            .or_insert_with(|| {
                let new = remint(kind, old, salt, subject);
                replaced.push(Reminted { kind, old: old.to_string(), new: new.clone() });
                new
            })
            .clone()
    };
    let old = records.descriptor.clock;
    let clock = Clock {
        id: mint("clock", &old.id, &mut replaced),
        reference: if old.reference == "utc" { "shifted_utc".into() } else { old.reference.clone() },
        ..old.clone()
    };
    let moved = |bounds: Option<Bounds>, event: &str| -> Result<Option<Bounds>> {
        let Some(b) = bounds else { return Ok(None) };
        match (b.lo.checked_add(shift_us), b.hi.checked_add(shift_us)) {
            (Some(lo), Some(hi)) => Ok(Some(Bounds::new(lo, hi))),
            _ => Err(Error::invalid(format!(
                "event {}: a time moved by the date shift leaves the range of the clock's 64-bit microseconds",
                repr_str(event)
            ))),
        }
    };
    let mut events = Vec::with_capacity(records.events.len());
    for event in records.events {
        let id = event.event_id.clone();
        events.push(Event {
            event_id: mint("event", &event.event_id, &mut replaced),
            record_id: mint("record", &event.record_id, &mut replaced),
            effective_start: moved(event.effective_start, &id)?,
            effective_end: moved(event.effective_end, &id)?,
            available: moved(event.available, &id)?,
            encounter_id: event.encounter_id.as_deref().map(|e| mint("encounter", e, &mut replaced)),
            ..event
        });
    }
    let documents = records
        .documents
        .into_iter()
        .map(|d| crate::clinical::model::Document { document_id: mint("document", &d.document_id, &mut replaced), ..d })
        .collect();
    // Only the endpoints the profile names are re-minted: an image, a
    // timepoint or an annotation keeps the id the imaging half gives it.
    let mut endpoint = |kind: &str, id: &str, replaced: &mut Vec<Reminted>| match kind {
        "event" => mint("event", id, replaced),
        "document" => mint("document", id, replaced),
        _ => id.to_string(),
    };
    let links = records
        .links
        .into_iter()
        .map(|link| Link {
            source_id: endpoint(&link.source_type, &link.source_id, &mut replaced),
            target_id: endpoint(&link.target_type, &link.target_id, &mut replaced),
            asserted_by_event_id: link.asserted_by_event_id.as_deref().map(|e| endpoint("event", e, &mut replaced)),
            ..link
        })
        .collect();
    let derived = ClinicalRecords { descriptor: Descriptor::new(clock), events, documents, links };
    Ok((derived, replaced))
}

/// The attestations to write: `tool`'s over every text when it is named
/// now; else, after a derivation, each earlier tool's over the texts it
/// attested, under their new keys.
pub(super) fn attestations_to_write(
    found: &Examined,
    tool: Option<&str>,
    digests: &BTreeMap<String, String>,
    reminted: Option<&[Reminted]>,
) -> Vec<(String, BTreeMap<String, String>)> {
    if let Some(tool) = tool {
        return vec![(tool.to_string(), digests.clone())];
    }
    let Some(reminted) = reminted else { return Vec::new() };
    let mut by_tool: BTreeMap<String, BTreeMap<String, String>> = BTreeMap::new();
    for (key, tool) in &found.attested_by {
        if let Some(digest) = found.digests.get(key) {
            let single = BTreeMap::from([(key.clone(), digest.clone())]);
            by_tool.entry(tool.clone()).or_default().extend(rekeyed(&single, reminted));
        }
    }
    by_tool.into_iter().collect()
}

/// The attestation keys of `digests` under the re-minted document ids.
pub(super) fn rekeyed(digests: &BTreeMap<String, String>, replaced: &[Reminted]) -> BTreeMap<String, String> {
    let renamed: HashMap<String, &str> =
        replaced.iter().filter(|r| r.kind == "document").map(|r| (document_key(&r.old), r.new.as_str())).collect();
    digests
        .iter()
        .map(|(key, digest)| match renamed.get(key) {
            Some(new) => (document_key(new), digest.clone()),
            None => (key.clone(), digest.clone()),
        })
        .collect()
}

/// The one shift in microseconds.
pub(super) fn shift_us(days: i64) -> Result<i64> {
    days.checked_mul(DAY).ok_or_else(|| Error::invalid(format!("a shift of {days} days is past the clock's range")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::clinical::worked_records;

    fn utc_records() -> ClinicalRecords {
        let mut records = worked_records();
        records.descriptor = Descriptor::new(Clock::utc("ehr"));
        records
    }

    #[test]
    fn s10_a_derivation_moves_every_bound_and_remints_every_id() {
        let source = utc_records();
        let shift = -30 * DAY;
        let (derived, reminted) = derive(source.clone(), shift, "pep", "s1").unwrap();
        assert_eq!(derived.descriptor.clock.reference, "shifted_utc");
        assert_eq!(derived.descriptor.clock.id, remint("clock", "ehr", "pep", "s1"));
        let id = |kind: &str, old: &str| reminted.iter().find(|r| r.kind == kind && r.old == old).unwrap().new.clone();
        for (was, now) in source.events.iter().zip(&derived.events) {
            assert_eq!(now.event_id, id("event", &was.event_id));
            assert_eq!(now.record_id, id("record", &was.record_id));
            for (a, b) in [(&was.effective_start, &now.effective_start), (&was.available, &now.available)] {
                assert_eq!(a.map(|x| (x.lo + shift, x.hi + shift)), b.map(|x| (x.lo, x.hi)));
            }
            assert_eq!(was.effective_end.is_none(), now.effective_end.is_none());
            assert_eq!(
                (&was.kind, &was.value_text, &was.timepoint_id),
                (&now.kind, &now.value_text, &now.timepoint_id)
            );
        }
        for (was, now) in source.links.iter().zip(&derived.links) {
            for (kind, old, new) in
                [(&was.source_type, &was.source_id, &now.source_id), (&was.target_type, &was.target_id, &now.target_id)]
            {
                match kind.as_str() {
                    "event" | "document" => assert_eq!(*new, id(kind, old)),
                    _ => assert_eq!(new, old, "{kind} keeps the id the imaging half gives it"),
                }
            }
            assert_eq!(now.asserted_by_event_id, was.asserted_by_event_id.as_deref().map(|e| id("event", e)));
            assert_eq!(now.source_span, was.source_span);
        }
        for (was, now) in source.documents.iter().zip(&derived.documents) {
            assert_eq!((now.document_id.clone(), &now.text), (id("document", &was.document_id), &was.text));
        }
        // The re-minted records are what the profile's schema accepts.
        assert!(crate::clinical::schema::validate_records(&derived.to_json()).is_empty());
    }

    #[test]
    fn s10_a_relative_clock_keeps_its_times() {
        let source = worked_records();
        let (derived, _) = derive(source.clone(), 0, "pep", "s1").unwrap();
        assert_eq!(derived.descriptor.clock.reference, source.descriptor.clock.reference);
        assert_eq!(derived.descriptor.clock.origin_description, source.descriptor.clock.origin_description);
        let times = |r: &ClinicalRecords| r.events.iter().map(|e| (e.effective_start, e.available)).collect::<Vec<_>>();
        assert_eq!(times(&derived), times(&source));
    }

    #[test]
    fn s10_one_salt_mints_one_id_and_another_salt_another() {
        assert_eq!(remint("clock", "ehr", "pep", "s1"), remint("clock", "ehr", "pep", "s1"));
        assert_ne!(remint("clock", "ehr", "pep", "s1"), remint("clock", "ehr", "salt", "s1"));
        assert_ne!(remint("event", "x", "pep", "s1"), remint("record", "x", "pep", "s1"));
        let id = remint("document", "report-1", "pep", "s1");
        assert!(id.starts_with("pseudo.document.") && id.len() == "pseudo.document.".len() + 24);
    }

    #[test]
    fn s10_two_subjects_sharing_an_id_get_different_ones() {
        // Each subject may be shifted by its own amount: one clock id for
        // both would claim their times share a clock.
        assert_ne!(remint("clock", "ehr", "pep", "s1"), remint("clock", "ehr", "pep", "s2"));
        assert_ne!(remint("event", "lab-1", "pep", "s1"), remint("event", "lab-1", "pep", "s2"));
    }

    #[test]
    fn s10_a_shift_past_the_clock_is_refused() {
        let mut records = utc_records();
        records.events[0].available = Some(Bounds::exact(i64::MIN + 1));
        assert!(derive(records, -DAY, "pep", "s1").is_err());
        assert!(shift_us(i64::MAX / 2).is_err());
    }

    #[test]
    fn s6_names_and_locale_dates_are_found_inside_text() {
        assert_eq!(name_spans("Seen with Smith^Alice today"), vec![(10, 21)]);
        assert!(name_spans("a ^ b, 2^10").is_empty());
        assert_eq!(locale_spans("on 31.01.2020 and 12/31/2020, not 131.01.2020"), vec![(3, 13), (18, 28)]);
        let h = hits("MRN-42 on 2020-01-31, uid 1.2.840.10008.1", &["MRN-42".to_string()]);
        assert_eq!((h.ids.len(), h.dates.len(), h.uids.len()), (1, 1, 1));
    }

    #[test]
    fn s6_an_attestation_is_rekeyed_to_the_new_document_ids() {
        let digests = BTreeMap::from([
            ("clinical/documents/a".to_string(), "sha256:1".to_string()),
            ("clinical/events".to_string(), "sha256:2".to_string()),
        ]);
        let reminted = vec![
            Reminted { kind: "event", old: "a".into(), new: "pseudo.event.x".into() },
            Reminted { kind: "document", old: "a".into(), new: "pseudo.document.y".into() },
        ];
        let rekeyed = rekeyed(&digests, &reminted);
        assert_eq!(rekeyed.keys().collect::<Vec<_>>(), ["clinical/documents/pseudo.document.y", "clinical/events"]);
    }
}
