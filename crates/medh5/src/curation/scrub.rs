//! `medh5 scrub` --- finding identifiers in a container, and attesting to it.
//!
//! **What this cannot do, stated first, because the attestation depends on
//! it.**  Scrubbing inspects metadata.  It does not look at voxels, so it
//! cannot see burned-in text or a face reconstructible from a head CT; the
//! record it writes says exactly what was and was not checked (§11.4).
//!
//! **Where it looks: everywhere a string can be** --- every string in
//! `/meta`; every link's name, a second hard link's and a soft link's (and the
//! path it holds) included; every attribute, a committed datatype's included;
//! and every string dataset, including unknown ones.  A dataset whose type
//! holds strings it does not decode (inside a compound, array or
//! variable-length type) is reported, as is one outside the imaging groups it
//! does not read at all.  The scan and `--apply` are one traversal, so a
//! finding the scan calls actionable is, by construction, one the clean acts
//! on.  UIDs are **pseudonymised**, not deleted: a UID is how two files agree
//! on a frame of reference, so it becomes a stable hash of itself.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use indexmap::IndexMap;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::array::{DType, NdArray};
use crate::curation::identity::{ID_SOURCE, PSEUDONYM_SOURCE};
use crate::document::{SampleDocument, META_DATASET};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::{py_float, py_str, repr_list, repr_str};
use crate::sample::writer::amend;
use crate::sample::{frame_references, open_sample, Sample, FRAME_ATTRS};
use crate::{Error, Result};

pub const PROFILES: [&str; 2] = ["basic", "strict"];

/// Attributes to remove, from the DICOM PS3.15 E.1 basic profile, matched
/// case-, space- and underscore-insensitively.
///
/// Every table of keys is stored as [`normalise`] leaves a key, which is what
/// a lookup compares: an entry written `nhs_number` could never be matched,
/// so an NHS number, a date of birth and three UID spellings passed every scan.
pub const IDENTIFYING_KEYS: [&str; 76] = [
    "patientname",
    "patientid",
    "patientbirthdate",
    "patientbirthtime",
    "patientaddress",
    "patienttelephonenumbers",
    "patienttelecominformation",
    "otherpatientids",
    "otherpatientidssequence",
    "otherpatientnames",
    "patientmothersbirthname",
    "patientinsuranceplancodesequence",
    "patientreligiouspreference",
    "patientinstitutionresidence",
    "currentpatientlocation",
    "countryofresidence",
    "regionofresidence",
    "militaryrank",
    "branchofservice",
    "occupation",
    "medicalrecordlocator",
    "issuerofpatientid",
    "mrn",
    "nhsnumber",
    "ssn",
    "patientcomments",
    "additionalpatienthistory",
    "medicalalerts",
    "allergies",
    "admittingdiagnosesdescription",
    "specialneeds",
    "referringphysicianname",
    "referringphysicianaddress",
    "referringphysiciantelephonenumbers",
    "consultingphysicianname",
    "performingphysicianname",
    "operatorsname",
    "physiciansofrecord",
    "physicianreadingstudy",
    "namesofintendedrecipientsofresults",
    "requestingphysician",
    "responsibleperson",
    "verifyingobservername",
    "contentcreatorname",
    "personname",
    "reviewername",
    "scheduledperformingphysicianname",
    "accessionnumber",
    "studyid",
    "admissionid",
    "issuerofadmissionid",
    "performedprocedurestepid",
    "performedprocedurestepdescription",
    "requestedprocedureid",
    "scheduledprocedurestepid",
    "scheduledstudylocation",
    "requestattributessequence",
    "institutionname",
    "institutionaddress",
    "institutionaldepartmentname",
    "institutioncodesequence",
    "stationname",
    "deviceserialnumber",
    "plateid",
    "cassetteid",
    "detectorid",
    "gantryid",
    "generatorid",
    "clinicaltrialsubjectid",
    "clinicaltrialsubjectreadingid",
    "clinicaltrialsitename",
    "clinicaltrialsiteid",
    "clinicaltrialsponsorname",
    "clinicaltrialprotocolid",
    "clinicaltrialprotocolname",
    "clinicaltrialtimepointid",
];

/// Attributes that identify *in combination* --- and that some pipelines need
/// (`PatientWeight` drives a PET SUV).  Reported under `basic`, removed under
/// `strict`.
pub const QUASI_IDENTIFYING_KEYS: [&str; 12] = [
    "patientage",
    "patientweight",
    "patientsize",
    "patientsex",
    "patientsexneutered",
    "ethnicgroup",
    "patientspeciesdescription",
    "patientstate",
    "patientbreeddescription",
    "pregnancystatus",
    "smokingstatus",
    "lastmenstrualdate",
];

/// Keys that hold a date.  A value that is wholly a date is shifted under any
/// key; these are shifted (or removed) whatever their value looks like.
pub const DATE_KEYS: [&str; 14] = [
    "studydate",
    "seriesdate",
    "acquisitiondate",
    "acquisitiondatetime",
    "contentdate",
    "instancecreationdate",
    "patientbirthdate",
    "admittingdate",
    "scheduledproceduredate",
    "scheduledprocedurestepstartdate",
    "performedprocedurestepstartdate",
    "lastmenstrualdate",
    "birthdate",
    "dateofbirth",
];

/// Keys that hold a UID: one whose value is not UID-shaped is reported for a
/// person, because a stable pseudonym needs something recognisable to hash.
pub const UID_KEYS: [&str; 12] = [
    "studyinstanceuid",
    "seriesinstanceuid",
    "sopinstanceuid",
    "frameofreferenceuid",
    "mediastoragesopinstanceuid",
    "referencedsopinstanceuid",
    "irradiationeventuid",
    "concatenationuid",
    "storagemediafilesetuid",
    "studyuid",
    "seriesuid",
    "frameuid",
];

/// How deep the walk goes before it reports rather than descends.
pub const MAX_DEPTH: usize = 8;
/// What [`pseudonymise`] produces, and therefore what the rules skip: a rule
/// that fired on its own output would make the tool non-idempotent.
pub const PSEUDONYM_PREFIX: &str = "pseudo:";
/// What `--profile strict` leaves where a filesystem path was.
pub const PATH_REMOVED: &str = "<path removed>";
/// HIPAA Safe Harbor aggregates every age over 89 as "90 or older".
pub const AGE_LIMIT: f64 = 90.0;
/// Provenance references to objects in this file, which are not paths.
pub const INTERNAL_REFERENCES: [&str; 5] = ["annotations/", "images/", "grids/", "transforms/", "index/"];
/// Characters above which a string cannot be reviewed by a rule, only a person.
pub const FREE_TEXT: usize = 200;
/// Findings `--apply` never acts on by itself: every other record refers to
/// the sample by these ids.
pub const UNFIXABLE_LOCATIONS: [&str; 2] = ["identity.sample_id", "identity.subject_id"];
/// Rules that only `--profile strict` acts on.
pub const STRICT_RULES: [&str; 6] =
    ["free_text", "staff_name", "person_name", "quasi_identifier", "age", "organization_name"];
/// Findings about what the sample is *called*.
pub const IDENTITY_RULES: [&str; 2] = ["identity", "file_name"];

const NAME_REVIEW: &str = "an identifier other records refer to, so --apply leaves it for a person";
const VOCABULARY_REVIEW: &str =
    "part of a label vocabulary shared across the cohort and pinned by its digest, so --apply leaves it for a person";
const UID_DETAIL: &str = "a real DICOM UID; SHOULD be a pseudonym (§11.4)";
const WHOLE_DATE_DETAIL: &str = "a value that is wholly a date: a date of the subject's, whatever key holds it";
const COPY_DETAIL: &str = "a copy of the sample document outside /meta, holding its identity: --apply pseudonymises \
                           /meta, and a copy keeps what that replaces --- --apply removes it";
const ALIAS_DETAIL: &str = "another name for /meta, the sample document: a tool that rewrites /meta can leave the old \
                            document here, with every identifier it replaced --- --apply removes the name";

/// Root attributes `commit` rewrites from the amended state.
const ROOT_MANAGED_ATTRS: [&str; 7] =
    ["medh5_version", "medh5_kind", "medh5_profiles", "created", "generator", "digest_algo", "content_id"];

/// Groups whose non-string datasets are what the format puts there --- pixel
/// data, geometry, transforms, the derived index --- or that a finding of
/// their own covers (the clinical profile).  Elsewhere a byte buffer or a
/// type this tool cannot decode may hold text, so it is reported, not passed.
const DEFINED_GROUPS: [&str; 6] = ["images", "annotations", "grids", "transforms", "index", "clinical"];

/// How many of the non-string datasets outside [`DEFINED_GROUPS`] the
/// `not_checked` note names; the rest are counted.
const UNEXAMINED_NAMED: usize = 10;

/// A replaced id shorter than this is not looked for elsewhere: `1` or `CT`
/// would be found in every number and every modality.
const MIN_SWEPT_ID: usize = 4;

/// How far apart, in days, the baselines two timepoints' dates imply may be
/// and still agree: a date without its time of day is up to a day off the
/// interval `days_from_baseline` states, which may itself be rounded.
const BASELINE_TOLERANCE: f64 = 1.5;

/// Attributes whose value is the id of another object, timepoint, activity
/// or record: reviewed like the ids they name and never rewritten.
const REFERENCE_ATTRS: [&str; 18] = [
    "grid",
    "grid_levels",
    "timepoint",
    "timepoints",
    "prov",
    "quality",
    "metrics",
    "derived_from",
    "ignore_mask",
    "valid_mask",
    "skeleton",
    "correspondence",
    "from_grid",
    "to_grid",
    "field_grid",
    "cp_grid",
    "inverse_id",
    "components",
];

fn regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("a valid pattern"))
}

/// One calendar date found in a string: where it is, and the day it names.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Dated {
    start: usize,
    end: usize,
    year: i64,
    month: u32,
    day: u32,
}

/// The three numbers of a date the patterns matched in ASCII digits.
fn ymd(year: &str, month: &str, day: &str) -> Option<(i64, u32, u32)> {
    Some((year.parse().ok()?, month.parse().ok()?, day.parse().ok()?))
}

/// Every ISO 8601 calendar date (`YYYY-MM-DD`) in `text`, alone or opening a
/// date-time (`2020-01-31T10:22:00`).
///
/// Digits are ASCII, as ISO 8601 writes them; a match must not continue a
/// longer run of digits and must name a real day.  A word boundary was not
/// enough: a date-time has none after its day (`1` and `T` are both word
/// characters), nor does a date in `CT_2020-01-31.nii`, so neither was found
/// or shifted --- while the attestation recorded the shift.
fn iso_dates(text: &str) -> Vec<Dated> {
    static CELL: OnceLock<Regex> = OnceLock::new();
    let digit_before = |at: usize| text[..at].chars().next_back().is_some_and(|c| c.is_ascii_digit());
    let digit_after = |at: usize| text[at..].chars().next().is_some_and(|c| c.is_ascii_digit());
    regex(&CELL, r"([0-9]{4})-([0-9]{2})-([0-9]{2})")
        .captures_iter(text)
        .filter_map(|c| {
            let whole = c.get(0)?;
            if digit_before(whole.start()) || digit_after(whole.end()) {
                return None;
            }
            let (year, month, day) = ymd(&c[1], &c[2], &c[3])?;
            valid_date(year, month, day).then_some(Dated { start: whole.start(), end: whole.end(), year, month, day })
        })
        .collect()
}

/// Python's `re.match` of a `^...$` pattern: `$` also matches before one
/// trailing newline.
fn anchored(re: &Regex, text: &str) -> bool {
    re.is_match(text) || text.strip_suffix('\n').is_some_and(|t| re.is_match(t))
}

/// A whole value that is a DICOM date (DA, `YYYYMMDD`) or date-time (DT,
/// `YYYYMMDDHHMMSS.FFFFFF&ZZXX`, every part after the day optional), or
/// ISO 8601's basic form of one (`20200131T102200`): the date, and the byte
/// where what follows it starts.  ASCII digits only, and a real day --- an
/// eight-digit ontology code (SNOMED CT `52988006`) is not a date.
fn dicom_date(text: &str) -> Option<Dated> {
    static CELL: OnceLock<Regex> = OnceLock::new();
    let pattern = regex(
        &CELL,
        r"^([0-9]{4})([0-9]{2})([0-9]{2})(?:[Tt]?[0-9]{2}(?:[0-9]{2}(?:[0-9]{2}(?:\.[0-9]{1,6})?)?)?)?(?:[+-][0-9]{4})?$",
    );
    // `$` as Python's `re.match` reads it: before one trailing newline too.
    let c = pattern.captures(text.strip_suffix('\n').unwrap_or(text))?;
    let (year, month, day) = ymd(&c[1], &c[2], &c[3])?;
    valid_date(year, month, day).then_some(Dated { start: 0, end: c.get(3)?.end(), year, month, day })
}

fn is_dicom_date(text: &str) -> bool {
    dicom_date(text).is_some()
}

/// A whole value that is one ISO 8601 date or date-time: `2020-01-31`,
/// `2020-01-31T10:22:00.5+01:00`, or a space for the `T` as Python's `str()`
/// of a `datetime` writes it.  ASCII digits, and a real day.
fn is_iso_date(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    let pattern = regex(
        &CELL,
        r"^([0-9]{4})-([0-9]{2})-([0-9]{2})(?:[Tt ][0-9]{2}(?::[0-9]{2}(?::[0-9]{2}(?:[.,][0-9]+)?)?)?(?:[Zz]|[+-][0-9]{2}(?::?[0-9]{2})?)?)?$",
    );
    let Some(c) = pattern.captures(text.strip_suffix('\n').unwrap_or(text)) else { return false };
    ymd(&c[1], &c[2], &c[3]).is_some_and(|(y, m, d)| valid_date(y, m, d))
}

/// A value that is wholly a date or a date-time, in a form [`shift`] moves:
/// whatever key holds it, it is a date of the subject's, not text about one.
fn is_whole_date(text: &str) -> bool {
    let text = text.trim();
    is_dicom_date(text) || is_iso_date(text)
}

/// The day a whole date or date-time names, as days since 1970-01-01.
fn day_number(text: &str) -> Option<i64> {
    let text = text.trim();
    let date = match dicom_date(text) {
        Some(date) => date,
        None if is_iso_date(text) => *iso_dates(text).first()?,
        None => return None,
    };
    Some(days_from_civil(date.year, date.month, date.day))
}

/// What a string holding a copy of a sample document names the sample: its
/// `identity`'s subject or sample id, when the string decodes to a JSON
/// object with an `identity` that has one.
fn document_copy(text: &str) -> Option<String> {
    let Value::Object(doc) = embedded_json(text)? else { return None };
    let identity = doc.get("identity")?.as_object()?;
    ["subject_id", "sample_id"].iter().find_map(|key| identity.get(*key)).map(py_str)
}

/// Where `id` occurs in `text` as a token of its own --- not continuing a run
/// of letters or digits on either side --- so `MRN-42` is found in
/// `case MRN-42.nii` and not in `MRN-421`.
fn id_tokens(text: &str, id: &str) -> Vec<(usize, usize)> {
    let alnum_before = |at: usize| text[..at].chars().next_back().is_some_and(char::is_alphanumeric);
    let alnum_after = |at: usize| text[at..].chars().next().is_some_and(char::is_alphanumeric);
    text.match_indices(id)
        .map(|(at, _)| (at, at + id.len()))
        .filter(|(start, end)| !alnum_before(*start) && !alnum_after(*end))
        .collect()
}

/// A dotted-numeric OID.  `pseudo:...` and UUIDs do not match, by design.
fn is_dicom_uid(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    anchored(regex(&CELL, r"^[0-9]+(\.[0-9]+){3,}$"), text)
}

fn has_date(text: &str) -> bool {
    !iso_dates(text).is_empty() || is_dicom_date(text)
}

/// DICOM PN form `Family^Given`, in any script.
fn is_person_name(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex(&CELL, r"^[\p{L}\p{Nl}\p{No}][^\^=\p{Nd}]*\^[\p{L}\p{Nl}\p{No}]").is_match(text)
}

/// Dotted-numeric UIDs *inside* a longer string: `(?<![0-9.])[0-9]+(?:\.[0-9]+){3,}(?![0-9.])`.
///
/// The look-arounds pin a match to a whole run of digits and dots, so a run
/// matches exactly when all of it is UID-shaped.
fn uid_tokens(text: &str) -> Vec<(usize, usize)> {
    static RUN: OnceLock<Regex> = OnceLock::new();
    static WHOLE: OnceLock<Regex> = OnceLock::new();
    let whole = regex(&WHOLE, r"^[0-9]+(?:\.[0-9]+){3,}$");
    regex(&RUN, r"[0-9.]+")
        .find_iter(text)
        .filter(|m| whole.is_match(m.as_str()))
        .map(|m| (m.start(), m.end()))
        .collect()
}

fn has_uid_token(text: &str) -> bool {
    !uid_tokens(text).is_empty()
}

/// `("scheme:", rest)` for a provenance reference, or `("", value)`.
fn split_reference(value: &str) -> (String, String) {
    static CELL: OnceLock<Regex> = OnceLock::new();
    match regex(&CELL, r"^([A-Za-z][A-Za-z0-9+.\-]+:)").find(value) {
        Some(m) => (m.as_str().to_string(), value[m.end()..].to_string()),
        None => (String::new(), value.to_string()),
    }
}

fn is_path(rest: &str) -> bool {
    (rest.contains('/') || rest.contains('\\')) && !INTERNAL_REFERENCES.iter().any(|p| rest.starts_with(p))
}

fn basename(rest: &str) -> String {
    let trimmed = rest.trim_end_matches(['\\', '/']);
    trimmed.rsplit(['\\', '/']).next().unwrap_or("").to_string()
}

fn normalise(key: &str) -> String {
    key.replace(['_', ' '], "").to_lowercase()
}

fn preview(text: &str) -> String {
    if text.chars().count() <= 60 {
        text.to_string()
    } else {
        format!("{}...", text.chars().take(57).collect::<String>())
    }
}

/// A stable pseudonym for a UID: same input, same output, everywhere.
pub fn pseudonymise(uid: &str, salt: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = hex::encode(Sha256::digest(format!("{salt}{uid}").as_bytes()));
    format!("{PSEUDONYM_PREFIX}{}", &digest[..32])
}

/// One place an identifier may live.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    pub rule: String,
    pub r#where: String,
    pub detail: String,
    pub value: Option<String>,
    pub actionable: bool,
    /// Whether `--apply` may ever act here.
    pub fixable: bool,
}

impl Finding {
    pub fn to_json(&self) -> Value {
        json!({
            "rule": self.rule,
            "where": self.r#where,
            "detail": self.detail,
            "value": self.value,
            "actionable": self.actionable,
        })
    }
}

impl std::fmt::Display for Finding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mark = if self.actionable { "*" } else { " " };
        let shown = self.value.as_ref().map(|v| format!(" = {}", repr_str(v))).unwrap_or_default();
        write!(f, "{mark} {:<14} {}{shown}\n    {}", self.rule, self.r#where, self.detail)
    }
}

fn names_the_sample(finding: &Finding) -> bool {
    IDENTITY_RULES.contains(&finding.rule.as_str()) || UNFIXABLE_LOCATIONS.contains(&finding.r#where.as_str())
}

/// What was found, what was changed, and what was never looked at.
#[derive(Debug, Clone, PartialEq)]
pub struct ScrubReport {
    pub path: String,
    pub profile: String,
    pub findings: Vec<Finding>,
    pub actions: Vec<String>,
    pub applied: bool,
    /// What a re-scan of the *written* file still finds (`apply` only).
    pub remaining: Vec<Finding>,
    /// Every original value this run replaced with a pseudonym: the key to
    /// the pseudonyms, to be kept with the salt and not with the data.
    pub uid_map: IndexMap<String, String>,
    pub not_checked: Vec<String>,
}

impl ScrubReport {
    pub fn new(path: &str, profile: &str) -> ScrubReport {
        ScrubReport {
            path: path.into(),
            profile: profile.into(),
            findings: Vec::new(),
            actions: Vec::new(),
            applied: false,
            remaining: Vec::new(),
            uid_map: IndexMap::new(),
            not_checked: vec![
                "pixel data (burned-in text, identifiable anatomy)".into(),
                "free text whose meaning this tool cannot judge".into(),
            ],
        }
    }

    pub fn add(
        &mut self,
        rule: &str,
        r#where: &str,
        detail: &str,
        value: Option<&str>,
        actionable: bool,
        fixable: bool,
    ) {
        self.findings.push(Finding {
            rule: rule.into(),
            r#where: r#where.into(),
            detail: detail.into(),
            value: value.map(preview),
            actionable: actionable && fixable,
            fixable,
        });
    }

    pub fn actionable(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| f.actionable).collect()
    }

    pub fn needs_review(&self) -> Vec<&Finding> {
        self.findings.iter().filter(|f| !f.actionable).collect()
    }

    pub fn clean(&self) -> bool {
        self.findings.is_empty()
    }

    /// Actionable findings a re-scan of the written file still reports.
    pub fn remaining_actionable(&self) -> Vec<&Finding> {
        self.remaining.iter().filter(|f| f.actionable).collect()
    }

    /// Findings on what the sample is *called*, still open after this run.
    pub fn open_identity(&self) -> Vec<&Finding> {
        let left = if self.applied { &self.remaining } else { &self.findings };
        left.iter().filter(|f| names_the_sample(f)).collect()
    }

    /// Whether this run leaves nothing a further `--apply` could fix.
    pub fn ok(&self) -> bool {
        if !self.applied {
            return self.clean();
        }
        if !self.remaining_actionable().is_empty() {
            return false;
        }
        !(self.profile == "strict" && !self.open_identity().is_empty())
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "profile": self.profile,
            "applied": self.applied,
            "clean": self.clean(),
            "ok": self.ok(),
            "findings": self.findings.iter().map(Finding::to_json).collect::<Vec<_>>(),
            "actions": self.actions,
            "remaining": self.remaining.iter().map(Finding::to_json).collect::<Vec<_>>(),
            "uid_map": self.uid_map,
            "not_checked": self.not_checked,
        })
    }

    pub fn format(&self) -> String {
        let mut lines = vec![format!(
            "{}: {} finding(s) ({} actionable, {} for review)",
            self.path,
            self.findings.len(),
            self.actionable().len(),
            self.needs_review().len()
        )];
        lines.extend(self.findings.iter().map(|f| f.to_string()));
        if !self.actions.is_empty() {
            lines.push(format!("  applied: {} change(s)", self.actions.len()));
        }
        if self.applied {
            let left = self.remaining_actionable();
            lines.push(format!(
                "  re-scanned after applying: {} finding(s) remain, {} actionable",
                self.remaining.len(),
                left.len()
            ));
            lines.extend(left.iter().map(|f| format!("  REMAINS {}", f.r#where)));
            if self.profile == "strict" {
                lines.extend(self.open_identity().iter().map(|f| {
                    format!(
                        "  IDENTITY {}: re-mint it, or re-run --apply with --pseudonymise-ids --salt ...",
                        f.r#where
                    )
                }));
            }
        }
        lines.push(format!("  NOT checked: {}", self.not_checked.join("; ")));
        lines.join("\n")
    }
}

/// A value in a slot the sweep visits: JSON, or an array attribute that the
/// walk treats as one opaque value (NumPy prints it as `text`).
enum Slot<'a> {
    Json(&'a Value),
    Opaque(&'a str),
}

impl Slot<'_> {
    fn text(&self) -> String {
        match self {
            Slot::Json(v) => py_str(v),
            Slot::Opaque(t) => t.to_string(),
        }
    }
}

/// What a rule leaves in a slot.
enum Kept {
    Same,
    Value(Value),
    Drop,
}

/// One sample id `--pseudonymise-ids` replaces: the old value, its pseudonym,
/// and which id it was.
#[derive(Debug, Clone)]
struct Replaced {
    old: String,
    pseudonym: String,
    name: String,
}

/// One walk over every string in a sample, for scanning and for cleaning.
struct Sweep<'r> {
    report: &'r mut ScrubReport,
    writing: bool,
    strict: bool,
    salt: String,
    date_shift_days: Option<i64>,
    /// The shift the sample's record states, when it states one that moved
    /// anything: a recorded 0 shifted nothing, so it silences nothing.
    recorded_shift: Option<i64>,
    dates_shifted: bool,
    /// Timepoint dates, by position, that disagree with the recorded shift.
    disagreeing: BTreeMap<usize, String>,
    /// The ids this run replaces, looked for in every other string.
    replaced: Vec<Replaced>,
    /// Non-string datasets outside [`DEFINED_GROUPS`], none of them read.
    unexamined: Vec<String>,
    uid_map: IndexMap<String, String>,
    actions: Vec<String>,
}

impl<'r> Sweep<'r> {
    fn new(report: &'r mut ScrubReport) -> Sweep<'r> {
        Sweep {
            report,
            writing: false,
            strict: false,
            salt: String::new(),
            date_shift_days: None,
            recorded_shift: None,
            dates_shifted: false,
            disagreeing: BTreeMap::new(),
            replaced: Vec::new(),
            unexamined: Vec::new(),
            uid_map: IndexMap::new(),
            actions: Vec::new(),
        }
    }

    fn find(
        &mut self,
        rule: &str,
        r#where: &str,
        detail: &str,
        value: Option<&str>,
        actionable: bool,
        review: Option<&str>,
    ) {
        match review {
            Some(why) => self.report.add(rule, r#where, &format!("{detail} --- {why}"), value, false, false),
            None => self.report.add(rule, r#where, detail, value, actionable, true),
        }
    }

    fn acts(&self, review: Option<&str>) -> bool {
        self.writing && review.is_none()
    }

    fn pseudonym(&mut self, uid: &str) -> String {
        let replacement = pseudonymise(uid, &self.salt);
        if self.writing {
            self.uid_map.insert(uid.to_string(), replacement.clone());
        }
        replacement
    }

    /// Walk arbitrary JSON, applying the key and value rules as it goes.
    fn json(&mut self, payload: &Value, r#where: &str, depth: usize, review: Option<&str>) -> Value {
        if depth > MAX_DEPTH {
            self.find(
                "too_deep",
                r#where,
                &format!("nested more than {MAX_DEPTH} levels; this tool did not inspect it, and a person must"),
                None,
                false,
                None,
            );
            if self.writing {
                self.actions.push(format!("{} left untouched: nested deeper than {MAX_DEPTH}", r#where));
            }
            return payload.clone();
        }
        match payload {
            Value::Object(map) => {
                let mut out = Map::new();
                for (key, value) in map {
                    let path = format!("{}.{key}", r#where);
                    let kept = match self.entry(key, Slot::Json(value), &path, depth, review) {
                        Kept::Drop => continue,
                        Kept::Same => value.clone(),
                        Kept::Value(v) => v,
                    };
                    let Some(name) = self.key(key, &path, review) else { continue };
                    out.insert(name, kept);
                }
                Value::Object(out)
            }
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .enumerate()
                    .map(|(i, v)| self.json(v, &format!("{}[{i}]", r#where), depth + 1, review))
                    .collect(),
            ),
            Value::String(s) => Value::String(self.text(s, r#where, depth, review, "")),
            other => other.clone(),
        }
    }

    /// One mapping entry: the rules keyed on its name, then its value.
    fn entry(&mut self, key: &str, value: Slot, path: &str, depth: usize, review: Option<&str>) -> Kept {
        if let Some(kept) = self.keyed(key, &value, path, review) {
            return kept;
        }
        match value {
            Slot::Json(v) => self.walked(v, path, depth + 1, review),
            Slot::Opaque(_) => Kept::Same,
        }
    }

    /// `value`, walked, as what an entry keeps.
    fn walked(&mut self, value: &Value, path: &str, depth: usize, review: Option<&str>) -> Kept {
        let walked = self.json(value, path, depth, review);
        if &walked == value {
            Kept::Same
        } else {
            Kept::Value(walked)
        }
    }

    /// The rules an entry's *name* decides --- an identifying keyword, a UID
    /// or quasi-identifying key, a date key --- and a value that is wholly a
    /// date; `None` when none applies and the value is the walk's.
    fn keyed(&mut self, key: &str, value: &Slot, path: &str, review: Option<&str>) -> Option<Kept> {
        let normal = normalise(key);
        let text = value.text();
        if IDENTIFYING_KEYS.contains(&normal.as_str()) {
            self.find(
                "identifier",
                path,
                "an identifying DICOM attribute; writers MUST NOT copy tags wholesale (§11.4)",
                Some(&text),
                true,
                review,
            );
            if self.acts(review) {
                self.actions.push(format!("{path} removed"));
                return Some(Kept::Drop);
            }
            return Some(Kept::Same);
        }
        // A pseudonym is what this rule asks for --- and the tool's own
        // output, which reported here kept every scrubbed file's gate red.
        if UID_KEYS.contains(&normal.as_str()) && !is_dicom_uid(&text) && !text.starts_with(PSEUDONYM_PREFIX) {
            self.find(
                "uid",
                path,
                "a UID attribute whose value is not a UID; it cannot be pseudonymised safely, so a person must decide",
                Some(&text),
                false,
                review,
            );
            return Some(Kept::Same);
        }
        if QUASI_IDENTIFYING_KEYS.contains(&normal.as_str()) {
            self.find(
                "quasi_identifier",
                path,
                "identifying in combination with others, and sometimes needed (PatientWeight drives a PET SUV); \
                 removed under --profile strict, your decision under basic",
                Some(&text),
                false,
                review,
            );
            if self.strict && self.acts(review) {
                self.actions.push(format!("{path} removed (quasi-identifier, strict)"));
                return Some(Kept::Drop);
            }
            return Some(Kept::Same);
        }
        let keyed = DATE_KEYS.contains(&normal.as_str());
        if keyed || matches!(value, Slot::Json(Value::String(s)) if is_whole_date(s)) {
            let detail = if keyed { "a date attribute copied from the source" } else { WHOLE_DATE_DETAIL };
            return Some(self.date(&text, path, detail, review));
        }
        None
    }

    /// A mapping *key* that is itself a name or a UID; `None` drops the entry.
    fn key(&mut self, key: &str, path: &str, review: Option<&str>) -> Option<String> {
        if key.starts_with(PSEUDONYM_PREFIX) {
            return Some(key.to_string());
        }
        if is_person_name(key) {
            self.find("person_name", path, "a key that reads as a DICOM person name", Some(key), true, review);
            if self.acts(review) {
                self.actions.push(format!("{path} removed (person name)"));
                return None;
            }
            return Some(key.to_string());
        }
        if is_dicom_uid(key) {
            self.find("uid", path, &format!("a key that is {UID_DETAIL}"), Some(key), true, review);
            let pseudonym = self.pseudonym(key);
            if self.acts(review) {
                self.actions.push(format!("{path} key pseudonymised"));
                return Some(pseudonym);
            }
            return Some(key.to_string());
        }
        if has_date(key) {
            self.find("date", path, "a key that looks like a date", Some(key), false, review);
        }
        Some(self.replaced_ids(key, path, review))
    }

    /// A date a rule acts on: reported, then shifted --- or removed, with no
    /// shift to make or none it can make.  A file that records a shift has
    /// had its dates handled, so there it is reported for a person and left.
    fn date(&mut self, text: &str, path: &str, detail: &str, review: Option<&str>) -> Kept {
        self.find("date", path, detail, Some(text), !self.dates_shifted, review);
        if self.dates_shifted || !self.acts(review) {
            return Kept::Same;
        }
        match shift(text, self.date_shift_days) {
            None => {
                self.actions.push(format!("{path} removed"));
                Kept::Drop
            }
            Some(moved) => {
                self.actions.push(format!("{path} shifted"));
                Kept::Value(Value::String(moved))
            }
        }
    }

    /// `text` with every UID inside it replaced by its pseudonym.
    fn swap_uids(&mut self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for (start, end) in uid_tokens(text) {
            out.push_str(&text[last..start]);
            out.push_str(&self.pseudonym(&text[start..end]));
            last = end;
        }
        out.push_str(&text[last..]);
        out
    }

    /// The ids this run replaces, wherever else a string holds one: a record
    /// number in a label or a description outlives the pseudonym that
    /// replaced it in `identity` (§11.4), and a later scan cannot tell it is
    /// one.  In an id, which nothing rewrites, it is reported for a person.
    fn replaced_ids(&mut self, value: &str, r#where: &str, review: Option<&str>) -> String {
        let mut out = value.to_string();
        for Replaced { old, pseudonym, name } in self.replaced.clone() {
            let found = id_tokens(&out, &old);
            if found.is_empty() {
                continue;
            }
            self.find(
                "identity",
                r#where,
                &format!("holds the {name} --pseudonymise-ids replaces, which would outlive its pseudonym here"),
                Some(value),
                true,
                review,
            );
            if self.acts(review) {
                for (start, end) in found.into_iter().rev() {
                    out.replace_range(start..end, &pseudonym);
                }
                self.uid_map.insert(old, pseudonym);
                self.actions.push(format!("{}: the {name} pseudonymised", r#where));
            }
        }
        out
    }

    /// The value rules, for one string.  `removed` is what a removal leaves.
    fn text(&mut self, value: &str, r#where: &str, depth: usize, review: Option<&str>, removed: &str) -> String {
        if let Some(parsed) = embedded_json(value) {
            let cleaned = self.json(&parsed, r#where, depth + 1, review);
            if self.acts(review) && cleaned != parsed {
                return crate::json::dumps(&cleaned, crate::json::Style::PYTHON.sorted());
            }
            return value.to_string();
        }
        if value.starts_with(PSEUDONYM_PREFIX) {
            return value.to_string();
        }
        if is_person_name(value) {
            self.find("person_name", r#where, "reads as a DICOM person name", Some(value), true, review);
            if self.acts(review) {
                self.actions.push(format!("{} removed (person name)", r#where));
                return removed.to_string();
            }
            return value.to_string();
        }
        if is_dicom_uid(value) {
            self.find("uid", r#where, UID_DETAIL, Some(value), true, review);
            let pseudonym = self.pseudonym(value);
            if self.acts(review) {
                self.actions.push(format!("{} pseudonymised", r#where));
                return pseudonym;
            }
            return value.to_string();
        }
        if is_whole_date(value) {
            return match self.date(value, r#where, WHOLE_DATE_DETAIL, review) {
                Kept::Value(Value::String(moved)) => moved,
                Kept::Drop => removed.to_string(),
                _ => value.to_string(),
            };
        }
        let mut value = value.to_string();
        if has_uid_token(&value) {
            self.find(
                "uid",
                r#where,
                "contains a real DICOM UID; SHOULD be a pseudonym (§11.4), and left in text it maps the pseudonym \
                 straight back",
                Some(&value),
                true,
                review,
            );
            if self.acts(review) {
                value = self.swap_uids(&value);
                self.actions.push(format!("{}: UID pseudonymised", r#where));
            }
        }
        let value = self.replaced_ids(&value, r#where, review);
        if has_date(&value) {
            self.find("date", r#where, "contains what looks like a date", Some(&value), false, review);
            return value;
        }
        let length = value.chars().count();
        if length > FREE_TEXT {
            self.find(
                "free_text",
                r#where,
                &format!("{length} characters of free text --- no rule can judge this; a person must"),
                None,
                false,
                review,
            );
            if self.strict && self.acts(review) {
                self.actions.push(format!("{} removed (free text, strict)", r#where));
                return removed.to_string();
            }
        }
        value
    }

    /// An id something else refers to: every rule, and never a rewrite.
    fn name(&mut self, value: &Value, r#where: &str) {
        if let Value::String(s) = value {
            self.text(s, r#where, 0, Some(NAME_REVIEW), "");
        }
    }

    /// One provenance `inputs`/`outputs` entry: UIDs, paths, then text.
    fn reference(&mut self, value: &str, r#where: &str) -> String {
        let (_, rest) = split_reference(value);
        let mut cleaned = value.to_string();
        let has_uid = has_uid_token(value);
        // The rules after this one read the reference with its UIDs already
        // pseudonyms, so a UID is reported here and only here.
        let swapped = if has_uid { self.swap_uids(value) } else { value.to_string() };
        if has_uid {
            self.find(
                "uid",
                r#where,
                "a real DICOM UID inside a provenance reference; the same UID is pseudonymised where it is a field, \
                 and left here it maps the pseudonym straight back (§11.4)",
                Some(value),
                true,
                None,
            );
            if self.writing {
                cleaned = swapped.clone();
                self.actions.push(format!("{}: UID pseudonymised", r#where));
            }
        }
        if is_path(&rest) {
            self.find(
                "path",
                r#where,
                "a filesystem path; export directories routinely name the patient or the site. --apply keeps only the \
                 file name, and --profile strict removes the path",
                Some(value),
                true,
                None,
            );
            let (scheme, rest) = split_reference(&swapped);
            // A name in the file name is still a name.
            let base = self.text(&basename(&rest), r#where, 0, None, "");
            if self.writing {
                cleaned = format!("{scheme}{}", if self.strict { PATH_REMOVED.to_string() } else { base });
                self.actions.push(format!("{}: path {}", r#where, if self.strict { "removed" } else { "reduced" }));
            }
        } else if !rest.is_empty() {
            let (scheme, rest) = split_reference(&swapped);
            cleaned = format!("{scheme}{}", self.text(&rest, r#where, 0, None, ""));
        }
        cleaned
    }

    // -- the sample document ---------------------------------------------------

    /// Every string in `/meta`; the cleaned document when writing.
    fn document(&mut self, doc: &Map<String, Value>) -> Map<String, Value> {
        let record = doc.get("deidentification").and_then(Value::as_object);
        self.recorded_shift = recorded_shift(record.and_then(|r| r.get("date_shift_days")));
        self.dates_shifted = self.recorded_shift.is_some();
        self.disagreeing = self.recorded_shift.map(|days| disagreeing(doc, days)).unwrap_or_default();
        let mut out = doc.clone();
        let identity = doc.get("identity").and_then(Value::as_object).cloned().unwrap_or_default();
        out.insert("identity".into(), Value::Object(self.identity(&identity, doc)));
        if let Some(cohort) = doc.get("cohort") {
            out.insert("cohort".into(), self.json(cohort, "cohort", 0, None));
        }
        let timepoints: Vec<Value> = doc
            .get("timepoints")
            .and_then(Value::as_array)
            .map(|tps| tps.iter().enumerate().map(|(i, tp)| self.timepoint(tp, i)).collect())
            .unwrap_or_default();
        out.insert("timepoints".into(), Value::Array(timepoints));
        for section in ["acquisition", "extra"] {
            if let Some(Value::Object(entries)) = doc.get(section) {
                let mut cleaned = Map::new();
                for (key, value) in entries {
                    let path = format!("{section}.{key}");
                    // A top-level key is a key like any other: `extra` holding
                    // `PatientID` scanned clean while one level down it did not.
                    let kept = match self.keyed(key, &Slot::Json(value), &path, None) {
                        Some(kept) => kept,
                        None => self.walked(value, &path, 0, None),
                    };
                    let name = match (section, &kept) {
                        (_, Kept::Drop) => None,
                        // Keyed by image id: an id, reviewed and never renamed.
                        ("acquisition", _) => {
                            self.name(&Value::String(key.clone()), &path);
                            Some(key.clone())
                        }
                        _ => self.key(key, &path, None),
                    };
                    match (kept, name) {
                        (Kept::Drop, _) | (_, None) => {}
                        (Kept::Value(v), Some(name)) => {
                            cleaned.insert(name, v);
                        }
                        (Kept::Same, Some(name)) => {
                            cleaned.insert(name, value.clone());
                        }
                    }
                }
                out.insert(section.into(), Value::Object(cleaned));
            }
        }
        if let Some(Value::Object(prov)) = doc.get("provenance") {
            out.insert("provenance".into(), Value::Object(self.provenance(prov)));
        }
        if let Some(Value::Array(claims)) = doc.get("splits") {
            let cleaned: Vec<Value> = claims.iter().enumerate().map(|(i, c)| self.split(c, i)).collect();
            out.insert("splits".into(), Value::Array(cleaned));
        }
        if let Some(Value::Object(records)) = doc.get("quality") {
            let mut cleaned = Map::new();
            for (key, record) in records {
                let path = format!("quality.{key}");
                self.name(&Value::String(key.clone()), &path);
                cleaned.insert(key.clone(), self.json(record, &path, 0, None));
            }
            out.insert("quality".into(), Value::Object(cleaned));
        }
        if let Some(label_set) = doc.get("label_set") {
            self.json(label_set, "label_set", 0, Some(VOCABULARY_REVIEW));
        }
        // The attestation is scanned, not cleaned: `--apply` replaces it
        // whole.  A gate reads the scan, and a name in a record is a name.
        if let (Some(record), false) = (record, self.writing) {
            self.record(record);
        }
        out
    }

    /// The de-identification record's strings.  `date` says when the record
    /// was made, not when the subject was seen, and the rest are its own
    /// scalars; `performed_by` names an agent.
    fn record(&mut self, record: &Map<String, Value>) {
        for (key, value) in record {
            let path = format!("deidentification.{key}");
            match key.as_str() {
                "date" | "date_shift_days" | "id_mapping" | "burned_in_annotation_checked" => {}
                "performed_by" => self.name(value, &path),
                _ => {
                    self.json(value, &path, 0, None);
                }
            }
        }
    }

    fn identity(&mut self, identity: &Map<String, Value>, doc: &Map<String, Value>) -> Map<String, Value> {
        let mut out = identity.clone();
        for name in ["sample_id", "subject_id"] {
            self.identity_id(identity, name, doc);
        }
        for (key, value) in identity {
            if ["sample_id", "subject_id", "sex", "laterality", ID_SOURCE].contains(&key.as_str()) {
                continue;
            }
            let r#where =
                if key == "bodypart" { "identity.bodypart".to_string() } else { format!("identity.extra.{key}") };
            let kept = self.entry(key, Slot::Json(value), &r#where, 0, None);
            let renamed = match kept {
                Kept::Drop => None,
                _ => self.key(key, &r#where, None),
            };
            match (kept, renamed) {
                (Kept::Drop, _) | (_, None) => {
                    out.shift_remove(key);
                }
                (kept, Some(name)) => {
                    let value = match kept {
                        Kept::Value(v) => v,
                        _ => value.clone(),
                    };
                    if &name != key {
                        out.shift_remove(key);
                    }
                    out.insert(name, value);
                }
            }
        }
        out
    }

    /// Whether one of the sample's own ids is an identifier (§11.4): never
    /// actionable by default, never silent either.
    fn identity_id(&mut self, identity: &Map<String, Value>, name: &str, doc: &Map<String, Value>) {
        let value = identity.get(name).map(py_str).unwrap_or_default();
        if value.starts_with(PSEUDONYM_PREFIX) {
            return;
        }
        let r#where = format!("identity.{name}");
        let remint = "re-mint it, or re-run --apply with --pseudonymise-ids";
        if is_person_name(&value) {
            self.find(
                "person_name",
                &r#where,
                &format!("reads as a DICOM person name, not a pseudonym (§11.4); {remint}"),
                Some(&value),
                false,
                None,
            );
        }
        if has_date(&value) {
            self.find(
                "date",
                &r#where,
                &format!("contains what looks like a date; {remint}"),
                Some(&value),
                false,
                None,
            );
        }
        let source = identity.get(ID_SOURCE).and_then(Value::as_object);
        let empty = source.is_none_or(|s| s.is_empty());
        let origin = source.and_then(|s| s.get(name)).map(py_str).unwrap_or_default();
        if let Some(tag) = origin.strip_prefix("dicom:") {
            self.find(
                "identity",
                &r#where,
                &format!("copied from DICOM {tag}, a direct identifier (§11.4); {remint}"),
                Some(&value),
                false,
                None,
            );
        } else if has_uid_token(&value) {
            self.find("identity", &r#where, &format!("contains a real DICOM UID; {remint}"), Some(&value), false, None);
        } else if empty && imported_from_dicom(doc) {
            self.find(
                "identity",
                &r#where,
                &format!(
                    "this sample was imported from DICOM, which keys a sample by PatientID, and records no source for \
                     this id; check it is not the record number and record where it came from in identity.{ID_SOURCE}, \
                     or {remint}"
                ),
                Some(&value),
                false,
                None,
            );
        }
    }

    fn timepoint(&mut self, tp: &Value, index: usize) -> Value {
        let Value::Object(tp) = tp else { return tp.clone() };
        let r#where = format!("timepoints[{index}]");
        let mut out = tp.clone();
        for (key, value) in tp {
            let path = format!("{}.{key}", r#where);
            match key.as_str() {
                "index" | "days_from_baseline" => {}
                "id" => self.name(value, &path),
                "study_uid" => {
                    out.insert(key.clone(), self.uid(value, &path));
                }
                "series_uids" => {
                    let mut cleaned = Map::new();
                    if let Value::Object(map) = value {
                        for (image, uid) in map {
                            cleaned.insert(image.clone(), self.uid(uid, &format!("{path}.{image}")));
                        }
                    }
                    out.insert(key.clone(), Value::Object(cleaned));
                }
                "date" => {
                    if !crate::json::py_truthy(value) {
                        continue;
                    }
                    if self.dates_shifted {
                        // Shifted once already; but a date added since was
                        // not, and only its interval can tell.
                        if let Some(detail) = self.disagreeing.get(&index).cloned() {
                            self.find("date", &path, &detail, Some(&py_str(value)), false, None);
                        }
                        continue;
                    }
                    let text = py_str(value);
                    self.find(
                        "date",
                        &path,
                        "a date with no recorded shift; either shift the cohort consistently or drop it (§11.4)",
                        Some(&text),
                        true,
                        None,
                    );
                    if self.writing {
                        let moved = shift(&text, self.date_shift_days);
                        let done = if moved.as_deref().is_some_and(|m| !m.is_empty()) { "shifted" } else { "removed" };
                        match moved {
                            None => {
                                out.shift_remove(key);
                            }
                            Some(m) => {
                                out.insert(key.clone(), Value::String(m));
                            }
                        }
                        self.actions.push(format!("{path} {done}"));
                    }
                }
                "subject_age_years" => {
                    let Some(age) = value.as_f64() else { continue };
                    if age <= AGE_LIMIT {
                        continue;
                    }
                    self.find(
                        "age",
                        &path,
                        "an age over 89 identifies on its own under HIPAA Safe Harbor, which aggregates them as 90 or \
                         older; --profile strict records it as 90",
                        Some(&py_str(value)),
                        false,
                        None,
                    );
                    if self.writing && self.strict {
                        out.insert(key.clone(), crate::json::num(AGE_LIMIT));
                        self.actions.push(format!("{path} recorded as 90 (strict)"));
                    }
                }
                _ => {
                    out.insert(key.clone(), self.json(value, &path, 0, None));
                }
            }
        }
        Value::Object(out)
    }

    /// A field that holds a UID: pseudonymised when it is a real one.
    fn uid(&mut self, value: &Value, r#where: &str) -> Value {
        let text = py_str(value);
        if is_dicom_uid(&text) {
            self.find("uid", r#where, UID_DETAIL, Some(&text), true, None);
            let pseudonym = self.pseudonym(&text);
            if self.writing {
                self.actions.push(format!("{} pseudonymised", r#where));
                return Value::String(pseudonym);
            }
            return value.clone();
        }
        self.json(value, r#where, 0, None)
    }

    fn provenance(&mut self, prov: &Map<String, Value>) -> Map<String, Value> {
        let mut out = prov.clone();
        let agents: Vec<Value> = prov
            .get("agents")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|agent| self.agent(agent)).collect())
            .unwrap_or_default();
        out.insert("agents".into(), Value::Array(agents));
        let activities: Vec<Value> = prov
            .get("activities")
            .and_then(Value::as_array)
            .map(|a| a.iter().map(|activity| self.activity(activity)).collect())
            .unwrap_or_default();
        out.insert("activities".into(), Value::Array(activities));
        out
    }

    /// An agent's name, by what the agent is; then everything else it holds.
    fn agent(&mut self, agent: &Value) -> Value {
        let Value::Object(agent) = agent else { return agent.clone() };
        let id = agent.get("id").cloned().unwrap_or(Value::Null);
        let r#where = format!("provenance.agents[{}]", py_str(&id));
        let mut out = agent.clone();
        let kind = agent.get("type").and_then(Value::as_str).unwrap_or_default();
        let name = agent.get("name").filter(|v| crate::json::py_truthy(v)).map(py_str).unwrap_or_default();
        let pseudonymous = name.starts_with(PSEUDONYM_PREFIX);
        if !name.is_empty() && !pseudonymous && (kind == "person" || kind == "organization") {
            if kind == "person" {
                self.find(
                    "staff_name",
                    &r#where,
                    "names a person; identifying for them, though not for the subject --- review against your \
                     governance",
                    Some(&name),
                    false,
                    None,
                );
            } else {
                self.find(
                    "organization_name",
                    &r#where,
                    "names an organization; if it is the site that imaged the subject it identifies them in \
                     combination (InstitutionName is removed by the DICOM basic profile) --- pseudonymised under \
                     --profile strict",
                    Some(&name),
                    false,
                    None,
                );
            }
            if self.writing && self.strict {
                // Salted and recorded like every other pseudonym: an unsalted
                // hash of a name is reversed by hashing candidate names.
                out.insert("name".into(), Value::String(self.pseudonym(&name)));
                self.actions.push(format!("{}.name pseudonymised", r#where));
            }
        } else if !name.is_empty() {
            // The format requires a name, so one that reads as a person's
            // becomes a pseudonym rather than blank.
            let removed = pseudonymise(&name, &self.salt);
            let cleaned = self.text(&name, &format!("{}.name", r#where), 0, None, &removed);
            if self.writing && cleaned == removed {
                self.uid_map.insert(name.clone(), removed);
            }
            out.insert("name".into(), Value::String(cleaned));
        }
        for (key, value) in agent {
            match key.as_str() {
                "id" | "type" | "name" => {}
                // The id of an organization agent (§11.1): reviewed as an id.
                "organization" => self.name(value, &format!("{}.organization", r#where)),
                // A build's date, not the subject's: reported, and kept.
                "version" if value.as_str().is_some_and(is_whole_date) => self.find(
                    "date",
                    &format!("{}.version", r#where),
                    "a software version that reads as a date: a build's, not the subject's, so --apply leaves it",
                    value.as_str(),
                    false,
                    None,
                ),
                _ => {
                    out.insert(key.clone(), self.json(value, &format!("{}.{key}", r#where), 0, None));
                }
            }
        }
        self.name(&id, &format!("{}.id", r#where));
        Value::Object(out)
    }

    fn activity(&mut self, activity: &Value) -> Value {
        let Value::Object(activity) = activity else { return activity.clone() };
        let id = activity.get("id").cloned().unwrap_or(Value::Null);
        let r#where = format!("provenance.activities[{}]", py_str(&id));
        let mut out = activity.clone();
        for (key, value) in activity {
            if ["id", "type", "agent", "started", "ended"].contains(&key.as_str()) {
                continue;
            }
            let path = format!("{}.{key}", r#where);
            if key == "inputs" || key == "outputs" {
                let cleaned: Vec<Value> = value
                    .as_array()
                    .map(|entries| {
                        entries
                            .iter()
                            .enumerate()
                            .map(|(i, e)| Value::String(self.reference(&py_str(e), &format!("{path}[{i}]"))))
                            .collect()
                    })
                    .unwrap_or_default();
                out.insert(key.clone(), Value::Array(cleaned));
            } else {
                out.insert(key.clone(), self.json(value, &path, 0, None));
            }
        }
        self.name(&id, &format!("{}.id", r#where));
        Value::Object(out)
    }

    fn split(&mut self, claim: &Value, index: usize) -> Value {
        let Value::Object(claim) = claim else { return claim.clone() };
        let r#where = format!("splits[{index}]");
        let mut out = claim.clone();
        for (key, value) in claim {
            match key.as_str() {
                "partition" | "fold" | "assigned_at" | "manifest_sha256" => {}
                "set_id" => self.name(value, &format!("{}.set_id", r#where)),
                _ => {
                    out.insert(key.clone(), self.json(value, &format!("{}.{key}", r#where), 0, None));
                }
            }
        }
        Value::Object(out)
    }

    // -- the HDF5 file ---------------------------------------------------------

    /// Every link name, attribute and string dataset outside `/meta`.
    ///
    /// One walk ([`links`]) reaches every link, so a name is read wherever a
    /// link holds one --- a later hard link, a soft link, a committed type ---
    /// and reaches every object once, at its first path, for what it holds.
    fn hdf5(&mut self, root: &hdf5::Group) -> Result<()> {
        self.attributes(root, "root", &ROOT_MANAGED_ATTRS)?;
        let meta = ops::object_id_by_name(root, META_DATASET);
        let links = links(root)?;
        let mut copies = Vec::new();
        let mut aliases = Vec::new();
        for link in &links {
            let r#where = link.path.replace('/', ".");
            let leaf = link.path.rsplit('/').next().unwrap_or(&link.path).to_string();
            self.name(&Value::String(leaf.clone()), &r#where);
            if let Some(target) = &link.target {
                // What a soft link points at is a string in the file too.
                for part in target.split('/').filter(|p| !p.is_empty()) {
                    self.name(&Value::String(part.to_string()), &format!("{} (link target)", r#where));
                }
            }
            // `/meta` under a second name: what a rewrite of `/meta` leaves
            // behind --- an amendment once copied the old document there, and
            // a scrub that pseudonymised the ids left them in it.
            if link.path != META_DATASET && meta.is_some() && link.id == meta {
                let named = link
                    .parent
                    .dataset(&leaf)
                    .ok()
                    .and_then(|d| data::read_strings(&d).ok())
                    .and_then(|values| values.iter().find_map(|v| document_copy(v)));
                self.find("identity", &r#where, ALIAS_DETAIL, named.as_deref(), true, None);
                if self.writing {
                    self.actions.push(format!("{} removed (another name for /meta)", r#where));
                    aliases.push(link);
                }
            }
            let Some(object) = &link.object else { continue };
            let parts: Vec<&str> = link.path.split('/').collect();
            // The frame graph is pseudonymised as one mapping (`scan_frames`).
            let skip: Vec<&str> = if parts.len() == 2 {
                FRAME_ATTRS.iter().find(|(g, _)| *g == parts[0]).map(|(_, a)| a.to_vec()).unwrap_or_default()
            } else {
                Vec::new()
            };
            match object {
                Object::Group(g) => self.attributes(g, &r#where, &skip)?,
                Object::Datatype(t) => self.attributes(t, &r#where, &[])?,
                Object::Dataset(d) => {
                    self.attributes(d, &r#where, &skip)?;
                    // `/meta` under any name is the document `document` read.
                    let document = link.path == META_DATASET || (meta.is_some() && link.id == meta);
                    if !document && self.dataset(d, &r#where, &link.path)? {
                        copies.extend(link.id);
                    }
                }
            }
        }
        if self.writing {
            // Every name a copy has goes, or its bytes stay reachable; an
            // alias goes alone, since the object it names is `/meta`.
            let named = links.iter().filter(|l| l.id.is_some_and(|id| copies.contains(&id)));
            for link in named.chain(aliases) {
                link.parent.unlink(link.path.rsplit('/').next().unwrap_or(&link.path))?;
            }
        }
        if !self.unexamined.is_empty() {
            let named: Vec<&str> = self.unexamined.iter().take(UNEXAMINED_NAMED).map(String::as_str).collect();
            let more = self.unexamined.len().saturating_sub(UNEXAMINED_NAMED);
            self.report.not_checked.push(format!(
                "non-string datasets outside the imaging groups, whose numbers and bytes are not decoded: {}{}",
                named.join(", "),
                if more > 0 { format!(", and {more} more") } else { String::new() }
            ));
        }
        Ok(())
    }

    fn attributes(&mut self, node: &hdf5::Location, r#where: &str, skip: &[&str]) -> Result<()> {
        for key in attrs::names(node)? {
            if skip.contains(&key.as_str()) {
                continue;
            }
            let path = format!("{}.{key}", r#where);
            let raw = match attrs::read(node, &key) {
                Ok(Some(v)) => v,
                Ok(None) => continue,
                Err(_) => {
                    self.find(
                        "unreadable",
                        &path,
                        "an attribute this tool cannot decode; a person must look at it",
                        None,
                        false,
                        None,
                    );
                    continue;
                }
            };
            let (value, opaque) = match &raw {
                AttrValue::Str(s) => (Value::String(s.clone()), None),
                AttrValue::Strs(items) => (json!(items), None),
                AttrValue::Bool(b) => (Value::Bool(*b), None),
                AttrValue::Int(i) => (json!(i), None),
                AttrValue::Float(f) => (crate::json::num(*f), None),
                AttrValue::Array(array) => (Value::Null, Some(numpy_str(array))),
                AttrValue::Unsupported(_) => {
                    self.find(
                        "unreadable",
                        &path,
                        "an attribute this tool cannot decode; a person must look at it",
                        None,
                        false,
                        None,
                    );
                    continue;
                }
            };
            if let Some(named) = raw.as_str_list().unwrap_or_default().iter().find_map(|s| document_copy(s)) {
                self.find("identity", &path, COPY_DETAIL, Some(&named), true, None);
                if self.writing {
                    attrs::delete(node, &key)?;
                    self.actions.push(format!("{path} removed (a copy of the sample document)"));
                }
                continue;
            }
            if REFERENCE_ATTRS.contains(&key.as_str()) {
                match &value {
                    Value::Array(items) => {
                        for (i, item) in items.iter().enumerate() {
                            self.name(item, &format!("{path}[{i}]"));
                        }
                    }
                    other => self.name(other, &path),
                }
                continue;
            }
            let slot = match &opaque {
                Some(text) => Slot::Opaque(text),
                None => Slot::Json(&value),
            };
            let kept = self.entry(&key, slot, &path, 0, None);
            if !self.writing {
                continue;
            }
            match kept {
                Kept::Same => {}
                Kept::Drop => attrs::delete(node, &key)?,
                Kept::Value(new) => {
                    if new != value {
                        let encoded = match &new {
                            Value::String(s) => AttrValue::Str(s.clone()),
                            Value::Array(items) if items.iter().all(Value::is_string) => {
                                AttrValue::Strs(items.iter().map(py_str).collect())
                            }
                            other => AttrValue::Str(crate::json::dumps(other, crate::json::Style::PYTHON)),
                        };
                        attrs::write(node, &key, &encoded)?;
                    }
                }
            }
        }
        Ok(())
    }

    /// A dataset's strings, or why they were not read.  `true` when it holds
    /// a copy of a sample document, which a writing sweep removes.
    fn dataset(&mut self, ds: &hdf5::Dataset, r#where: &str, path: &str) -> Result<bool> {
        let kind = data::kind(ds)?;
        if kind != data::Kind::Strings {
            self.unread(ds, &kind, r#where, path);
            return Ok(false);
        }
        let values = match data::read_strings(ds) {
            Ok(v) => v,
            Err(_) => {
                self.find(
                    "unreadable",
                    r#where,
                    "a string dataset this tool cannot read; a person must look at it",
                    None,
                    false,
                    None,
                );
                return Ok(false);
            }
        };
        if let Some(named) = values.iter().find_map(|v| document_copy(v)) {
            self.find("identity", r#where, COPY_DETAIL, Some(&named), true, None);
            if self.writing {
                self.actions.push(format!("{} removed (a copy of the sample document)", r#where));
            }
            return Ok(true);
        }
        let kept: Vec<String> =
            values.iter().enumerate().map(|(i, v)| self.text(v, &format!("{}[{i}]", r#where), 0, None, "")).collect();
        if self.writing && kept != values {
            rewrite_strings(ds, &kept)?;
        }
        Ok(false)
    }

    /// A dataset whose values this tool does not read.  Strings inside its
    /// type are reported wherever it is; outside [`DEFINED_GROUPS`] so is a
    /// byte buffer or a type it cannot decode, either of which can carry text,
    /// and every one there is named in `not_checked`.
    fn unread(&mut self, ds: &hdf5::Dataset, kind: &data::Kind, r#where: &str, path: &str) {
        let defined = path.split('/').next().is_some_and(|top| DEFINED_GROUPS.contains(&top));
        let detail = match strings_inside(ds) {
            Inside::Strings => Some(
                "strings inside a compound, array or variable-length type; this tool does not decode those, so a \
                 person must look at it"
                    .to_string(),
            ),
            Inside::TooDeep => Some(format!(
                "a type nested more than {MAX_DEPTH} levels deep; this tool did not look inside it, and a person must"
            )),
            _ if defined => None,
            Inside::Unknown => Some(
                "a type this tool cannot decode, outside the imaging groups; it may hold text, so a person must look \
                 at it"
                    .to_string(),
            ),
            Inside::Nothing => match kind {
                data::Kind::Numeric(DType::U8 | DType::I8) if ds.ndim() <= 1 => Some(
                    "a byte buffer outside the imaging groups; it may hold text this tool does not decode, so a person \
                     must look at it"
                        .to_string(),
                ),
                data::Kind::Other(t) => Some(format!(
                    "a dataset of type {t}, outside the imaging groups; this tool does not decode it, so a person \
                     must look at it"
                )),
                _ => None,
            },
        };
        if !defined {
            self.unexamined.push(path.to_string());
        }
        if let Some(detail) = detail {
            self.find("unreadable", r#where, &detail, None, false, None);
        }
    }
}

/// One link [`links`] reached.
struct Link {
    /// `/`-separated, from the walk's root.
    path: String,
    /// The group that holds the link.
    parent: hdf5::Group,
    /// The object a hard link names; `None` for any other link.
    id: Option<ops::ObjectId>,
    /// The path a soft link holds.
    target: Option<String>,
    /// The object, the first time the walk reaches it.
    object: Option<Object>,
}

/// An object [`links`] reached.
enum Object {
    Group(hdf5::Group),
    Dataset(hdf5::Dataset),
    Datatype(hdf5::CommittedDatatype),
}

/// Every link under `root`, depth first in name order.
///
/// [`ops::visit`] reaches each object once, which is right for a digest and
/// wrong for names: a second hard link, a soft link and a committed type each
/// hold a name it never yields.  This walk yields every link, and expands each
/// group once, so a hard link back to an ancestor ends rather than loops.
/// The objects it reaches first are the ones `ops::visit` reaches, at the
/// same paths.
fn links(root: &hdf5::Group) -> Result<Vec<Link>> {
    fn walk(
        group: &hdf5::Group,
        prefix: &str,
        depth: usize,
        seen: &mut std::collections::HashSet<ops::ObjectId>,
        out: &mut Vec<Link>,
    ) -> Result<()> {
        if depth > ops::MAX_DEPTH {
            return Err(Error::File(format!(
                "{} is nested more than {} groups deep; it is refused rather than followed",
                repr_str(&group.name()),
                ops::MAX_DEPTH
            )));
        }
        for name in ops::members(group)? {
            let path = if prefix.is_empty() { name.clone() } else { format!("{prefix}/{name}") };
            let kind = ops::link_kind(group, &name);
            let hard = kind == Some(ops::LinkKind::Hard);
            let id = if hard { ops::object_id_by_name(group, &name) } else { None };
            let target =
                if kind == Some(ops::LinkKind::Soft) { ops::soft_link_target(group, &name).ok() } else { None };
            let first = hard && id.is_none_or(|id| seen.insert(id));
            let object = if first {
                match ops::node_kind(group, &name) {
                    Some(ops::NodeKind::Group) => Some(Object::Group(group.group(&name)?)),
                    Some(ops::NodeKind::Dataset) => Some(Object::Dataset(group.dataset(&name)?)),
                    Some(ops::NodeKind::Other) => group.committed_datatype(&name).ok().map(Object::Datatype),
                    None => None,
                }
            } else {
                None
            };
            let child = match &object {
                Some(Object::Group(g)) => Some(g.clone()),
                _ => None,
            };
            out.push(Link { path: path.clone(), parent: group.clone(), id, target, object });
            if let Some(child) = child {
                walk(&child, &path, depth + 1, seen, out)?;
            }
        }
        Ok(())
    }
    // The root itself: a link back to it is a cycle too.
    let mut seen = ops::object_id_of(root).into_iter().collect();
    let mut out = Vec::new();
    walk(root, "", 0, &mut seen, &mut out)?;
    Ok(out)
}

/// Whether a type holds strings, and where a walk of it stopped.
enum Inside {
    Nothing,
    Strings,
    TooDeep,
    /// The type could not be described at all.
    Unknown,
}

/// Strings anywhere in a dataset's type: a compound's fields, an array's or
/// a variable-length sequence's elements, nested --- a flat compound was all
/// the check saw, and a nested one passed as clean.
fn strings_inside(ds: &hdf5::Dataset) -> Inside {
    match ds.dtype().and_then(|t| t.to_descriptor()) {
        Ok(td) => holds_strings(&td, 0),
        Err(_) => Inside::Unknown,
    }
}

fn holds_strings(td: &hdf5::types::TypeDescriptor, depth: usize) -> Inside {
    use hdf5::types::TypeDescriptor as TD;
    if depth > MAX_DEPTH {
        return Inside::TooDeep;
    }
    match td {
        TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_) => Inside::Strings,
        TD::Compound(c) => c
            .fields
            .iter()
            .map(|f| holds_strings(&f.ty, depth + 1))
            .find(|inside| !matches!(inside, Inside::Nothing))
            .unwrap_or(Inside::Nothing),
        TD::FixedArray(inner, _) | TD::VarLenArray(inner) => holds_strings(inner, depth + 1),
        _ => Inside::Nothing,
    }
}

/// A JSON object or array stored as a string, decoded; `None` otherwise.
fn embedded_json(value: &str) -> Option<Value> {
    let text = value.trim();
    if text.chars().count() < 2 || !(text.starts_with('{') || text.starts_with('[')) {
        return None;
    }
    match crate::json::loads(text) {
        Ok(v @ (Value::Object(_) | Value::Array(_))) => Some(v),
        _ => None,
    }
}

/// NumPy's `str()` of an array attribute, for previews.
fn numpy_str(array: &NdArray) -> String {
    let values: Vec<f64> = array.to_f64().iter().copied().collect();
    let items: Vec<String> = match array.dtype() {
        crate::array::DType::Bool => {
            values.iter().map(|v| if *v != 0.0 { " True".into() } else { "False".into() }).collect()
        }
        d if d.is_integer() => array.cast::<i64>().iter().map(|v| v.to_string()).collect(),
        _ => {
            let reprs: Vec<String> = values.iter().map(|v| py_float(*v)).collect();
            let trimmed: Vec<String> = reprs
                .iter()
                .map(|r| r.strip_suffix(".0").map(|s| format!("{s}.")).unwrap_or_else(|| r.clone()))
                .collect();
            let width = trimmed.iter().map(String::len).max().unwrap_or(0);
            trimmed.into_iter().map(|t| format!("{t:<width$}")).collect()
        }
    };
    if array.ndim() == 0 {
        return items.into_iter().next().unwrap_or_default().trim().to_string();
    }
    let joined = items.join(" ");
    let joined = if array.dtype() == crate::array::DType::Bool { joined } else { joined.trim_end().to_string() };
    format!("[{joined}]")
}

/// Write cleaned strings back, keeping the dataset's filter pipeline.
///
/// In place wherever the values fit; a fixed-length dataset narrower than a
/// pseudonym is recreated wider with a copy of its creation property list
/// and its attributes, under a temporary name moved into place.
fn rewrite_strings(ds: &hdf5::Dataset, values: &[String]) -> Result<()> {
    use hdf5::types::{TypeDescriptor as TD, VarLenUnicode};
    let shape = ds.shape();
    let descriptor = ds.dtype()?.to_descriptor()?;
    match descriptor {
        TD::VarLenUnicode | TD::VarLenAscii => {
            let encoded: Vec<VarLenUnicode> = values
                .iter()
                .map(|v| v.parse().map_err(|_| Error::Value(format!("{} is not UTF-8", repr_str(v)))))
                .collect::<Result<_>>()?;
            let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), encoded).map_err(Error::from)?;
            ds.write(&array)?;
            Ok(())
        }
        TD::FixedAscii(n) | TD::FixedUnicode(n) => {
            let ascii = matches!(descriptor, TD::FixedAscii(_)) && values.iter().all(|v| v.is_ascii());
            let width = values.iter().map(|v| v.len()).max().unwrap_or(0).max(n);
            if width == n && (ascii || matches!(descriptor, TD::FixedUnicode(_))) {
                crate::h5::data::write_fixed_strings(ds, values, n, ascii)?;
                return Ok(());
            }
            widen(ds, values, width, ascii)?;
            Ok(())
        }
        _ => Err(Error::Type(format!("{} does not hold strings", ds.name()))),
    }
}

fn widen(ds: &hdf5::Dataset, values: &[String], width: usize, ascii: bool) -> Result<()> {
    crate::h5::data::recreate_fixed_strings(ds, values, width, ascii)
}

fn imported_from_dicom(doc: &Map<String, Value>) -> bool {
    let activities =
        doc.get("provenance").and_then(|p| p.get("activities")).and_then(Value::as_array).cloned().unwrap_or_default();
    for activity in &activities {
        if activity.get("tool").map(py_str).unwrap_or_default().contains("from-dicom") {
            return true;
        }
        if activity
            .get("inputs")
            .and_then(Value::as_array)
            .is_some_and(|inputs| inputs.iter().any(|v| py_str(v).starts_with("dicom:")))
        {
            return true;
        }
    }
    false
}

/// Every rule, over every string in one sample document.  Reads only `/meta`.
pub fn scan_document(document: &SampleDocument, report: &mut ScrubReport) {
    if let Value::Object(doc) = document.to_json() {
        Sweep::new(report).document(&doc);
    }
}

/// Frame-of-reference UIDs, wherever they are named (§3.4).
fn scan_frames(frames: &IndexMap<String, Vec<String>>, report: &mut ScrubReport) {
    for (uid, places) in frames {
        if !is_dicom_uid(uid) {
            continue;
        }
        for location in places {
            report.add(
                "uid",
                location,
                "a real FrameOfReferenceUID; SHOULD be pseudonymised (§3.4)",
                Some(uid),
                true,
                true,
            );
        }
    }
}

/// What a file name says a sample is called: without its suffix, and up to
/// the first dot.
fn stems(path: &Path) -> Vec<String> {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stem = Path::new(&name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let first = name.split('.').next().unwrap_or("").to_string();
    vec![stem, first]
}

/// A file named after an id that is itself an identifier.
fn scan_file_name(path: &Path, sample_id: &str, subject_id: &str, report: &mut ScrubReport) {
    let name_of_file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let stems = stems(path);
    if stems.iter().any(|s| has_uid_token(s)) {
        report.add(
            "file_name",
            "file name",
            "the file name contains a real DICOM UID; rename it before sharing --- medh5 does not rename files",
            Some(&name_of_file),
            false,
            true,
        );
        return;
    }
    let flagged: Vec<String> = report.findings.iter().map(|f| f.r#where.clone()).collect();
    for (name, value) in [("sample_id", sample_id), ("subject_id", subject_id)] {
        if stems.iter().any(|s| s == value) && flagged.contains(&format!("identity.{name}")) {
            report.add(
                "file_name",
                "file name",
                &format!(
                    "the file is named after identity.{name}, which is reported above; rename it before sharing --- \
                     medh5 does not rename files"
                ),
                Some(&name_of_file),
                false,
                true,
            );
            return;
        }
    }
}

/// Mark the strict-profile rules actionable, except where nothing can act.
fn escalate(report: &mut ScrubReport) {
    if report.profile != "strict" {
        return;
    }
    for finding in &mut report.findings {
        if STRICT_RULES.contains(&finding.rule.as_str())
            && finding.fixable
            && !UNFIXABLE_LOCATIONS.contains(&finding.r#where.as_str())
        {
            finding.actionable = true;
        }
    }
}

fn check_profile(profile: &str) -> Result<()> {
    if PROFILES.contains(&profile) {
        Ok(())
    } else {
        Err(Error::invalid(format!("unknown profile {}; expected one of {}", repr_str(profile), repr_list(&PROFILES))))
    }
}

/// Find identifiers in one file.  Changes nothing.
pub fn scan(path: &Path, profile: &str) -> Result<ScrubReport> {
    scan_with(path, profile, &[])
}

/// [`scan`], looking for the `replaced` ids as well: only the run that
/// replaced them knows what to look for.
fn scan_with(path: &Path, profile: &str, replaced: &[Replaced]) -> Result<ScrubReport> {
    check_profile(profile)?;
    let mut report = ScrubReport::new(&path.to_string_lossy(), profile);
    let sample = open_sample(path)?;
    let document = sample.document()?;
    let doc = match document.to_json() {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    {
        // One sweep: what `document` learns (a recorded date shift) governs
        // the attributes too.
        let mut sweep = Sweep::new(&mut report);
        sweep.replaced = replaced.to_vec();
        sweep.document(&doc);
        scan_frames(&frame_references(&sample.root)?, sweep.report);
        sweep.hdf5(&sample.root)?;
    }
    scan_file_name(path, &document.identity.sample_id, &document.identity.subject_id, &mut report);
    if let Some(extent) = clinical_extent(&sample)? {
        report.add(
            "clinical",
            "clinical",
            &format!(
                "the clinical profile ({extent}) is not examined by this tool: its documents, coded and free-text \
                 values and its clock carry what the sweep does not read, and a date shift would not move that clock \
                 (1.1 §6) --- a person must review it, or share the imaging projection (`medh5 clinical strip`)"
            ),
            None,
            false,
            false,
        );
        report.not_checked.push("the clinical profile: its documents, its event text and its clock".into());
    }
    escalate(&mut report);
    Ok(report)
}

/// How much a sample's clinical profile holds, when it has one.
///
/// The profile is outside what this tool examines.  Its text is packed UTF-8
/// (`uint8` buffers), which the string sweep never reads, and a date shift
/// moves `/meta`'s dates, not the clinical clock, so the images and the
/// history would tell two times.  1.1 §6 says de-identification covers the
/// body as well as the metadata; until this tool covers it, a clinical sample
/// is never reported clean and is not scrubbed (see [`apply`]).  A
/// `clinical` group the sample does not declare (E803) is read no better, so
/// it counts too.
fn clinical_extent(sample: &Sample) -> Result<Option<String>> {
    if !sample.profiles()?.contains(crate::clinical::PROFILE) {
        let undeclared = ops::exists(&sample.root, crate::clinical::GROUP);
        return Ok(undeclared.then(|| "a `clinical` group the sample does not declare".to_string()));
    }
    Ok(Some(match sample.clinical() {
        Ok(Some(c)) => format!("{} event version(s), {} document(s)", c.events.len(), c.documents().len()),
        _ => "tables this tool could not read".into(),
    }))
}

/// What `apply` does beyond scanning.
#[derive(Debug, Clone, Default)]
pub struct ApplyOptions {
    pub profile: String,
    pub salt: String,
    /// Shift every date found by this many days instead of dropping it.
    pub date_shift_days: Option<i64>,
    pub performed_by: Option<String>,
    /// Replace `sample_id` and `subject_id` with salted stable pseudonyms.
    pub pseudonymise_ids: bool,
}

/// Act on the actionable findings and write the §11.4 attestation.
pub fn apply(path: &Path, options: &ApplyOptions) -> Result<ScrubReport> {
    let profile = if options.profile.is_empty() { "basic" } else { options.profile.as_str() };
    if options.pseudonymise_ids && options.salt.is_empty() {
        return Err(Error::invalid(
            "pseudonymise_ids needs a salt: a sample or subject id is usually a record number, and an unsalted hash \
             of one is reversed by hashing every record number; pass --salt and keep it apart from the data",
        ));
    }
    if options.date_shift_days == Some(0) {
        // Recorded, a shift of nothing silenced every later date scan.
        return Err(Error::invalid(
            "date_shift_days=0 shifts nothing, and a file that records a shift has its dates passed over by every \
             later scan; pass a shift of at least one day, or none to remove the dates",
        ));
    }
    let mut report = scan(path, profile)?;
    let shown = repr_str(&path.to_string_lossy());
    // Refused before anything is written.
    let (recorded, replaced) = {
        let sample = open_sample(path)?;
        if let Some(extent) = clinical_extent(&sample)? {
            // The attestation would claim a de-identification this tool did
            // not do (`clinical_extent`).
            return Err(Error::invalid(format!(
                "{shown} carries the clinical profile ({extent}), which this tool does not de-identify: its text and \
                 clock would keep what a scrub removes elsewhere, under an attestation that says otherwise. Share the \
                 imaging projection (`medh5 clinical strip`), or de-identify the clinical records with a tool that \
                 covers them and record it"
            )));
        }
        let document = sample.document()?;
        let days = document.deidentification.as_ref().and_then(|r| r.date_shift_days.clone()).map(Value::Number);
        let replaced = if options.pseudonymise_ids { ids_replaced(document, &options.salt) } else { Vec::new() };
        (recorded_shift(days.as_ref()), replaced)
    };
    if let (Some(asked), Some(recorded)) = (options.date_shift_days, recorded) {
        if asked != recorded {
            // Its dates moved by one amount; a second would leave the record
            // true of some dates and false of the rest.
            return Err(Error::invalid(format!(
                "{shown} records a date shift of {recorded} days, which its dates were moved by; shifting by {asked} \
                 would make the record false --- pass the recorded shift, or none"
            )));
        }
    }
    let strict = profile == "strict";
    let mut renamed: Vec<(String, String)> = Vec::new();
    let mut writer = amend(path, None)?;
    let result = (|| -> Result<()> {
        let mut date_shift_days = options.date_shift_days;
        let mut notes = Vec::new();
        if let Some(days) = recorded {
            // Shifting twice makes the recorded offset a lie.
            date_shift_days = Some(days);
            notes.push(format!("dates were left alone: already shifted by {days} days"));
        }
        // The performer is declared before the sweep, so the name this run
        // adds gets the same treatment as the names it found.
        let agent = match &options.performed_by {
            Some(name) => writer.person(name, None, Map::new())?,
            None => writer.software("medh5", Some(crate::VERSION), Map::new())?,
        };
        let mut scratch = ScrubReport::new("<apply>", profile);
        let doc = match writer.document().to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        let mut sweep = Sweep::new(&mut scratch);
        sweep.writing = true;
        sweep.strict = strict;
        sweep.salt = options.salt.clone();
        sweep.date_shift_days = date_shift_days;
        sweep.uid_map = report.uid_map.clone();
        sweep.replaced = replaced.clone();
        let mut cleaned = sweep.document(&doc);
        if options.pseudonymise_ids {
            renamed = pseudonymise_ids(&mut cleaned, &options.salt, &mut sweep.uid_map, &mut sweep.actions);
        }
        writer.set_document(SampleDocument::from_json(&Value::Object(cleaned))?);
        let root = writer.root()?;
        sweep.hdf5(&root)?;
        let actions = std::mem::take(&mut sweep.actions);
        let uid_map = std::mem::take(&mut sweep.uid_map);
        drop(sweep);
        report.uid_map = uid_map;

        // One mapping, applied to every reference at once.
        let mut frame_map = std::collections::HashMap::new();
        for uid in writer.frame_uids()?.keys() {
            if is_dicom_uid(uid) {
                let pseudonym = pseudonymise(uid, &options.salt);
                report.uid_map.insert(uid.clone(), pseudonym.clone());
                frame_map.insert(uid.clone(), pseudonym);
            }
        }
        let mut removed: Vec<String> = notes;
        removed.extend(actions);
        removed.extend(writer.remap_frame_uids(&frame_map)?.into_iter().map(|l| format!("{l} -> pseudonymised")));

        let summary = format!(
            "medh5 scrub {profile}: container metadata only, voxel data not examined; {}{}",
            if strict { "quasi-identifiers removed" } else { "quasi-identifiers retained for review" },
            if options.pseudonymise_ids {
                "; sample and subject ids pseudonymised"
            } else {
                "; sample and subject ids retained"
            }
        );
        let mut record = Map::new();
        record.insert("method".into(), json!("medh5-scrub"));
        record.insert("profile".into(), json!(summary));
        record.insert("date_shift_days".into(), json!(date_shift_days));
        record.insert(
            "id_mapping".into(),
            json!(if !options.salt.is_empty() && !report.uid_map.is_empty() { "external" } else { "none" }),
        );
        record.insert("performed_by".into(), json!(agent.id));
        record.insert("burned_in_annotation_checked".into(), json!(false));
        writer.deidentification(record)?;
        // Re-scan the cleaned state after the record exists, and put the
        // result in the activity, where an auditor of the attestation looks.
        let left = rescan(&writer, profile, &replaced)?;
        let mut fields = Map::new();
        fields.insert("tool".into(), json!(format!("medh5 scrub --profile {profile}")));
        fields.insert(
            "params".into(),
            json!({
                "findings": report.findings.len(),
                "changes": removed.len(),
                "remaining": left.findings.len(),
                "remaining_actionable": left.actionable().len(),
            }),
        );
        writer.activity("deidentify", Some(&agent.id), None, fields)?;
        report.actions = removed;
        report.applied = true;
        // HDF5 does not reclaim superseded storage, so every value this run
        // removed or replaced would survive in the file's freed space: the
        // commit rewrites the file from its root before it replaces anything.
        writer.compact = true;
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    // The claim is checked against the file, not against the intention ---
    // and for the ids it replaced, which a later scan cannot know to look for.
    report.remaining = scan_with(path, profile, &replaced)?.findings;
    let stems = stems(path);
    let name_of_file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    for (name, old) in &renamed {
        if stems.contains(old) {
            report.remaining.push(Finding {
                rule: "file_name".into(),
                r#where: "file name".into(),
                detail: format!(
                    "the file is still named after the {name} this run replaced; rename it before sharing --- medh5 \
                     does not rename files, and a later scan cannot tell the name was an id"
                ),
                value: Some(name_of_file.clone()),
                actionable: false,
                fixable: true,
            });
            break;
        }
    }
    Ok(report)
}

/// Replace the sample's ids with salted pseudonyms; return what was replaced.
///
/// A `cohort.group_id` equal to a replaced id is replaced with it, so every
/// file of one subject, scrubbed with one salt, still groups together.
fn pseudonymise_ids(
    doc: &mut Map<String, Value>,
    salt: &str,
    uid_map: &mut IndexMap<String, String>,
    actions: &mut Vec<String>,
) -> Vec<(String, String)> {
    let mut replaced = Vec::new();
    let mut mapping: BTreeMap<String, String> = BTreeMap::new();
    let Some(Value::Object(identity)) = doc.get_mut("identity") else { return replaced };
    for name in ["sample_id", "subject_id"] {
        let value = identity.get(name).map(py_str).unwrap_or_default();
        if value.starts_with(PSEUDONYM_PREFIX) {
            continue;
        }
        let pseudonym = pseudonymise(&value, salt);
        identity.insert(name.into(), Value::String(pseudonym.clone()));
        uid_map.insert(value.clone(), pseudonym.clone());
        mapping.insert(value.clone(), pseudonym);
        replaced.push((name.to_string(), value));
        actions.push(format!("identity.{name} pseudonymised"));
    }
    if replaced.is_empty() {
        return replaced;
    }
    let mut source = identity.get(ID_SOURCE).and_then(Value::as_object).cloned().unwrap_or_default();
    for (name, _) in &replaced {
        source.insert(name.clone(), json!(PSEUDONYM_SOURCE));
    }
    identity.insert(ID_SOURCE.into(), Value::Object(source));
    if let Some(Value::Object(cohort)) = doc.get_mut("cohort") {
        let current = cohort.get("group_id").and_then(Value::as_str).map(str::to_string);
        if let Some(new) = current.and_then(|g| mapping.get(&g).cloned()) {
            cohort.insert("group_id".into(), Value::String(new));
            actions.push("cohort.group_id pseudonymised with the id it equalled".into());
        }
    }
    replaced
}

/// The ids `--pseudonymise-ids` replaces in a document, each value once;
/// one too short to tell from a word or a number is replaced, not sought.
fn ids_replaced(document: &SampleDocument, salt: &str) -> Vec<Replaced> {
    let mut out: Vec<Replaced> = Vec::new();
    for (name, value) in [("sample_id", &document.identity.sample_id), ("subject_id", &document.identity.subject_id)] {
        if value.starts_with(PSEUDONYM_PREFIX) || value.chars().count() < MIN_SWEPT_ID {
            continue;
        }
        match out.iter_mut().find(|r| &r.old == value) {
            Some(both) => both.name = format!("{} and {name}", both.name),
            None => out.push(Replaced { old: value.clone(), pseudonym: pseudonymise(value, salt), name: name.into() }),
        }
    }
    out
}

/// Every rule over a writer's state, without going back to disk, so the count
/// recorded in the attestation describes the file it ships in.
fn rescan(writer: &crate::sample::writer::SampleWriter, profile: &str, replaced: &[Replaced]) -> Result<ScrubReport> {
    let mut report = ScrubReport::new("<amend>", profile);
    let doc = match writer.document().to_json() {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    {
        let mut sweep = Sweep::new(&mut report);
        sweep.replaced = replaced.to_vec();
        sweep.document(&doc);
        scan_frames(&writer.frame_uids()?, sweep.report);
        sweep.hdf5(&writer.root()?)?;
    }
    escalate(&mut report);
    Ok(report)
}

/// The shift a de-identification record states, if it moved anything: a
/// recorded 0 shifted nothing, so it is read as no shift --- left standing,
/// it silenced every later date scan.
fn recorded_shift(days: Option<&Value>) -> Option<i64> {
    days.and_then(|d| d.as_i64().or_else(|| d.as_f64().map(|f| f as i64))).filter(|d| *d != 0)
}

/// Timepoints, by position, whose date disagrees with the recorded shift,
/// with the detail that says so.
///
/// A shift moves every date by one amount and keeps `days_from_baseline`
/// (§3.7), so each shifted date less its interval names one baseline day; a
/// date added after the shift names another.  The baseline is the one the
/// earliest dated timepoint names (its interval is 0 when it states none),
/// and a date with no interval to check it by is not reported.
fn disagreeing(doc: &Map<String, Value>, shift: i64) -> BTreeMap<usize, String> {
    let timepoints = doc.get("timepoints").and_then(Value::as_array).map(Vec::as_slice).unwrap_or_default();
    let mut implied: Vec<(usize, u64, f64)> = Vec::new();
    for (position, tp) in timepoints.iter().enumerate() {
        let Some(day) = tp.get("date").and_then(Value::as_str).and_then(day_number) else { continue };
        let index = tp.get("index").and_then(Value::as_u64).unwrap_or(position as u64);
        let interval = match tp.get("days_from_baseline").and_then(Value::as_f64) {
            Some(days) => days,
            None if index == 0 => 0.0,
            None => continue,
        };
        implied.push((position, index, day as f64 - interval));
    }
    let Some(&(reference, _, baseline)) = implied.iter().min_by_key(|(_, index, _)| *index) else {
        return BTreeMap::new();
    };
    implied
        .iter()
        .filter(|(_, _, day)| (day - baseline).abs() > BASELINE_TOLERANCE)
        .map(|(position, _, day)| {
            let apart = (day - baseline).round() as i64;
            let detail = format!(
                "a date {apart} day(s) off what timepoints[{reference}].date and both days_from_baseline imply, in a \
                 sample whose record states a shift of {shift} days: a date added after the shift was not shifted --- \
                 a person must check it (§11.4)"
            );
            (*position, detail)
        })
        .collect()
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (i64::from(m) + 9) % 12;
    let doy = (153 * mp + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

fn valid_date(y: i64, m: u32, d: u32) -> bool {
    if !(1..=9999).contains(&y) || !(1..=12).contains(&m) || d == 0 {
        return false;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][(m - 1) as usize];
    d <= days
}

fn moved(y: i64, m: u32, d: u32, days: i64) -> Option<(i64, u32, u32)> {
    if !valid_date(y, m, d) {
        return None;
    }
    let (ny, nm, nd) = crate::sample::writer::civil_from_days(days_from_civil(y, m, d) + days);
    valid_date(ny, nm, nd).then_some((ny, nm, nd))
}

/// Shift every date in a value by `days`, or `None` meaning "drop it".
///
/// What surrounds a date --- the time and zone of a date-time, the words
/// around one in a sentence --- is kept, and every date in the value moves:
/// a value naming two days lost only the first's to a shift that moved one.
fn shift(value: &str, days: Option<i64>) -> Option<String> {
    let days = days?;
    let text = value.trim();
    if let Some(date) = dicom_date(text) {
        let (y, m, d) = moved(date.year, date.month, date.day, days)?;
        return Some(format!("{y:04}{m:02}{d:02}{}", &text[date.end..]));
    }
    let dates = iso_dates(text);
    if dates.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    let mut last = 0;
    for date in dates {
        let (y, m, d) = moved(date.year, date.month, date.day, days)?;
        out.push_str(&text[last..date.start]);
        out.push_str(&format!("{y:04}-{m:02}-{d:02}"));
        last = date.end;
    }
    out.push_str(&text[last..]);
    Some(out)
}

/// Scan or apply over several files.
pub fn scrub_paths<P: AsRef<Path>>(
    paths: &[P],
    apply_changes: bool,
    options: &ApplyOptions,
) -> Result<Vec<ScrubReport>> {
    let profile = if options.profile.is_empty() { "basic" } else { options.profile.as_str() };
    paths.iter().map(|p| if apply_changes { apply(p.as_ref(), options) } else { scan(p.as_ref(), profile) }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s11_4_rules_recognise_names_uids_and_dates() {
        assert!(is_person_name("Doe^Jane"));
        assert!(is_person_name("Müller^Hans"));
        assert!(is_person_name("山田^太郎"));
        assert!(!is_person_name("1Doe^Jane"));
        assert!(!is_person_name("Doe=^Jane"));
        assert!(is_dicom_uid("1.2.840.10008"));
        assert!(is_dicom_uid("1.2.840.10008\n"));
        assert!(!is_dicom_uid("1.2.3"));
        assert_eq!(uid_tokens("dicom:1.2.840.10008.5"), vec![(6, 21)]);
        assert!(uid_tokens("v1.2.3.4.").is_empty());
        assert!(has_date("taken 2020-01-31 at noon"));
        assert!(is_dicom_date("20200131"));
        assert!(is_dicom_date("20200131\n"));
    }

    #[test]
    fn s11_4_a_date_time_is_a_date() {
        assert!(has_date("2020-01-31T10:22:00"));
        assert!(has_date("scanned 2020-01-31T10:22:00Z at the north wing"));
        assert!(has_date("CT_2020-01-31.nii.gz"));
        assert!(has_date("20200131102200.123456+0100"));
        assert!(has_date("20200131T102200"));
        assert!(!has_date("12020-01-311"));
        assert!(!has_date("2020-13-31"));
    }

    #[test]
    fn s11_4_an_ontology_code_is_not_a_date() {
        assert!(!has_date("52988006"));
        assert!(!is_dicom_date("20201301"));
    }

    #[test]
    fn s11_4_digits_are_ascii_and_never_panic() {
        // Devanagari digits: `\d` matched them, and the shift then sliced a
        // multi-byte character in two.
        for text in ["२०२००१३१", "२०२०-०१-३१", "१.२.८४०.१०००८"] {
            assert!(!has_date(text), "{text}");
            assert!(!is_dicom_uid(text), "{text}");
            assert!(uid_tokens(text).is_empty(), "{text}");
            assert_eq!(shift(text, Some(5)), None, "{text}");
        }
        assert!(!has_date("२०२०-01-31"));
    }

    #[test]
    fn s11_4_dates_shift_or_drop() {
        assert_eq!(shift("20200131", Some(30)).as_deref(), Some("20200301"));
        assert_eq!(shift(" on 2020-02-28 ", Some(1)).as_deref(), Some("on 2020-02-29"));
        assert_eq!(shift("20201301", Some(1)), None);
        assert_eq!(shift("20200131", None), None);
    }

    #[test]
    fn s11_4_a_shift_keeps_the_time_and_moves_every_date() {
        assert_eq!(shift("2020-01-31T10:22:00Z", Some(10)).as_deref(), Some("2020-02-10T10:22:00Z"));
        assert_eq!(shift("20200131102200.5+0100", Some(1)).as_deref(), Some("20200201102200.5+0100"));
        assert_eq!(shift("2020-01-31 and 2020-02-03", Some(-31)).as_deref(), Some("2019-12-31 and 2020-01-03"));
        assert_eq!(shift("9999-12-31", Some(1)), None);
    }

    /// A lookup compares `normalise(key)`, so an entry it would change could
    /// never match: `nhs_number`, `date_of_birth` and three UID keys did not.
    #[test]
    fn s11_4_every_key_table_entry_is_stored_normalised() {
        for table in [&IDENTIFYING_KEYS[..], &QUASI_IDENTIFYING_KEYS, &DATE_KEYS, &UID_KEYS] {
            for entry in table {
                assert_eq!(*entry, normalise(entry), "{entry} can never match a key");
            }
            let distinct: std::collections::HashSet<&&str> = table.iter().collect();
            assert_eq!(distinct.len(), table.len(), "a table lists an entry twice");
        }
        for (key, table) in [
            ("nhs_number", &IDENTIFYING_KEYS[..]),
            ("Patient_Name", &IDENTIFYING_KEYS),
            ("date_of_birth", &DATE_KEYS),
            ("birth_date", &DATE_KEYS),
            ("Study_UID", &UID_KEYS),
            ("series uid", &UID_KEYS),
            ("frame_uid", &UID_KEYS),
        ] {
            assert!(table.contains(&normalise(key).as_str()), "{key}");
        }
    }

    #[test]
    fn s11_4_a_value_that_is_wholly_a_date_is_told_from_text_holding_one() {
        for whole in [
            "2020-01-31",
            "2020-01-31T10:22:00",
            "2020-01-31 10:22:00.123456",
            "2020-01-31T10:22Z",
            "2020-01-31T10:22:00+01:00",
            "20200131",
            "20200131102200",
            "20200131102200.5+0100",
            " 20200131 ",
            "2020-01-31\n",
        ] {
            assert!(is_whole_date(whole), "{whole:?}");
        }
        for text in [
            "scanned 2020-01-31",
            "2020-01-31 and 2020-02-03",
            "CT_2020-01-31.nii",
            "52988006",
            "2020-13-31",
            "2020-02-30",
            "2020-01-31T",
            "२०२०-०१-३१",
        ] {
            assert!(!is_whole_date(text), "{text:?}");
        }
        assert_eq!(day_number("2020-01-31T10:22:00"), day_number("20200131"));
        assert_eq!(day_number("1970-01-02"), Some(1));
        assert_eq!(day_number("scanned 2020-01-31"), None);
    }

    #[test]
    fn s11_4_a_replaced_id_is_found_as_a_token_of_its_own() {
        assert_eq!(id_tokens("case MRN-42.nii", "MRN-42"), vec![(5, 11)]);
        assert_eq!(id_tokens("MRN-42", "MRN-42"), vec![(0, 6)]);
        assert_eq!(id_tokens("MRN-42_CT and MRN-42", "MRN-42"), vec![(0, 6), (14, 20)]);
        assert!(id_tokens("MRN-421", "MRN-42").is_empty());
        assert!(id_tokens("xMRN-42", "MRN-42").is_empty());
        assert!(id_tokens("Müller-MRN-42ä", "MRN-42").is_empty());
    }

    #[test]
    fn s11_4_a_recorded_shift_of_nothing_is_no_shift() {
        assert_eq!(recorded_shift(Some(&json!(0))), None);
        assert_eq!(recorded_shift(Some(&json!(0.0))), None);
        assert_eq!(recorded_shift(Some(&json!(-117))), Some(-117));
        assert_eq!(recorded_shift(Some(&Value::Null)), None);
        assert_eq!(recorded_shift(None), None);
    }

    /// Shifted dates less their intervals name one baseline; a date added
    /// after the shift names another.  Within a day is agreement: a date
    /// without its time is up to a day off its interval.
    #[test]
    fn s11_4_a_date_added_after_a_shift_disagrees_with_the_shifted_ones() {
        let doc = json!({"timepoints": [
            {"id": "tp0", "index": 0, "date": "2020-01-01"},
            {"id": "tp1", "index": 1, "date": "2020-01-11", "days_from_baseline": 10},
            {"id": "tp2", "index": 2, "date": "2020-04-30", "days_from_baseline": 2},
            {"id": "tp3", "index": 3, "date": "2020-01-21T23:59:00", "days_from_baseline": 20.9},
            {"id": "tp4", "index": 4, "date": "2021-06-01"},
        ]});
        let found = disagreeing(doc.as_object().unwrap(), -117);
        assert_eq!(found.keys().copied().collect::<Vec<_>>(), vec![2]);
        assert!(found[&2].contains("118 day(s) off what timepoints[0].date"), "{}", found[&2]);
        assert!(found[&2].contains("a shift of -117 days"), "{}", found[&2]);
    }

    #[test]
    fn s11_4_strings_are_found_however_deep_a_type_holds_them() {
        use hdf5::types::{CompoundField, CompoundType, IntSize, TypeDescriptor as TD};
        let field = |name: &str, ty: TD| CompoundField { name: name.into(), ty, offset: 0, index: 0 };
        let compound = |fields: Vec<CompoundField>| TD::Compound(CompoundType { fields, size: 0 });
        let nested = compound(vec![field("outer", compound(vec![field("name", TD::VarLenUnicode)]))]);
        assert!(matches!(holds_strings(&nested, 0), Inside::Strings));
        let names = compound(vec![field("names", TD::FixedArray(Box::new(TD::VarLenUnicode), 2))]);
        assert!(matches!(holds_strings(&names, 0), Inside::Strings));
        assert!(matches!(holds_strings(&TD::VarLenArray(Box::new(TD::FixedAscii(8))), 0), Inside::Strings));
        let numbers = compound(vec![field("n", TD::Integer(IntSize::U4)), field("v", TD::Unsigned(IntSize::U1))]);
        assert!(matches!(holds_strings(&numbers, 0), Inside::Nothing));
        let mut deep = TD::Integer(IntSize::U4);
        for _ in 0..=MAX_DEPTH {
            deep = TD::VarLenArray(Box::new(deep));
        }
        assert!(matches!(holds_strings(&deep, 0), Inside::TooDeep), "past the cap, reported rather than passed");
    }

    /// Every link is reached and each group expanded once: a later hard
    /// link, a soft link and a committed type each hold a name `ops::visit`
    /// never yields, and a hard link back to an ancestor ends the walk.
    #[test]
    fn s11_4_the_name_walk_reaches_every_link_and_ends() {
        let dir = tempfile::tempdir().unwrap();
        let file = hdf5::File::create(dir.path().join("links.h5")).unwrap();
        let a = file.create_group("x_a").unwrap();
        a.new_dataset::<u8>().shape([2]).create("table").unwrap();
        file.create_group("x_b").unwrap().link_hard("/x_a/table", "Doe^Jane").unwrap();
        a.link_soft("/x_a/table", "Roe^Rick").unwrap();
        a.link_hard("/x_a", "loop").unwrap();
        a.commit_datatype("rowtype", &hdf5::Datatype::from_type::<i32>().unwrap()).unwrap();
        let reached = links(&file).unwrap();
        let paths: Vec<&str> = reached.iter().map(|l| l.path.as_str()).collect();
        assert_eq!(paths, ["x_a", "x_a/Roe^Rick", "x_a/loop", "x_a/rowtype", "x_a/table", "x_b", "x_b/Doe^Jane"]);
        let first: Vec<&str> = reached.iter().filter(|l| l.object.is_some()).map(|l| l.path.as_str()).collect();
        assert_eq!(first, ["x_a", "x_a/rowtype", "x_a/table", "x_b"], "each object once, as ops::visit");
        assert!(matches!(reached[3].object, Some(Object::Datatype(_))));
        assert_eq!(reached[1].target.as_deref(), Some("/x_a/table"));
        assert_eq!(reached[6].id, reached[4].id, "the later name is the same object");
        assert_eq!(reached[2].id, reached[0].id, "the loop is the group it is in");
    }

    #[test]
    fn pseudonyms_are_stable_and_salted() {
        let a = pseudonymise("1.2.3.4", "");
        assert_eq!(a, pseudonymise("1.2.3.4", ""));
        assert!(a.starts_with(PSEUDONYM_PREFIX) && a.len() == PSEUDONYM_PREFIX.len() + 32);
        assert_ne!(a, pseudonymise("1.2.3.4", "salt"));
    }
}
