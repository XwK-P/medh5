//! `medh5 scrub` --- finding identifiers in a container, and attesting to it.
//!
//! **What this cannot do, stated first, because the attestation depends on
//! it.**  Scrubbing inspects metadata.  It does not look at voxels, so it
//! cannot see burned-in text or a face reconstructible from a head CT; the
//! record it writes says exactly what was and was not checked (§11.4).
//!
//! **Where it looks: everywhere a string can be** --- every string in
//! `/meta`, every object name, every attribute and every string dataset,
//! including unknown ones.  The scan and `--apply` are one traversal, so a
//! finding the scan calls actionable is, by construction, one the clean acts
//! on.  UIDs are **pseudonymised**, not deleted: a UID is how two files agree
//! on a frame of reference, so it becomes a stable hash of itself.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

use indexmap::IndexMap;
use regex::Regex;
use serde_json::{json, Map, Value};

use crate::array::NdArray;
use crate::curation::identity::{ID_SOURCE, PSEUDONYM_SOURCE};
use crate::document::SampleDocument;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::{py_float, py_str, repr_list, repr_str};
use crate::sample::writer::amend;
use crate::sample::{frame_references, open_sample, repack, FRAME_ATTRS};
use crate::{Error, Result};

pub const PROFILES: [&str; 2] = ["basic", "strict"];

/// Attributes to remove, from the DICOM PS3.15 E.1 basic profile, matched
/// case-, space- and underscore-insensitively.
pub const IDENTIFYING_KEYS: [&str; 78] = [
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
    "patient_name",
    "patient_id",
    "mrn",
    "nhs_number",
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

pub const DATE_KEYS: [&str; 16] = [
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
    "study_date",
    "acquisition_date",
    "birth_date",
    "date_of_birth",
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
    "study_uid",
    "series_uid",
    "frame_uid",
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

/// Root attributes `commit` rewrites from the amended state.
const ROOT_MANAGED_ATTRS: [&str; 7] =
    ["medh5_version", "medh5_kind", "medh5_profiles", "created", "generator", "digest_algo", "content_id"];

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

fn iso_date() -> &'static Regex {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex(&CELL, r"\b\d{4}-\d{2}-\d{2}\b")
}

/// Python's `re.match` of a `^...$` pattern: `$` also matches before one
/// trailing newline.
fn anchored(re: &Regex, text: &str) -> bool {
    re.is_match(text) || text.strip_suffix('\n').is_some_and(|t| re.is_match(t))
}

fn is_dicom_date(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    anchored(regex(&CELL, r"^\d{8}$"), text)
}

/// A dotted-numeric OID.  `pseudo:...` and UUIDs do not match, by design.
fn is_dicom_uid(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    anchored(regex(&CELL, r"^\d+(\.\d+){3,}$"), text)
}

fn has_date(text: &str) -> bool {
    iso_date().is_match(text) || is_dicom_date(text)
}

/// DICOM PN form `Family^Given`, in any script.
fn is_person_name(text: &str) -> bool {
    static CELL: OnceLock<Regex> = OnceLock::new();
    regex(&CELL, r"^[\p{L}\p{Nl}\p{No}][^\^=\p{Nd}]*\^[\p{L}\p{Nl}\p{No}]").is_match(text)
}

/// Dotted-numeric UIDs *inside* a longer string: `(?<![\d.])\d+(?:\.\d+){3,}(?![\d.])`.
///
/// The look-arounds pin a match to a whole run of digits and dots, so a run
/// matches exactly when all of it is UID-shaped.
fn uid_tokens(text: &str) -> Vec<(usize, usize)> {
    static RUN: OnceLock<Regex> = OnceLock::new();
    static WHOLE: OnceLock<Regex> = OnceLock::new();
    let whole = regex(&WHOLE, r"^\d+(?:\.\d+){3,}$");
    regex(&RUN, r"[\d.]+")
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

/// One walk over every string in a sample, for scanning and for cleaning.
struct Sweep<'r> {
    report: &'r mut ScrubReport,
    writing: bool,
    strict: bool,
    salt: String,
    date_shift_days: Option<i64>,
    dates_shifted: bool,
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
            dates_shifted: false,
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
                return Kept::Drop;
            }
            return Kept::Same;
        }
        if UID_KEYS.contains(&normal.as_str()) && !is_dicom_uid(&text) {
            self.find(
                "uid",
                path,
                "a UID attribute whose value is not a UID; it cannot be pseudonymised safely, so a person must decide",
                Some(&text),
                false,
                review,
            );
            return Kept::Same;
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
                return Kept::Drop;
            }
            return Kept::Same;
        }
        if DATE_KEYS.contains(&normal.as_str()) {
            // A file that already records a shift has had its dates handled.
            let actionable = !self.dates_shifted;
            self.find("date", path, "a date attribute copied from the source", Some(&text), actionable, review);
            if self.dates_shifted || !self.acts(review) {
                return Kept::Same;
            }
            return match shift(&text, self.date_shift_days) {
                None => {
                    self.actions.push(format!("{path} removed"));
                    Kept::Drop
                }
                Some(moved) => {
                    self.actions.push(format!("{path} shifted"));
                    Kept::Value(Value::String(moved))
                }
            };
        }
        match value {
            Slot::Json(v) => {
                let walked = self.json(v, path, depth + 1, review);
                if &walked == v {
                    Kept::Same
                } else {
                    Kept::Value(walked)
                }
            }
            Slot::Opaque(_) => Kept::Same,
        }
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
        Some(key.to_string())
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
        if has_date(value) {
            self.find("date", r#where, "contains what looks like a date", Some(value), false, review);
            return value.to_string();
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
        value.to_string()
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
                let mut out = String::new();
                let mut last = 0;
                for (start, end) in uid_tokens(&cleaned) {
                    out.push_str(&cleaned[last..start]);
                    let token = cleaned[start..end].to_string();
                    out.push_str(&self.pseudonym(&token));
                    last = end;
                }
                out.push_str(&cleaned[last..]);
                cleaned = out;
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
            let (scheme, rest) = split_reference(&cleaned);
            // A name in the file name is still a name.
            let base = self.text(&basename(&rest), r#where, 0, None, "");
            if self.writing {
                cleaned = format!("{scheme}{}", if self.strict { PATH_REMOVED.to_string() } else { base });
                self.actions.push(format!("{}: path {}", r#where, if self.strict { "removed" } else { "reduced" }));
            }
        } else if !has_uid && !rest.is_empty() {
            let (scheme, rest) = split_reference(value);
            cleaned = format!("{scheme}{}", self.text(&rest, r#where, 0, None, ""));
        }
        cleaned
    }

    // -- the sample document ---------------------------------------------------

    /// Every string in `/meta`; the cleaned document when writing.
    fn document(&mut self, doc: &Map<String, Value>) -> Map<String, Value> {
        let record = doc.get("deidentification").and_then(Value::as_object);
        self.dates_shifted = record.and_then(|r| r.get("date_shift_days")).is_some_and(|v| !v.is_null());
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
                    self.name(&Value::String(key.clone()), &path);
                    cleaned.insert(key.clone(), self.json(value, &path, 0, None));
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
        // `deidentification` is not scanned: it is the attestation, and
        // `--apply` replaces it whole.
        out
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
                    if !crate::json::py_truthy(value) || self.dates_shifted {
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
                out.insert("name".into(), Value::String(pseudonymise(&name, &self.salt)));
                self.actions.push(format!("{}.name pseudonymised", r#where));
            }
        } else if !name.is_empty() {
            // The format requires a name, so one that reads as a person's
            // becomes a pseudonym rather than blank.
            let removed = pseudonymise(&name, &self.salt);
            let cleaned = self.text(&name, &format!("{}.name", r#where), 0, None, &removed);
            out.insert("name".into(), Value::String(cleaned));
        }
        for (key, value) in agent {
            if ["id", "type", "name", "organization"].contains(&key.as_str()) {
                continue;
            }
            out.insert(key.clone(), self.json(value, &format!("{}.{key}", r#where), 0, None));
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

    /// Every object name, attribute and string dataset outside `/meta`.
    fn hdf5(&mut self, root: &hdf5::Group) -> Result<()> {
        self.attributes(root, "root", &ROOT_MANAGED_ATTRS)?;
        let mut names: Vec<(String, ops::Node)> = Vec::new();
        ops::visit(root, &mut |name, node| {
            let owned = match node {
                ops::Node::Group(g) => ops::Node::Group(g.clone()),
                ops::Node::Dataset(d) => ops::Node::Dataset(d.clone()),
            };
            names.push((name.to_string(), owned));
            Ok(true)
        })?;
        for (name, node) in names {
            let r#where = name.replace('/', ".");
            let leaf = name.rsplit('/').next().unwrap_or(&name).to_string();
            self.name(&Value::String(leaf), &r#where);
            let parts: Vec<&str> = name.split('/').collect();
            // The frame graph is pseudonymised as one mapping (`scan_frames`).
            let skip: Vec<&str> = if parts.len() == 2 {
                FRAME_ATTRS.iter().find(|(g, _)| *g == parts[0]).map(|(_, a)| a.to_vec()).unwrap_or_default()
            } else {
                Vec::new()
            };
            match &node {
                ops::Node::Group(g) => self.attributes(g, &r#where, &skip)?,
                ops::Node::Dataset(d) => {
                    self.attributes(d, &r#where, &skip)?;
                    if name != "meta" {
                        self.dataset(d, &r#where)?;
                    }
                }
            }
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

    fn dataset(&mut self, ds: &hdf5::Dataset, r#where: &str) -> Result<()> {
        match data::kind(ds)? {
            data::Kind::Strings => {}
            data::Kind::Other(_) if compound_with_strings(ds) => {
                self.find(
                    "unreadable",
                    r#where,
                    "a compound dataset with string fields; this tool does not decode compound types, so a person \
                     must look at it",
                    None,
                    false,
                    None,
                );
                return Ok(());
            }
            _ => return Ok(()),
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
                return Ok(());
            }
        };
        let kept: Vec<String> =
            values.iter().enumerate().map(|(i, v)| self.text(v, &format!("{}[{i}]", r#where), 0, None, "")).collect();
        if self.writing && kept != values {
            rewrite_strings(ds, &kept)?;
        }
        Ok(())
    }
}

fn compound_with_strings(ds: &hdf5::Dataset) -> bool {
    use hdf5::types::TypeDescriptor as TD;
    match ds.dtype().and_then(|t| t.to_descriptor()) {
        Ok(TD::Compound(c)) => c
            .fields
            .iter()
            .any(|f| matches!(f.ty, TD::VarLenUnicode | TD::VarLenAscii | TD::FixedAscii(_) | TD::FixedUnicode(_))),
        _ => false,
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
        sweep.document(&doc);
        scan_frames(&frame_references(&sample.root)?, sweep.report);
        sweep.hdf5(&sample.root)?;
    }
    scan_file_name(path, &document.identity.sample_id, &document.identity.subject_id, &mut report);
    escalate(&mut report);
    Ok(report)
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
    let mut report = scan(path, profile)?;
    let strict = profile == "strict";
    let mut replaced: Vec<(String, String)> = Vec::new();
    let mut writer = amend(path, None)?;
    let result = (|| -> Result<()> {
        let mut date_shift_days = options.date_shift_days;
        let mut notes = Vec::new();
        if let Some(already) = writer.document().deidentification.clone() {
            if let Some(days) = &already.date_shift_days {
                // Shifting twice makes the recorded offset a lie.
                date_shift_days = days.as_i64().or_else(|| days.as_f64().map(|f| f as i64));
                notes.push(format!(
                    "dates were left alone: already shifted by {} days",
                    py_str(&Value::Number(days.clone()))
                ));
            }
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
        let mut cleaned = sweep.document(&doc);
        if options.pseudonymise_ids {
            replaced = pseudonymise_ids(&mut cleaned, &options.salt, &mut sweep.uid_map, &mut sweep.actions);
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
        let left = rescan(&writer, profile)?;
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
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    // HDF5 does not reclaim superseded storage, so the original values would
    // survive in freed space; compacting is part of de-identifying.
    repack(path)?;
    // The claim is checked against the file, not against the intention.
    report.remaining = scan(path, profile)?.findings;
    let stems = stems(path);
    let name_of_file = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    for (name, old) in &replaced {
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

/// Every rule over a writer's state, without going back to disk, so the count
/// recorded in the attestation describes the file it ships in.
fn rescan(writer: &crate::sample::writer::SampleWriter, profile: &str) -> Result<ScrubReport> {
    let mut report = ScrubReport::new("<amend>", profile);
    let doc = match writer.document().to_json() {
        Value::Object(m) => m,
        _ => Map::new(),
    };
    {
        let mut sweep = Sweep::new(&mut report);
        sweep.document(&doc);
        scan_frames(&writer.frame_uids()?, sweep.report);
        sweep.hdf5(&writer.root()?)?;
    }
    escalate(&mut report);
    Ok(report)
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

/// Shift a date by `days`, or `None` meaning "drop it".
fn shift(value: &str, days: Option<i64>) -> Option<String> {
    let days = days?;
    let text = value.trim();
    let digits = |s: &str| -> Option<i64> { s.parse::<i64>().ok() };
    if is_dicom_date(text) {
        let (y, m, d) = (digits(&text[..4])?, digits(&text[4..6])? as u32, digits(&text[6..8])? as u32);
        let (ny, nm, nd) = moved(y, m, d, days)?;
        return Some(format!("{ny:04}{nm:02}{nd:02}"));
    }
    if let Some(found) = iso_date().find(text) {
        let s = found.as_str();
        let (y, m, d) = (digits(&s[..4])?, digits(&s[5..7])? as u32, digits(&s[8..10])? as u32);
        let (ny, nm, nd) = moved(y, m, d, days)?;
        return Some(text.replace(s, &format!("{ny:04}-{nm:02}-{nd:02}")));
    }
    None
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
    }

    #[test]
    fn s11_4_dates_shift_or_drop() {
        assert_eq!(shift("20200131", Some(30)).as_deref(), Some("20200301"));
        assert_eq!(shift(" on 2020-02-28 ", Some(1)).as_deref(), Some("on 2020-02-29"));
        assert_eq!(shift("20201301", Some(1)), None);
        assert_eq!(shift("20200131", None), None);
    }

    #[test]
    fn pseudonyms_are_stable_and_salted() {
        let a = pseudonymise("1.2.3.4", "");
        assert_eq!(a, pseudonymise("1.2.3.4", ""));
        assert!(a.starts_with(PSEUDONYM_PREFIX) && a.len() == PSEUDONYM_PREFIX.len() + 32);
        assert_ne!(a, pseudonymise("1.2.3.4", "salt"));
    }
}
