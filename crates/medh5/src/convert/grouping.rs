//! Grouping study-scoped sources into subject-scoped samples (§3.7).
//!
//! Identity comes from a **declared key**, never a filename or a date; when
//! it cannot be established --- or the source contradicts it --- each study
//! becomes its own sample.  Timepoint order comes from a date when there is
//! one and is reported as a guess otherwise.  Instance correspondence across
//! merged studies is never inferred (§7.4).

use std::collections::{BTreeMap, BTreeSet};

use indexmap::IndexMap;
use serde_json::{json, Value};

use super::report::ConversionReport;
use crate::json::repr_str;
use crate::{Error, Result};

/// The subject prefix of a study that could not be grouped.
pub const FALLBACK_PREFIX: &str = "study";

/// One study/visit of one subject, before it becomes a timepoint.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Occasion {
    /// The source's own identifier --- a StudyInstanceUID, a file path.
    pub key: String,
    pub subject_id: Option<String>,
    pub date: Option<String>,
    /// A last-resort ordering value (an mtime); using it is a guess.
    pub order_hint: Option<f64>,
    /// Facts no visit can change (`PatientBirthDate`, `PatientSex`).
    pub demographics: IndexMap<String, String>,
    /// The caller's handle on its own payload (an index into its list).
    pub payload: usize,
}

/// Occasions belonging to one subject, in timepoint order.
#[derive(Debug, Clone, PartialEq)]
pub struct SubjectGroup {
    pub subject_id: String,
    pub occasions: Vec<Occasion>,
    /// `date`, `given` or `order_hint` --- the last is a guess.
    pub ordered_by: String,
}

impl SubjectGroup {
    pub fn new(subject_id: &str) -> SubjectGroup {
        SubjectGroup { subject_id: subject_id.into(), occasions: Vec::new(), ordered_by: "date".into() }
    }

    pub fn is_longitudinal(&self) -> bool {
        self.occasions.len() > 1
    }

    pub fn timepoint_ids(&self) -> Vec<String> {
        (0..self.occasions.len()).map(|i| format!("tp{i}")).collect()
    }

    /// Intervals in days, or `None` where a date is missing.
    pub fn days_from_baseline(&self) -> Vec<Option<i64>> {
        let dates: Vec<Option<i64>> = self.occasions.iter().map(|o| parse_date(o.date.as_deref())).collect();
        match dates.first() {
            Some(Some(base)) => dates.iter().map(|d| d.map(|v| v - base)).collect(),
            _ => vec![None; dates.len()],
        }
    }

    pub fn to_json(&self) -> Value {
        json!({
            "subject_id": self.subject_id,
            "ordered_by": self.ordered_by,
            "occasions": self.occasions.iter().map(|o| o.key.clone()).collect::<Vec<_>>(),
            "days_from_baseline": self.days_from_baseline(),
        })
    }
}

/// Days since 1970-01-01 of a valid proleptic Gregorian date.
pub fn days_from_civil(y: i64, m: u32, d: u32) -> Option<i64> {
    if !(1..=12).contains(&m) || d == 0 || !(1..=9999).contains(&y) {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let max = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][(m - 1) as usize];
    if d > max {
        return None;
    }
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + doe - 719_468)
}

fn digits(s: &str) -> Option<i64> {
    if s.is_empty() || !s.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Parse a DICOM `YYYYMMDD` or an ISO `YYYY-MM-DD`; `None` when unparseable.
pub fn parse_date(value: Option<&str>) -> Option<i64> {
    let text = value?.trim();
    if text.is_empty() {
        return None;
    }
    // Sliced by byte below, so only an ASCII head is a candidate: a short
    // head holding a multibyte character used to split it, and panic.
    let head: String = text.chars().take(8).collect();
    if head.len() == 8 && head.is_ascii() {
        if let (Some(y), Some(m), Some(d)) = (digits(&head[..4]), digits(&head[4..6]), digits(&head[6..8])) {
            if let Some(days) = days_from_civil(y, m as u32, d as u32) {
                return Some(days);
            }
        }
    }
    let iso: String = text.chars().take(10).collect();
    if iso.len() == 10 && iso.is_ascii() && &iso[4..5] == "-" && &iso[7..8] == "-" {
        if let (Some(y), Some(m), Some(d)) = (digits(&iso[..4]), digits(&iso[5..7]), digits(&iso[8..10])) {
            return days_from_civil(y, m as u32, d as u32);
        }
    }
    None
}

/// Demographics on which `occasions` disagree: `{field: [values]}`.
pub fn contradictions(occasions: &[&Occasion]) -> BTreeMap<String, Vec<String>> {
    let mut seen: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for o in occasions {
        for (name, value) in &o.demographics {
            let text = value.trim().to_uppercase();
            if !text.is_empty() {
                seen.entry(name.clone()).or_default().insert(text);
            }
        }
    }
    seen.into_iter().filter(|(_, v)| v.len() > 1).map(|(k, v)| (k, v.into_iter().collect())).collect()
}

fn contradicted(occasions: &[Occasion], log: &mut Option<&mut ConversionReport>) -> BTreeSet<String> {
    let mut by_subject: IndexMap<String, Vec<&Occasion>> = IndexMap::new();
    for o in occasions {
        if let Some(s) = o.subject_id.as_ref().filter(|s| !s.is_empty()) {
            by_subject.entry(s.clone()).or_default().push(o);
        }
    }
    let mut out = BTreeSet::new();
    for (subject, members) in by_subject {
        let conflicts = contradictions(&members);
        if conflicts.is_empty() {
            continue;
        }
        out.insert(subject.clone());
        if let Some(log) = log.as_deref_mut() {
            let fields: Vec<&String> = conflicts.keys().collect();
            log.guess(
                "identity",
                format!(
                    "subject key {} names {} studies whose {} disagree, so they are not one person on the evidence; each \
                     study was given its own subject ({FALLBACK_PREFIX}:<study>) rather than grouped under the key",
                    repr_str(&subject),
                    members.len(),
                    fields.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" and ")
                ),
                json!({
                    "subject": subject,
                    "conflicts": conflicts,
                    "occasions": members.iter().map(|o| o.key.clone()).collect::<Vec<_>>(),
                }),
            );
        }
    }
    out
}

/// Group occasions into subjects (`subject`) or leave them apart (`study`).
pub fn group_by_subject(
    occasions: Vec<Occasion>,
    mode: &str,
    mut report: Option<&mut ConversionReport>,
) -> Result<Vec<SubjectGroup>> {
    if mode != "subject" && mode != "study" {
        return Err(Error::Value(format!("unknown grouping mode {}", repr_str(mode))));
    }
    let contradicted = contradicted(&occasions, &mut report);
    let subject_of = |o: &Occasion| -> Option<String> {
        match &o.subject_id {
            Some(s) if !s.is_empty() && !contradicted.contains(s) => Some(s.clone()),
            _ => None,
        }
    };
    if mode == "study" {
        return Ok(occasions
            .into_iter()
            .map(|o| SubjectGroup {
                subject_id: subject_of(&o).unwrap_or_else(|| format!("{FALLBACK_PREFIX}:{}", o.key)),
                occasions: vec![o],
                ordered_by: "given".into(),
            })
            .collect());
    }
    let unidentified_count = occasions.iter().filter(|o| o.subject_id.as_deref().unwrap_or("").is_empty()).count();
    if unidentified_count > 0 {
        if let Some(log) = report.as_deref_mut() {
            let inputs: Vec<String> = occasions
                .iter()
                .filter(|o| o.subject_id.as_deref().unwrap_or("").is_empty())
                .take(20)
                .map(|o| o.key.clone())
                .collect();
            log.warn(
                "grouping",
                format!(
                    "{unidentified_count} input(s) carry no subject key, so each became its own sample; identity is never \
                     inferred from filenames or dates"
                ),
                json!({"inputs": inputs}),
            );
        }
    }
    let mut identified = Vec::new();
    let mut unidentified = Vec::new();
    let mut contradicted_list = Vec::new();
    for o in occasions {
        if subject_of(&o).is_some() {
            identified.push(o);
        } else if o.subject_id.as_deref().unwrap_or("").is_empty() {
            unidentified.push(o);
        } else {
            contradicted_list.push(o);
        }
    }
    unidentified.extend(contradicted_list);
    let mut groups: IndexMap<String, SubjectGroup> = IndexMap::new();
    for o in identified {
        let s = o.subject_id.clone().unwrap_or_default();
        groups.entry(s.clone()).or_insert_with(|| SubjectGroup::new(&s)).occasions.push(o);
    }
    for o in unidentified {
        let key = format!("{FALLBACK_PREFIX}:{}", o.key);
        groups.insert(key.clone(), SubjectGroup { subject_id: key, occasions: vec![o], ordered_by: "date".into() });
    }
    let mut out: Vec<SubjectGroup> = groups.into_values().collect();
    for g in out.iter_mut() {
        order(g, &mut report);
    }
    out.sort_by(|a, b| a.subject_id.cmp(&b.subject_id));
    Ok(out)
}

fn order(group: &mut SubjectGroup, log: &mut Option<&mut ConversionReport>) {
    if group.occasions.len() < 2 {
        group.ordered_by = "given".into();
        return;
    }
    if group.occasions.iter().all(|o| parse_date(o.date.as_deref()).is_some()) {
        group.occasions.sort_by_key(|o| parse_date(o.date.as_deref()));
        group.ordered_by = "date".into();
        return;
    }
    if group.occasions.iter().all(|o| o.order_hint.is_some()) {
        group.occasions.sort_by(|a, b| {
            a.order_hint.unwrap_or(0.0).partial_cmp(&b.order_hint.unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal)
        });
        group.ordered_by = "order_hint".into();
        if let Some(log) = log.as_deref_mut() {
            log.guess(
                "timepoint_order",
                format!(
                    "subject {} has no study dates; its {} visits were ordered by a file timestamp, which is a heuristic \
                     and not evidence",
                    repr_str(&group.subject_id),
                    group.occasions.len()
                ),
                json!({"subject": group.subject_id, "occasions": group.occasions.iter().map(|o| o.key.clone()).collect::<Vec<_>>()}),
            );
        }
        return;
    }
    group.ordered_by = "given".into();
    if let Some(log) = log.as_deref_mut() {
        log.guess(
            "timepoint_order",
            format!(
                "subject {} has neither dates nor timestamps; its visits kept the order they were supplied in",
                repr_str(&group.subject_id)
            ),
            json!({"subject": group.subject_id}),
        );
    }
}

fn is_uid(key: &str) -> bool {
    let parts: Vec<&str> = key.split('.').collect();
    parts.len() >= 2 && parts.iter().all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))
}

/// An occasion key as a file-name stem: a path's stem, a UID whole.
pub fn key_stem(key: &str) -> String {
    if is_uid(key) {
        return key.to_string();
    }
    let name = std::path::Path::new(key).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match name.rfind('.') {
        Some(i) if i > 0 => name[..i].to_string(),
        _ => name,
    }
}

/// Whether `group` is a study [`group_by_subject`] gave no subject: its
/// subject id is the fallback `study:<key>` rather than one the source named.
pub(crate) fn is_fallback(group: &SubjectGroup) -> bool {
    match group.occasions.as_slice() {
        [only] => group.subject_id == format!("{FALLBACK_PREFIX}:{}", only.key),
        _ => false,
    }
}

/// A unique filename stem for one group.
///
/// A study with no subject is named for its source; a subject is named for
/// itself, `Study-7` included, which a test of the stem's first letters took
/// for the fallback.
pub fn output_name(group: &SubjectGroup, used: &mut BTreeSet<String>, safe: &dyn Fn(&str) -> String) -> String {
    let mut base = safe(&group.subject_id);
    if is_fallback(group) {
        if let Some(first) = group.occasions.first() {
            let stem = safe(&key_stem(&first.key));
            if !stem.is_empty() {
                base = stem;
            }
        }
    }
    if used.insert(base.clone()) {
        return base;
    }
    if let Some(first) = group.occasions.first() {
        let candidate = format!("{base}_{}", safe(&key_stem(&first.key)));
        if used.insert(candidate.clone()) {
            return candidate;
        }
    }
    let mut index = 2;
    while used.contains(&format!("{base}_{index}")) {
        index += 1;
    }
    let name = format!("{base}_{index}");
    used.insert(name.clone());
    name
}

/// Record that objects were *not* joined across merged studies (§7.4).
pub fn note_instance_ids(group: &SubjectGroup, log: &mut ConversionReport) {
    if group.is_longitudinal() {
        log.decision(
            "instance_ids",
            format!(
                "subject {} merged {} studies; each study's objects kept independent instance ids, because asserting \
                 correspondence across visits would fabricate tracking ground truth",
                repr_str(&group.subject_id),
                group.occasions.len()
            ),
            json!({"subject": group.subject_id, "occasions": group.occasions.len()}),
        );
    }
}

/// A label-set `key` from free text (§5.2): `^[a-z0-9][a-z0-9_]*$`.
pub fn sanitize_key(name: &str, fallback: &str) -> String {
    let lowered = name.trim().to_lowercase();
    let cleaned: String = lowered
        .chars()
        .map(|c| if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' { c } else { '_' })
        .collect();
    // What is left of a name starts with a letter or digit, so only a name
    // with nothing left needs the fallback.
    let cleaned = cleaned.trim_matches('_');
    let cleaned = if cleaned.is_empty() { fallback } else { cleaned };
    cleaned.chars().take(128).collect()
}

/// `U+00C0..=U+017F` once NFKD has dropped their combining marks: the base
/// letter, or `_` where NFKD keeps the letter whole (`Æ`, `ß`, `Ø`, `Ł`).
/// The five that decompose to two characters are matched before the table.
const LATIN: &[u8; 192] = b"AAAAAA_CEEEEIIII_NOOOOO__UUUUY__aaaaaa_ceeeeiiii_nooooo__uuuuy_y\
AaAaAaCcCcCcCcDd__EeEeEeEeEeGgGgGgGgHh__IiIiIiIiI___JjKk_LlLlLl____NnNnNn___OoOoOo__RrRrRrSsSsSsSsTtTt__UuUuUuUuUuUuWwYyYZzZzZzs";

/// Text folded to ASCII, as `medh5.io._common.sanitize_ascii_stem` folds it
/// with NFKD: Latin letters lose their accents (`Müller` is `Muller`, written
/// precomposed or not), the fullwidth and superscript forms of ASCII are
/// ASCII, and any other character outside ASCII is `_`.
fn ascii_fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            _ if c.is_ascii() => out.push(c),
            // Combining marks, as NFKD leaves them and the fold drops them.
            '\u{300}'..='\u{36f}'
            | '\u{1ab0}'..='\u{1aff}'
            | '\u{1dc0}'..='\u{1dff}'
            | '\u{20d0}'..='\u{20ff}'
            | '\u{fe20}'..='\u{fe2f}' => {}
            '\u{ff01}'..='\u{ff5e}' => out.push(char::from_u32(c as u32 - 0xfee0).unwrap_or('_')),
            'ª' => out.push('a'),
            'º' => out.push('o'),
            '¹' => out.push('1'),
            '²' => out.push('2'),
            '³' => out.push('3'),
            'Ĳ' => out.push_str("IJ"),
            'ĳ' => out.push_str("ij"),
            'Ŀ' => out.push_str("L_"),
            'ŀ' => out.push_str("l_"),
            'ŉ' => out.push_str("_n"),
            '\u{c0}'..='\u{17f}' => out.push(char::from(LATIN[c as usize - 0xc0])),
            _ => out.push('_'),
        }
    }
    out
}

/// A filename stem from free text: ASCII identifier characters, truncated.
///
/// A sample id is `[A-Za-z0-9_.-]` (§2.2, E003), and the letters this kept
/// outside ASCII made an id the writer refused: a subject key `Müller`
/// failed its whole group.
pub fn sanitize_stem(text: &str, limit: usize) -> String {
    ascii_fold(text)
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' })
        .take(limit)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_and_keys() {
        assert_eq!(parse_date(Some("20200105")), Some(days_from_civil(2020, 1, 5).unwrap()));
        assert_eq!(parse_date(Some("2020-01-05T10:00")), Some(days_from_civil(2020, 1, 5).unwrap()));
        assert_eq!(parse_date(Some("20200230")), None);
        assert_eq!(days_from_civil(1970, 1, 1), Some(0));
        assert_eq!(sanitize_key("GTV-1", "class"), "gtv_1");
        assert_eq!(sanitize_key("__", "class"), "class");
        assert_eq!(key_stem("1.2.840.113"), "1.2.840.113");
        assert_eq!(key_stem("/data/case_01.medh5"), "case_01");
    }

    #[test]
    fn s2_2_a_stem_is_an_ascii_identifier() {
        assert_eq!(sanitize_stem("Müller Hans", 128), "Muller_Hans");
        // Decomposed, as a macOS file name is.
        assert_eq!(sanitize_stem("Mu\u{308}ller", 128), "Muller");
        assert_eq!(sanitize_stem("Łódź ß Ĳssel", 128), "_odz___IJssel");
        assert_eq!(sanitize_stem("ＡＢＣ１２", 128), "ABC12");
        assert_eq!(sanitize_stem("患者 7", 128), "___7");
        assert_eq!(sanitize_stem("a b/c", 4), "a_b_");
        assert!(LATIN.iter().all(|b| b.is_ascii_alphabetic() || *b == b'_'));
        assert_eq!(sanitize_stem("Ÿ", 128), "Y");
    }

    #[test]
    fn s3_7_only_a_fallback_is_named_for_its_source() {
        let group = |subject: &str, key: &str| SubjectGroup {
            subject_id: subject.into(),
            occasions: vec![Occasion { key: key.into(), ..Default::default() }],
            ordered_by: "given".into(),
        };
        let safe = |s: &str| sanitize_stem(s, 128).to_lowercase();
        let mut used = BTreeSet::new();
        assert_eq!(output_name(&group("Study-7", "/x/s0.medh5"), &mut used, &safe), "study-7");
        assert_eq!(output_name(&group("STUDY001", "/x/s1.medh5"), &mut used, &safe), "study001");
        assert_eq!(output_name(&group("study:/x/s2.medh5", "/x/s2.medh5"), &mut used, &safe), "s2");
    }

    #[test]
    fn s3_7_a_non_ascii_date_is_unparseable_not_a_panic() {
        // Eight or ten bytes, fewer characters: the byte slices split one.
        for text in ["202é001", "202é-01-0", "２０２０-01-05"] {
            assert_eq!(parse_date(Some(text)), None, "{text}");
        }
        let o = |key: &str, date: &str| Occasion {
            key: key.into(),
            subject_id: Some("P".into()),
            date: Some(date.into()),
            ..Default::default()
        };
        let groups = group_by_subject(vec![o("a", "202é001"), o("b", "20200105")], "subject", None).unwrap();
        assert_eq!(groups[0].occasions.len(), 2);
    }
}
