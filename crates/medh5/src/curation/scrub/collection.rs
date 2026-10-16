//! A collection under `medh5 scrub` (§2.2, D-E8): scanned and scrubbed
//! member by member.
//!
//! **What is read.**  The collection, read-only: a scan opens each member
//! where it is, and `--apply` copies each member out, as `unpack` does, into
//! a directory of its own beside the result, and scrubs the copy by the
//! sample rules.  The source is never written.
//!
//! **What `--apply` writes.**  A new collection, never the source rewritten:
//! each member's scrubbed root packed under its key --- the pseudonym of its
//! sample id under `--pseudonymise-ids`, which a member key often is, else
//! the key it had --- each with the `content_id` its own scrub computed.  One
//! salt keys every pseudonym, every re-minted id and every shift a range
//! takes, so members of one subject agree as separate files scrubbed with
//! that salt do.  A member that refuses fails the whole collection, and
//! nothing is written; under `--skip-refused` it is left out and listed.
//!
//! **What the report says.**  One report for the collection: each member's
//! findings and actions under `KEY::`, its source key, and what a re-scan of
//! the written collection still finds under the key it was written as.

use std::path::{Path, PathBuf};

use super::{apply_with, scan_sample, Finding, ScrubOptions, ScrubReport};
use crate::collection::{is_collection, open_collection, pack, write_sample_root, SUFFIX};
use crate::h5::file::{open_read, same_file};
use crate::json::{repr_list, repr_str};
use crate::{Error, Result};

/// The rule a member `--skip-refused` left out is reported under.
pub const REFUSED_MEMBER: &str = "refused_member";
/// Where a member's key is reported: the collection's name for a sample,
/// as a file name is a lone sample's.
const MEMBER_KEY: &str = "member key";

/// Whether `path` is a collection (§2.2).  A file that does not open is not
/// one: the sample path opens it again, and says why.
pub(super) fn names_collection(path: &Path) -> bool {
    open_read(path).is_ok_and(|handle| is_collection(&handle).unwrap_or(false))
}

/// The part of a merged report's location inside its member: what the
/// sample rules compare (a member key holds no `:`, §2.2).
pub(super) fn in_member(location: &str) -> &str {
    location.split_once("::").map_or(location, |(_, inside)| inside)
}

/// `text` with every spelling of the copy's path replaced by the member's
/// name: a report or a refusal names the member, never the scratch copy.
fn renamed(text: &str, copy: &Path, shown: &str) -> String {
    let raw = copy.to_string_lossy();
    text.replace(&repr_str(&raw), &repr_str(shown)).replace(raw.as_ref(), shown)
}

/// A `file_name` finding about a member, which the scan reads from a name
/// `KEY.medh5`, said of the member's key.
fn keyed(mut finding: Finding, key: &str) -> Finding {
    if finding.rule == "file_name" {
        finding.r#where = MEMBER_KEY.into();
        finding.value = Some(key.into());
        finding.detail = finding
            .detail
            .replace("the file name contains", "the member key contains")
            .replace("the file is still named after", "the member is still keyed by")
            .replace("the file is named after", "the member is keyed by")
            .replace(
                "rename it before sharing --- medh5 does not rename files",
                "--apply --pseudonymise-ids --salt keys each member by its pseudonym",
            )
            .replace(", and a later scan cannot tell the name was an id", "");
    }
    finding
}

/// One member's report, merged into the collection's.
struct Member {
    /// Its key in the source.
    key: String,
    /// Its key in the written collection, once applied.
    written_as: Option<String>,
    report: ScrubReport,
}

/// The collection's report: every member's, its locations under `KEY::`.
fn merged(
    path: &str,
    profile: &str,
    applied: bool,
    members: Vec<Member>,
    skipped: Vec<(String, String)>,
) -> ScrubReport {
    let mut report = ScrubReport::new(path, profile);
    report.applied = applied;
    report.not_checked.clear();
    let under = |key: &str, finding: Finding| Finding { r#where: format!("{key}::{}", finding.r#where), ..finding };
    for member in members {
        let Member { key, written_as, report: one } = member;
        report.findings.extend(one.findings.into_iter().map(|f| under(&key, keyed(f, &key))));
        report.actions.extend(one.actions.into_iter().map(|a| format!("{key}::{a}")));
        if let Some(new) = written_as.as_deref() {
            if new != key {
                report.actions.push(format!("{key} written as {new}"));
            }
            report.remaining.extend(one.remaining.into_iter().map(|f| under(new, keyed(f, new))));
        }
        for (old, new) in one.uid_map {
            report.uid_map.insert(old, new);
        }
        for note in one.not_checked {
            if !report.not_checked.contains(&note) {
                report.not_checked.push(note);
            }
        }
    }
    for (key, why) in skipped {
        report.add(
            REFUSED_MEMBER,
            &key,
            &format!("left out of the new collection (--skip-refused): {why}"),
            None,
            false,
            false,
        );
        report.actions.push(format!("{key} left out: refused"));
    }
    report
}

/// Find identifiers in every member of a collection.  Changes nothing.
pub(super) fn scan(path: &Path, profile: &str) -> Result<ScrubReport> {
    super::check_profile(profile)?;
    let collection = open_collection(path)?;
    let shown = path.to_string_lossy().into_owned();
    let mut members = Vec::new();
    for key in collection.keys()? {
        let sample = collection.get(&key)?;
        let named = PathBuf::from(format!("{key}.medh5"));
        let (report, _) = scan_sample(&sample, &named, &format!("{shown}::{key}"), profile, &[])?;
        members.push(Member { key, written_as: None, report });
    }
    Ok(merged(&shown, profile, false, members, Vec::new()))
}

/// A directory beside the result for the members' copies, removed with
/// everything in it when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn beside(out: &Path) -> Result<Scratch> {
        let name = out.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let parent = match out.parent() {
            Some(p) if !p.as_os_str().is_empty() => p.to_path_buf(),
            _ => PathBuf::from("."),
        };
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
        for attempt in 0..64u32 {
            let dir = parent.join(format!(".{name}.scrub-{}-{:x}", std::process::id(), stamp + u128::from(attempt)));
            match std::fs::create_dir(&dir) {
                Ok(()) => return Ok(Scratch(dir)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(crate::error::os_error(&e, &dir)),
            }
        }
        Err(Error::Io(format!("no free name for a scratch directory beside {}", repr_str(&out.to_string_lossy()))))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Where a collection's scrub writes: `--out`, or the source's name in
/// `--out-dir`.  Never the source, and never over a file.
fn output(path: &Path, options: &ScrubOptions) -> Result<PathBuf> {
    let shown = repr_str(&path.to_string_lossy());
    let out = match (&options.out, &options.out_dir) {
        (Some(out), _) => out.clone(),
        (None, Some(dir)) => dir.join(path.file_name().unwrap_or_default()),
        (None, None) => {
            return Err(Error::invalid(format!(
                "{shown} is a collection, which a scrub writes as a new collection and never rewrites: give --out \
                 NEW{SUFFIX} (out) or --out-dir (out_dir)"
            )))
        }
    };
    let same = same_file(&out, path)
        || matches!((std::fs::canonicalize(&out), std::fs::canonicalize(path)), (Ok(a), Ok(b)) if a == b);
    if same || out == path {
        return Err(Error::invalid(format!(
            "{} is the collection being scrubbed: a scrub writes a new collection and leaves the source as it is",
            repr_str(&out.to_string_lossy())
        )));
    }
    if out.exists() {
        return Err(Error::invalid(format!(
            "{} exists; a scrub to a copy writes a new file and does not overwrite one",
            repr_str(&out.to_string_lossy())
        )));
    }
    Ok(out)
}

/// Scrub every member of the collection at `path` into a new collection.
pub(super) fn apply(path: &Path, options: &ScrubOptions) -> Result<ScrubReport> {
    let out = output(path, options)?;
    let shown = path.to_string_lossy().into_owned();
    let collection = open_collection(path)?;
    let keys = collection.keys()?;
    if keys.is_empty() {
        return Err(Error::coded("E008", format!("{} holds no member to scrub", repr_str(&shown))));
    }
    let scratch = Scratch::beside(&out)?;
    let group = collection.group()?;
    let mut members: Vec<Member> = Vec::new();
    let mut written: Vec<PathBuf> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for (i, key) in keys.iter().enumerate() {
        let member = format!("{shown}::{key}");
        // Each in a directory of its own, so two results cannot share a name
        // before they are told apart below.
        let dir = scratch.0.join(i.to_string());
        std::fs::create_dir(&dir).map_err(|e| crate::error::os_error(&e, &dir))?;
        let copy = dir.join(format!("{key}.medh5"));
        write_sample_root(&group.group(key)?, &copy)?;
        let mut one = options.clone().with_out_dir(dir.join("out"));
        one.out = None;
        std::fs::create_dir(dir.join("out")).map_err(|e| crate::error::os_error(&e, &dir))?;
        match apply_with(&copy, &one) {
            Ok(mut report) => {
                let result = PathBuf::from(&report.path);
                let new = result.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                report.path = format!("{}::{new}", out.to_string_lossy());
                written.push(result);
                members.push(Member { key: key.clone(), written_as: Some(new), report });
            }
            Err(e) => {
                let why = renamed(&e.to_string(), &copy, &member);
                if !options.skip_refused {
                    let said = format!(
                        "member {} of {} was refused, so nothing was written (--skip-refused leaves it out): {}",
                        repr_str(key),
                        repr_str(&shown),
                        renamed(e.message(), &copy, &member)
                    );
                    return Err(with_message(e, said));
                }
                skipped.push((key.clone(), why));
            }
        }
    }
    if written.is_empty() {
        return Err(Error::invalid(format!(
            "every member of {} was refused, so there is no collection to write: {}",
            repr_str(&shown),
            skipped.iter().map(|(k, why)| format!("{k}: {why}")).collect::<Vec<_>>().join("; ")
        )));
    }
    let new_keys: Vec<String> = members.iter().filter_map(|m| m.written_as.clone()).collect();
    let mut seen: std::collections::BTreeMap<&str, &str> = std::collections::BTreeMap::new();
    for (member, new) in members.iter().zip(&new_keys) {
        if let Some(first) = seen.insert(new, &member.key) {
            return Err(Error::coded(
                "E003",
                format!(
                    "members {} of {} would both be written as {}: they hold one sample id; nothing was written",
                    repr_list(&[first, member.key.as_str()]),
                    repr_str(&shown),
                    repr_str(new)
                ),
            ));
        }
    }
    drop(group);
    drop(collection);
    let sources: Vec<&Path> = written.iter().map(PathBuf::as_path).collect();
    pack(&sources, &out, Some(&new_keys))?;
    // Each member's own re-scan is of the root packed here, byte for byte
    // (`pack` refuses a copy that changes its `content_id`), and knew the
    // ids its run replaced, which a scan of the collection cannot.
    let report = merged(&out.to_string_lossy(), options.profile(), true, members, skipped);
    drop(scratch);
    Ok(report)
}

/// `error` with its message replaced, its kind and code kept.
fn with_message(error: Error, message: String) -> Error {
    match error {
        Error::File(_) => Error::File(message),
        Error::Version(_) => Error::Version(message),
        Error::Schema(_) => Error::Schema(message),
        Error::Validation { code, .. } => Error::Validation { message, code },
        Error::Integrity(_) => Error::Integrity(message),
        Error::Key(_) => Error::Key(message),
        Error::Value(_) => Error::Value(message),
        Error::Type(_) => Error::Type(message),
        Error::Index(_) => Error::Index(message),
        Error::Io(_) => Error::Io(message),
        Error::NotImplemented(_) => Error::NotImplemented(message),
        Error::Runtime(_) => Error::Runtime(message),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_2_a_member_location_is_read_inside_its_member() {
        assert_eq!(in_member("k1::identity.sample_id"), "identity.sample_id");
        assert_eq!(in_member("identity.sample_id"), "identity.sample_id");
        assert_eq!(in_member("k.1::extra.a::b"), "extra.a::b");
    }

    #[test]
    fn s2_2_a_refusal_names_the_member_never_the_copy() {
        let copy = Path::new("/out/.x.medh5c.scrub-1-2/0/k1.medh5");
        let text = format!("{} holds clinical text; see {}", repr_str(&copy.to_string_lossy()), copy.display());
        let said = renamed(&text, copy, "c.medh5c::k1");
        assert_eq!(said, "'c.medh5c::k1' holds clinical text; see c.medh5c::k1");
        let e = with_message(Error::coded("E809", "x"), "y".into());
        assert_eq!((e.code(), e.message()), (Some("E809"), "y"));
        let e = with_message(Error::Io("gone".into()), "member: gone".into());
        assert_eq!(e.python_line(), "OSError: member: gone");
    }

    #[test]
    fn s2_2_a_file_name_finding_is_said_of_the_member_key() {
        let finding = Finding {
            rule: "file_name".into(),
            r#where: "file name".into(),
            detail: "the file is named after identity.sample_id, which is reported above; rename it before sharing \
                     --- medh5 does not rename files"
                .into(),
            value: Some("MRN-1.medh5".into()),
            actionable: false,
            fixable: true,
        };
        let keyed = keyed(finding, "MRN-1");
        assert_eq!((keyed.r#where.as_str(), keyed.value.as_deref()), (MEMBER_KEY, Some("MRN-1")));
        assert!(keyed.detail.starts_with("the member is keyed by identity.sample_id"));
        assert!(!keyed.detail.contains("rename"), "{}", keyed.detail);
    }
}
