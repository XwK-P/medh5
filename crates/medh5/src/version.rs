//! Format versions: what this engine reads, validates and writes (1.1 §2).
//!
//! `medh5_version` is `MAJOR.MINOR`.  This engine implements **1.0 and 1.1**,
//! and keeps three capabilities apart (1.1 §2.2):
//!
//! - **reading** --- any 1.x file opens; a minor above 1.1 is read as its
//!   *supported projection*, the objects this engine knows;
//! - **validating** --- 1.0 and 1.1 fully; a higher minor only as a projection,
//!   reported as such (W913), never as conformance;
//! - **amending** --- only what it can preserve: a higher minor, or a profile
//!   this engine does not implement, is refused before anything is written.
//!
//! A writer emits the *lowest* version its content needs: 1.0 for an imaging
//! sample, 1.1 once it carries the `clinical` profile.  Migrating a 1.0 sample
//! to 1.1 changes its `content_id`, because the version is part of what that
//! address covers (1.0 §13.2).

use crate::{Error, Result};

/// The format versions this engine implements, oldest first.
pub const FORMAT_VERSIONS: [&str; 2] = ["1.0", "1.1"];
/// The version an imaging-only sample is written in.
pub const BASE_VERSION: &str = "1.0";
/// The newest version this engine implements (`medh5::FORMAT_VERSION`).
pub const LATEST_VERSION: &str = "1.1";

/// What this engine can do with a file of a given version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Support {
    /// A version this engine implements: read, validate and amend.
    Full,
    /// A higher minor of the major this engine implements: readable as a
    /// projection, validated only as one, never amended.
    Projection,
    /// Another major, or no version at all: refused.
    Unsupported,
}

/// `(major, minor)` of a version string: `MAJOR.MINOR`, each a run of ASCII
/// digits; a bare major reads as minor 0.  Nothing else parses: whitespace
/// and signs are no part of a version (1.0 §2.1).
pub fn parse(version: &str) -> Option<(u32, u32)> {
    let number = |text: &str| if text.bytes().all(|b| b.is_ascii_digit()) { text.parse().ok() } else { None };
    let (major, minor) = version.split_once('.').unwrap_or((version, "0"));
    Some((number(major)?, number(minor)?))
}

/// The support a file of `version` gets.  The major is the text before the
/// first `.`, compared exactly: ` 1` or `01` is no major this engine
/// implements.  The readers refuse the file this calls unsupported
/// ([`crate::sample::require_major`]), so the two cannot disagree.
pub fn support(version: &str) -> Support {
    if version.split('.').next() != LATEST_VERSION.split('.').next() {
        return Support::Unsupported;
    }
    match (parse(version), parse(LATEST_VERSION)) {
        (Some((_, minor)), Some((_, latest))) if minor <= latest => Support::Full,
        // A higher minor, or one this engine cannot parse (`1.x`, `1.0 `):
        // still major 1, read as what this engine knows.
        _ => Support::Projection,
    }
}

/// Whether a file of `version` is read as a projection (a higher minor).
pub fn is_projection(version: &str) -> bool {
    support(version) == Support::Projection
}

/// Whether `version` is at least `floor` (both `MAJOR.MINOR`).
pub fn at_least(version: &str, floor: &str) -> bool {
    match (parse(version), parse(floor)) {
        (Some(a), Some(b)) => a >= b,
        _ => false,
    }
}

/// The later of two versions (the first when either does not parse).
pub fn later<'a>(a: &'a str, b: &'a str) -> &'a str {
    match (parse(a), parse(b)) {
        (Some(x), Some(y)) if y > x => b,
        _ => a,
    }
}

/// The lowest version that can declare every one of `profiles`.
pub fn required_for<S: AsRef<str>>(profiles: &[S]) -> &'static str {
    if profiles.iter().any(|p| p.as_ref() == crate::clinical::PROFILE) {
        crate::clinical::MIN_VERSION
    } else {
        BASE_VERSION
    }
}

/// Refuse to amend a file this engine cannot preserve (1.1 §2.3).
///
/// A higher minor may have added objects, references and attestation this
/// engine cannot see; copying them through and recomputing `content_id` under
/// the rules it does know would make a no-op amend lossy, or forge an address
/// for content it never checked.  So it refuses before anything is written.
pub fn require_amendable(version: &str, profiles: &[String], what: &str) -> Result<()> {
    match support(version) {
        Support::Full => {}
        Support::Projection => {
            return Err(Error::Version(format!(
                "{what} is MEDH5 {version}; this engine implements up to {LATEST_VERSION} and can read it only as a \
                 projection --- amending it would drop or misattest what a later minor defines, so it is refused \
                 (1.1 §2.3)"
            )))
        }
        Support::Unsupported => {
            return Err(Error::Version(format!("{what} is MEDH5 {version}; this engine implements {LATEST_VERSION}")))
        }
    }
    let mut unknown: Vec<&String> =
        profiles.iter().filter(|p| !crate::sample::PROFILES.contains(&p.as_str())).collect();
    unknown.sort();
    unknown.dedup();
    if !unknown.is_empty() {
        return Err(Error::coded(
            "E007",
            format!(
                "{what} declares profile(s) {} this engine does not implement; amending it could break their \
                 requirements unseen, so it is refused (1.1 §2.3)",
                crate::json::repr_list(&unknown)
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s2_2_versions_are_full_projection_or_unsupported() {
        assert_eq!(support("1.0"), Support::Full);
        assert_eq!(support("1.1"), Support::Full);
        assert_eq!(support("1"), Support::Full);
        assert_eq!(support("1.2"), Support::Projection);
        assert_eq!(support("1.10"), Support::Projection);
        assert_eq!(support("1.x"), Support::Projection);
        assert_eq!(support("2.0"), Support::Unsupported);
        assert_eq!(support(""), Support::Unsupported);
        assert!(at_least("1.1", "1.1") && at_least("1.2", "1.1") && !at_least("1.0", "1.1"));
        assert_eq!(later("1.0", "1.1"), "1.1");
        assert_eq!(later("1.1", "1.0"), "1.1");
        assert_eq!(required_for(&["core", "seg"]), "1.0");
        assert_eq!(required_for(&["core", "clinical"]), "1.1");
    }

    /// `parse` trimmed, and the readers compare the major as text: a file
    /// of ` 1.0` was fully supported, amendable and validated, and no
    /// reader would open it.
    #[test]
    fn s2_1_whitespace_is_no_part_of_a_version() {
        // `01` is the number 1, but not the major the readers name.
        for refused in [" 1.0", "01.0", "+1.0", "1 .0", "\t1.1"] {
            assert_eq!(support(refused), Support::Unsupported, "{refused:?}");
        }
        // Major 1 with a minor that is no number: a projection, as `1.x`.
        for unknown in ["1.0 ", "1. 0", "1.+1", "1.0\n"] {
            assert_eq!(support(unknown), Support::Projection, "{unknown:?}");
        }
        for unparsed in [" 1.0", "+1.0", "1 .0", "1.0 ", "1. 0", "1.+1", "", "1."] {
            assert_eq!(parse(unparsed), None, "{unparsed:?}");
        }
        assert_eq!(parse("1.10"), Some((1, 10)));
        assert_eq!(parse("1"), Some((1, 0)));
    }

    #[test]
    fn s2_3_a_higher_minor_or_unknown_profile_is_not_amendable() {
        assert!(require_amendable("1.1", &["core".into(), "clinical".into()], "x").is_ok());
        assert!(matches!(require_amendable("1.2", &["core".into()], "x"), Err(Error::Version(_))));
        let err = require_amendable("1.0", &["core".into(), "quantum".into()], "x").unwrap_err();
        assert_eq!(err.code(), Some("E007"));
    }
}
