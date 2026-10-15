//! The normative diagnostic code table (spec 1.0 §15.2, extended by 1.1 §11.2).
//!
//! Codes are **stable API**: a code's meaning never changes, codes are never
//! reused, and third-party validators are expected to emit the same code for
//! the same defect.  The table lives in `data/codes.json` --- the one source
//! the validator, the conformance corpus, every frontend and the documentation
//! site all read --- and is embedded here at compile time.
//!
//! Codes are grouped by domain:
//!
//! ```text
//! E0xx  container      E1xx  geometry     E2xx  images
//! E3xx  label set      E4xx  annotations  E5xx  transforms
//! E6xx  curation       E7xx  integrity    E8xx  clinical (1.1)
//! W9xx  warnings
//! ```
//!
//! The table is allocated in numbered *revisions* (§15.2): revision 1 is every
//! code allocated before revisions were numbered, and each later allocation,
//! by a minor version or an erratum, is the next number.  Each code names the
//! version whose rules it reports ([`since`]) and the revision that allocated
//! it ([`revision`]); a validator pinned to a revision reports no later code
//! ([`crate::validate::ValidateOptions::codes_revision`]).

use std::sync::OnceLock;

use serde::Deserialize;

const TABLE_JSON: &str = include_str!("../data/codes.json");

/// One diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Code {
    /// `E001` ... `W914`.
    pub code: String,
    /// `error` or `warning`.
    pub severity: String,
    /// The spec domain the code belongs to.
    pub domain: String,
    /// One-line meaning.
    pub summary: String,
}

#[derive(Deserialize)]
struct Table {
    codes: Vec<Code>,
}

/// What `Code` does not carry: it is an exhaustive struct, so the members
/// revisions added are read beside it.
#[derive(Deserialize)]
struct Allocation {
    code: String,
    since: String,
    revision: u32,
}

#[derive(Deserialize)]
struct Allocations {
    revision: u32,
    codes: Vec<Allocation>,
}

/// The domains, in the order the specification lists them.
pub const DOMAINS: [&str; 9] =
    ["container", "geometry", "images", "labels", "annotations", "transforms", "curation", "integrity", "clinical"];

/// Every code, in table order.
pub fn all() -> &'static [Code] {
    static TABLE: OnceLock<Vec<Code>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let table: Table = serde_json::from_str(TABLE_JSON).expect("data/codes.json is valid by construction");
        table.codes
    })
}

fn allocations() -> &'static Allocations {
    static ALLOCATIONS: OnceLock<Allocations> = OnceLock::new();
    ALLOCATIONS.get_or_init(|| serde_json::from_str(TABLE_JSON).expect("data/codes.json is valid by construction"))
}

/// The revision of the table this engine carries (§15.2): the latest
/// revision any of its codes was allocated in.
///
/// A function, not a constant: the number lives in `data/codes.json`, which
/// a `const` cannot read.
pub fn table_revision() -> u32 {
    allocations().revision
}

/// The format version whose rules `code` reports (`"1.0"` or `"1.1"`), or
/// `None` for a code the table does not hold.
pub fn since(code: &str) -> Option<&'static str> {
    allocations().codes.iter().find(|c| c.code == code).map(|c| c.since.as_str())
}

/// The revision of the table that allocated `code`, or `None` for a code the
/// table does not hold.
pub fn revision(code: &str) -> Option<u32> {
    allocations().codes.iter().find(|c| c.code == code).map(|c| c.revision)
}

/// Look up one code.
pub fn get(code: &str) -> Option<&'static Code> {
    all().iter().find(|c| c.code == code)
}

/// Look up one code, as an error naming the unknown code.
pub fn code(name: &str) -> crate::Result<&'static Code> {
    get(name).ok_or_else(|| crate::Error::Key(format!("unknown diagnostic code {}", crate::json::repr_str(name))))
}

/// Every code in one domain, in table order.
pub fn for_domain(domain: &str) -> Vec<&'static Code> {
    all().iter().filter(|c| c.domain == domain).collect()
}

/// The raw table, for publishing next to a conformance suite.
pub fn table_json() -> &'static str {
    TABLE_JSON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unknown_code_is_quoted_as_python_quotes_it() {
        assert_eq!(code("E101").unwrap().code, "E101");
        assert_eq!(code("X999").unwrap_err().message(), "unknown diagnostic code 'X999'");
        assert_eq!(code("it's").unwrap_err().message(), "unknown diagnostic code \"it's\"");
    }

    #[test]
    fn table_is_complete_and_well_formed() {
        let codes = all();
        assert_eq!(codes.len(), 100);
        for c in codes {
            let expected = if c.code.starts_with('W') { "warning" } else { "error" };
            assert_eq!(c.severity, expected, "{}", c.code);
            assert!(DOMAINS.contains(&c.domain.as_str()), "{}", c.code);
        }
        let names: std::collections::BTreeSet<&str> = codes.iter().map(|c| c.code.as_str()).collect();
        assert_eq!(names.len(), codes.len(), "a code is listed twice");
    }

    #[test]
    fn s15_2_every_code_has_a_version_and_a_revision() {
        let mut latest = 0;
        for c in all() {
            let version = since(&c.code).unwrap_or_else(|| panic!("{} names no version", c.code));
            assert!(["1.0", "1.1"].contains(&version), "{}: {version}", c.code);
            let allocated = revision(&c.code).unwrap_or_else(|| panic!("{} names no revision", c.code));
            assert!((1..=table_revision()).contains(&allocated), "{}: revision {allocated}", c.code);
            latest = latest.max(allocated);
        }
        // The table's revision is the latest allocation: no number is skipped
        // at the top, and no code is ahead of the table that holds it.
        assert_eq!(latest, table_revision());
        assert_eq!((since("E999"), revision("E999")), (None, None));
    }
}
