//! The normative diagnostic code table (spec §15.2).
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

use std::sync::OnceLock;

use serde::Deserialize;

const TABLE_JSON: &str = include_str!("../data/codes.json");

/// One diagnostic code.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Code {
    /// `E001` ... `W912`.
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

/// Look up one code.
pub fn get(code: &str) -> Option<&'static Code> {
    all().iter().find(|c| c.code == code)
}

/// Look up one code, as an error naming the unknown code.
pub fn code(name: &str) -> crate::Result<&'static Code> {
    get(name).ok_or_else(|| crate::Error::Key(format!("unknown diagnostic code {name:?}")))
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
    fn table_is_complete_and_well_formed() {
        let codes = all();
        assert_eq!(codes.len(), 94);
        for c in codes {
            let expected = if c.code.starts_with('W') { "warning" } else { "error" };
            assert_eq!(c.severity, expected, "{}", c.code);
            assert!(DOMAINS.contains(&c.domain.as_str()), "{}", c.code);
        }
        let mut names: Vec<_> = codes.iter().map(|c| c.code.as_str()).collect();
        names.dedup();
        assert_eq!(names.len(), codes.len());
    }
}
