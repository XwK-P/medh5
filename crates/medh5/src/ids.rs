//! Identifier rules (spec §2.2, §2.3).
//!
//! Object identifiers --- grid, image, annotation and transform ids, timepoint
//! ids --- match `[A-Za-z0-9_.-]{1,128}` and never take a reserved name.  Sample
//! keys in a collection match `[A-Za-z0-9_.-]{1,255}`.

use crate::json::repr_str;
use crate::{Error, Result};

/// Names an identifier must not take (spec §2.3).
pub const RESERVED_IDS: [&str; 1] = ["meta"];

fn is_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'
}

/// Whether `name` matches `[A-Za-z0-9_.-]{1,max_len}`.
pub fn matches_pattern(name: &str, max_len: usize) -> bool {
    !name.is_empty() && name.len() <= max_len && name.chars().all(is_id_char)
}

/// Whether `name` matches the object-identifier pattern.
pub fn is_valid_id(name: &str) -> bool {
    matches_pattern(name, 128)
}

/// Check an object identifier against spec §2.3 and return it unchanged.
pub fn validate_id<'a>(name: &'a str, what: &str) -> Result<&'a str> {
    if !is_valid_id(name) {
        return Err(Error::coded("E003", format!("{what} {} must match [A-Za-z0-9_.-]{{1,128}}", repr_str(name))));
    }
    if RESERVED_IDS.contains(&name) {
        return Err(Error::coded("E003", format!("{what} {} is reserved", repr_str(name))));
    }
    Ok(name)
}

/// Check a collection sample key against spec §2.2 and return it unchanged.
pub fn validate_sample_key(name: &str) -> Result<&str> {
    if !matches_pattern(name, 255) {
        return Err(Error::coded("E003", format!("sample key {} must match [A-Za-z0-9_.-]{{1,255}}", repr_str(name))));
    }
    Ok(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers() {
        assert!(validate_id("ct_tp0", "grid id").is_ok());
        assert!(validate_id("a.b-c_1", "grid id").is_ok());
        let err = validate_id("bad id", "grid id").unwrap_err();
        assert_eq!(err.code(), Some("E003"));
        assert_eq!(err.to_string(), "[E003] grid id 'bad id' must match [A-Za-z0-9_.-]{1,128}");
        assert_eq!(validate_id("meta", "image id").unwrap_err().message(), "image id 'meta' is reserved");
        assert!(validate_id(&"x".repeat(129), "id").is_err());
        assert!(validate_sample_key(&"x".repeat(255)).is_ok());
    }
}
