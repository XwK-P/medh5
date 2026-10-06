//! Hash algorithms and digest strings (spec §2.1, §13).
//!
//! A digest string is `"<algo>:<hex>"` with `algo` one of `sha256` (the
//! default), `sha512` or `blake2b` (512-bit, as Python's `hashlib.new` makes
//! it).  Any other algorithm is E703, normatively.

use blake2::Blake2b512;
use digest::Digest;
use sha2::{Sha256, Sha512};

use crate::json::{repr_list, repr_str};
use crate::{Error, Result};

/// The default digest algorithm.
pub const DEFAULT_ALGO: &str = "sha256";
/// The algorithms §2.1 names.
pub const DIGEST_ALGOS: [&str; 3] = ["sha256", "sha512", "blake2b"];

/// An incremental hash under one of the §2.1 algorithms.
pub enum Hasher {
    Sha256(Sha256),
    Sha512(Sha512),
    Blake2b(Blake2b512),
}

impl Hasher {
    /// A fresh hasher, or E703 for an algorithm outside §2.1.
    pub fn new(algo: &str) -> Result<Self> {
        match algo {
            "sha256" => Ok(Hasher::Sha256(Sha256::new())),
            "sha512" => Ok(Hasher::Sha512(Sha512::new())),
            "blake2b" => Ok(Hasher::Blake2b(Blake2b512::new())),
            other => Err(Error::coded(
                "E703",
                format!(
                    "unsupported digest algorithm {}; expected one of {}",
                    repr_str(other),
                    repr_list(&DIGEST_ALGOS)
                ),
            )),
        }
    }

    /// Feed bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        match self {
            Hasher::Sha256(h) => h.update(bytes),
            Hasher::Sha512(h) => h.update(bytes),
            Hasher::Blake2b(h) => h.update(bytes),
        }
    }

    /// The algorithm's name.
    pub fn algo(&self) -> &'static str {
        match self {
            Hasher::Sha256(_) => "sha256",
            Hasher::Sha512(_) => "sha512",
            Hasher::Blake2b(_) => "blake2b",
        }
    }

    /// The lowercase hex digest.
    pub fn hexdigest(self) -> String {
        match self {
            Hasher::Sha256(h) => hex::encode(h.finalize()),
            Hasher::Sha512(h) => hex::encode(h.finalize()),
            Hasher::Blake2b(h) => hex::encode(h.finalize()),
        }
    }

    /// `"<algo>:<hex>"`.
    pub fn finish(self) -> String {
        let algo = self.algo();
        format!("{algo}:{}", self.hexdigest())
    }
}

/// The bare hex digest of `payload`.
pub fn hash_hex(algo: &str, payload: &[u8]) -> Result<String> {
    let mut h = Hasher::new(algo)?;
    h.update(payload);
    Ok(h.hexdigest())
}

/// `"<algo>:<hex>"` over a byte string.
pub fn digest_bytes(payload: &[u8], algo: &str) -> Result<String> {
    let mut h = Hasher::new(algo)?;
    h.update(payload);
    Ok(h.finish())
}

/// Split `"<algo>:<hex>"`, raising E703 when malformed.
pub fn parse_digest(value: &str) -> Result<(String, String)> {
    let malformed = || Error::coded("E703", format!("malformed digest {}", repr_str(value)));
    let (algo, hexdigest) = value.split_once(':').ok_or_else(malformed)?;
    if hexdigest.is_empty() || !DIGEST_ALGOS.contains(&algo) {
        return Err(malformed());
    }
    // `int(hex, 16)` accepts an optional sign, a `0x` prefix and underscores
    // between digits; a digest is plain hex, which is all that matters here.
    let body = hexdigest.trim();
    let body = body.strip_prefix(['+', '-']).unwrap_or(body);
    let body = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")).unwrap_or(body);
    if body.is_empty()
        || body.starts_with('_')
        || body.ends_with('_')
        || body.contains("__")
        || !body.chars().all(|c| c.is_ascii_hexdigit() || c == '_')
    {
        return Err(malformed());
    }
    Ok((algo.to_string(), hexdigest.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digests() {
        assert_eq!(
            digest_bytes(b"abc", "sha256").unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(hash_hex("blake2b", b"").unwrap().len(), 128);
        assert_eq!(Hasher::new("blake3").err().unwrap().code(), Some("E703"));
        assert!(parse_digest("sha256:00ff").is_ok());
        assert!(parse_digest("sha256:").is_err());
        assert!(parse_digest("md5:00").is_err());
        assert!(parse_digest("sha256:xyz").is_err());
    }
}
