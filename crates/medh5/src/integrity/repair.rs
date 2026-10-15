//! `medh5 fix` --- rebuilding what is derived, restamping what is claimed.
//!
//! **Rebuilding an index** recomputes a cache from the data it caches
//! (§14.3); a stale index is a performance bug and rebuilding it is repair.
//!
//! **Rewriting digests** is not repair: a digest that no longer matches is
//! evidence that the bytes changed, and recomputing it destroys the evidence.
//! So it needs a reason, which is recorded as a provenance activity naming
//! what was restamped and that the content was *not* verified.

use std::path::Path;

use serde_json::{json, Map, Value};

use super::{restated, stale_index_entries, stamp_digests, verify_root};
use crate::geometry::grid::read_grid;
use crate::h5::{attrs, ops};
use crate::json::repr_str;
use crate::sample::writer::{amend_reattesting, unknown_algo};
use crate::sample::{amend, open_sample};
use crate::storage::index::DEFAULT_OCCUPANCY_FACTOR;
use crate::{Error, Result, VERSION};

/// What is wrong with one file, before anything is done about it.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Diagnosis {
    pub path: String,
    pub mismatched: Vec<String>,
    /// Datasets whose `digest` does not parse: `verify` fails them, so they
    /// need restamping as a mismatch does.
    pub malformed: Vec<String>,
    pub undigested: Vec<String>,
    pub unattested: Vec<String>,
    pub stale_index: Vec<String>,
    pub missing_index: Vec<String>,
    pub content_id_ok: Option<bool>,
}

impl Diagnosis {
    /// A *stale* index is a defect; an absent one is a choice.
    pub fn needs_index(&self) -> bool {
        !self.stale_index.is_empty()
    }

    /// Whether the digests no longer attest the bytes: a mismatch, a digest
    /// that does not parse, a path no line covers, or a stale `content_id`.
    /// A file whose digests did not parse was "nothing to fix" while `verify`
    /// failed it.
    pub fn needs_digests(&self) -> bool {
        !self.mismatched.is_empty()
            || !self.malformed.is_empty()
            || !self.unattested.is_empty()
            || self.content_id_ok == Some(false)
    }

    pub fn clean(&self) -> bool {
        !self.needs_index() && !self.needs_digests()
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "mismatched": self.mismatched,
            "malformed": self.malformed,
            "undigested": self.undigested,
            "unattested": self.unattested,
            "stale_index": self.stale_index,
            "missing_index": self.missing_index,
            "content_id_ok": self.content_id_ok,
            "needs_index": self.needs_index(),
            "needs_digests": self.needs_digests(),
        })
    }
}

/// What was actually done to one file.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Repair {
    pub path: String,
    pub diagnosis: Diagnosis,
    pub rebuilt_index: Vec<String>,
    pub rewrote_digests: bool,
    pub content_id: Option<String>,
    pub notes: Vec<String>,
}

impl Repair {
    pub fn changed(&self) -> bool {
        !self.rebuilt_index.is_empty() || self.rewrote_digests
    }

    pub fn to_json(&self) -> Value {
        json!({
            "path": self.path,
            "diagnosis": self.diagnosis.to_json(),
            "rebuilt_index": self.rebuilt_index,
            "rewrote_digests": self.rewrote_digests,
            "content_id": self.content_id,
            "changed": self.changed(),
            "notes": self.notes,
        })
    }
}

/// What a file needs, without touching it.
///
/// A file declaring a `digest_algo` outside §2.1 cannot be verified, and is
/// refused (E703) with the repair that re-attests it named.
pub fn diagnose(path: &Path) -> Result<Diagnosis> {
    diagnose_with(path, false)
}

/// [`diagnose`]; `reattesting`, for a re-attestation, the root of a file
/// declaring a `digest_algo` outside §2.1 is left unverified rather than
/// refused --- its `content_id` is unjudged --- and its datasets are each
/// verified in the algorithm their digest names.
fn diagnose_with(path: &Path, reattesting: bool) -> Result<Diagnosis> {
    let sample = open_sample(path)?;
    let unknown = unknown_algo(&sample.root)?;
    if let (Some(algo), false) = (&unknown, reattesting) {
        return Err(unknown_algo_refused(path, algo));
    }
    let result = verify_root(&sample.root, Some(&sample.attr_name_map()?), None, unknown.is_none())?;
    let indexable: std::collections::BTreeSet<String> = sample
        .annotations()?
        .iter()
        .filter(|(_, a)| a.is_voxel() && a.kind() != "mask")
        .map(|(n, _)| n.clone())
        .collect();
    let present: std::collections::BTreeSet<String> = sample.index()?.keys().cloned().collect();
    Ok(Diagnosis {
        path: path.to_string_lossy().into_owned(),
        mismatched: result.mismatched.clone(),
        malformed: result.malformed.clone(),
        undigested: result.undigested.clone(),
        unattested: result.unattested.clone(),
        stale_index: super::stale_index_entries(&sample.root)?,
        missing_index: indexable.difference(&present).cloned().collect(),
        content_id_ok: result.content_id_ok(),
    })
}

/// E703 for the file at `path`, declaring the `digest_algo` `algo` outside
/// §2.1, naming the repair.
pub(crate) fn unknown_algo_refused(path: &Path, algo: &str) -> Error {
    Error::coded(
        "E703",
        format!(
            "{} declares digest_algo {}: unsupported digest algorithm {}; §2.1 permits sha256, sha512 and blake2b, \
             so its digests can be neither checked nor kept. Re-attest it deliberately with `medh5 fix \
             --rewrite-digests --reason ...`, which restamps it in sha256 and records the reason in the file's \
             provenance",
            repr_str(&path.to_string_lossy()),
            repr_str(algo),
            repr_str(algo)
        ),
    )
}

/// Options for [`fix`].
#[derive(Debug, Clone, Default)]
pub struct FixOptions {
    pub rebuild_index: bool,
    pub rewrite_digests: bool,
    pub reason: Option<String>,
    pub performed_by: Option<String>,
    pub max_coords: Option<usize>,
}

/// Refuse what no file could be repaired with.
fn check_options(options: &FixOptions) -> Result<()> {
    if options.rewrite_digests && options.reason.as_deref().unwrap_or("").is_empty() {
        return Err(Error::invalid(
            "rewriting digests re-attests content this tool did not verify; pass a reason, which is recorded in the file's provenance",
        ));
    }
    Ok(())
}

/// Repair one file.  Neither flag means diagnose and change nothing.
///
/// Only a *stale* index is rebuilt; `missing_index` is reported and never
/// acted on, because an index is optional per annotation (§14.3).  An entry
/// is rebuilt as it was built --- its `max_coords` (unless `options` gives
/// one), its `seed` and its occupancy planes --- where the defaults replaced
/// them.  A file the validator rejects is not restamped: the commit refuses
/// to write it, as it refuses every file it would fail (§15).
pub fn fix(path: &Path, options: &FixOptions) -> Result<Repair> {
    if options.rebuild_index || options.rewrite_digests {
        check_options(options)?;
    }
    let diagnosis = diagnose_with(path, options.rewrite_digests)?;
    let mut repair =
        Repair { path: path.to_string_lossy().into_owned(), diagnosis: diagnosis.clone(), ..Default::default() };
    if !options.rebuild_index && !options.rewrite_digests {
        return Ok(repair);
    }
    if options.rebuild_index && !options.rewrite_digests && diagnosis.needs_digests() {
        let mut message = format!(
            "{}: cannot rebuild the index without restamping digests, and this file has {} that no longer match its bytes",
            path.to_string_lossy(),
            diagnosis.mismatched.len()
        );
        if !diagnosis.malformed.is_empty() {
            message.push_str(&format!(" and {} that do not parse", diagnosis.malformed.len()));
        }
        if !diagnosis.unattested.is_empty() {
            message.push_str(&format!(
                " and {} path(s) inside attested objects that no line of content_id covers",
                diagnosis.unattested.len()
            ));
        }
        if diagnosis.content_id_ok == Some(false) {
            message.push_str(" (and a stale content_id)");
        }
        message.push_str(
            ". Rebuilding would recompute them and the mismatch would vanish unrecorded. Restore the content, or pass \
             rewrite_digests with a reason so the re-attestation is written into the file.",
        );
        return Err(Error::invalid(message));
    }
    if diagnosis.stale_index.is_empty() && !options.rewrite_digests {
        let mut note = String::from("nothing to rebuild: no sampling index is stale");
        if !diagnosis.missing_index.is_empty() {
            note.push_str(&format!(
                "; {} voxel annotation(s) have none ({}), which `medh5 index build` writes",
                diagnosis.missing_index.len(),
                diagnosis.missing_index.join(", ")
            ));
        }
        repair.notes.push(note);
        return Ok(repair);
    }
    // Re-attesting is the one amendment of a file whose attestation broke
    // (§14.4); anything else amends as every writer does, and is refused it.
    let replaced = if options.rewrite_digests { unknown_algo(&open_sample(path)?.root)? } else { None };
    let mut writer = if options.rewrite_digests { amend_reattesting(path, None)? } else { amend(path, None)? };
    let result = (|| -> Result<()> {
        if options.rewrite_digests {
            // Restamped before the index is planned: an entry pins its
            // annotation's digest (§13.3), so one rebuilt from the digests the
            // file had went stale when the commit restamped them --- and one
            // that matched before the restamp went stale without being rebuilt.
            stamp_digests(&writer.root()?, writer.digest_algo(), &["index"], false)?;
        }
        let mut names = stale_index_entries(&writer.root()?)?;
        names.sort();
        if options.rebuild_index && !names.is_empty() {
            for name in &names {
                let built = built_with(&writer.root()?, name)?;
                let max_coords = options.max_coords.or(built.max_coords);
                let one = std::slice::from_ref(name);
                repair.rebuilt_index.extend(writer.build_index(
                    Some(one),
                    max_coords,
                    Some(built.occupancy),
                    built.seed,
                )?);
            }
        } else if !names.is_empty() {
            repair.notes.push(format!(
                "{} sampling index entr{} no longer match their annotations ({}); rebuild them with rebuild_index",
                names.len(),
                if names.len() == 1 { "y" } else { "ies" },
                names.join(", ")
            ));
        }
        if options.rewrite_digests {
            let agent = match &options.performed_by {
                Some(who) => writer.person(who, None, Map::new())?,
                None => writer.software("medh5", Some(VERSION), Map::new())?,
            };
            let mut restamped: Vec<String> =
                diagnosis.mismatched.iter().chain(&diagnosis.malformed).chain(&diagnosis.unattested).cloned().collect();
            restamped.sort();
            restamped.dedup();
            let mut fields = Map::new();
            fields.insert("tool".into(), Value::String("medh5 fix --rewrite-digests".into()));
            let mut params = json!({
                "reason": options.reason,
                "restamped": if restamped.is_empty() { json!("all") } else { json!(restamped) },
                "verified_content": false,
            });
            if let Some(algo) = &replaced {
                params["digest_algo"] = json!({"from": algo, "to": writer.digest_algo()});
            }
            fields.insert("params".into(), params);
            writer.activity("other", Some(&agent.id), None, fields)?;
            repair.rewrote_digests = true;
            repair.notes.push(
                "digests were recomputed from the current bytes; this asserts nothing about whether those bytes are correct"
                    .into(),
            );
            if let Some(algo) = &replaced {
                repair.notes.push(format!(
                    "the file declared digest_algo {}, which §2.1 does not permit; it was restamped in {}",
                    repr_str(algo),
                    writer.digest_algo()
                ));
            }
        }
        writer.commit(true)?;
        Ok(())
    })();
    if let Err(e) = result {
        writer.abort();
        return Err(e);
    }
    repair.content_id = open_sample(path)?.content_id()?;
    Ok(repair)
}

/// How an index entry was built: what a rebuild of it keeps.
struct Built {
    /// `None` when the entry does not say: the default.
    max_coords: Option<usize>,
    /// The occupancy factor, `None` for an entry without occupancy planes.
    occupancy: Option<usize>,
    seed: u64,
}

/// What `index/<name>` was built with: its `max_coords` and `seed`, and the
/// block edge its occupancy planes were cut at --- not stored, so read back
/// from their shape against the annotation's grid, the default where it fits.
fn built_with(root: &hdf5::Group, name: &str) -> Result<Built> {
    let entry = root.group("index")?.group(name)?;
    let max_coords = attrs::get_i64(&entry, "max_coords")?.and_then(|n| usize::try_from(n).ok()).filter(|n| *n > 0);
    let seed = attrs::get_i64(&entry, "seed")?.and_then(|n| u64::try_from(n).ok()).unwrap_or(0);
    let Some(planes) = ops::child_dataset(&entry, "occupancy") else {
        return Ok(Built { max_coords, occupancy: None, seed });
    };
    let spatial = (|| -> Result<Option<Vec<usize>>> {
        let annotation = root.group("annotations")?.group(name)?;
        let Some(grid_id) = attrs::get_str(&annotation, "grid")? else { return Ok(None) };
        let grid = read_grid(&root.group("grids")?.group(&grid_id)?, Some(&grid_id))?;
        Ok(Some(grid.spatial_shape()))
    })()
    .ok()
    .flatten();
    let factor = spatial.and_then(|spatial| occupancy_factor(&planes.shape(), &spatial));
    Ok(Built { max_coords, occupancy: Some(factor.unwrap_or(DEFAULT_OCCUPANCY_FACTOR)), seed })
}

/// The block edge that cuts `spatial` into the coarse shape of `planes`,
/// `(C, *coarse)`: the default where it does, else the smallest that does.
fn occupancy_factor(planes: &[usize], spatial: &[usize]) -> Option<usize> {
    let coarse = planes.get(1..)?;
    if coarse.len() != spatial.len() {
        return None;
    }
    let cuts = |f: usize| spatial.iter().zip(coarse).all(|(n, c)| n.div_ceil(f).max(1) == *c);
    if cuts(DEFAULT_OCCUPANCY_FACTOR) {
        return Some(DEFAULT_OCCUPANCY_FACTOR);
    }
    (1..=spatial.iter().copied().max().unwrap_or(1).max(1)).find(|f| cuts(*f))
}

/// Repair many files: every one of them, a file that cannot be repaired
/// stopping no other.
///
/// When any could not be, the error --- the first failure's kind --- names
/// each file that failed and why, and how many of the others were repaired.
/// Stopping at the first failure left the files after it unrepaired, and
/// said nothing of those before it.  One file fails as [`fix`] does.
pub fn fix_paths(paths: &[&Path], options: &FixOptions) -> Result<Vec<Repair>> {
    if options.rebuild_index || options.rewrite_digests {
        check_options(options)?;
    }
    let mut done = Vec::new();
    let mut failed: Vec<(String, Error)> = Vec::new();
    for path in paths {
        match fix(path, options) {
            Ok(repair) => done.push(repair),
            Err(e) => failed.push((path.to_string_lossy().into_owned(), e)),
        }
    }
    if failed.is_empty() {
        return Ok(done);
    }
    if paths.len() == 1 {
        return Err(failed.remove(0).1);
    }
    let listed: Vec<String> = failed.iter().map(|(path, e)| format!("{}: {}", repr_str(path), e.message())).collect();
    let message = format!(
        "{} of {} files could not be repaired, and {} were: {}",
        failed.len(),
        paths.len(),
        done.len(),
        listed.join("; ")
    );
    Err(restated(failed.remove(0).1, message))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block edge is not stored, so it is read back from the planes'
    /// shape: the default where it fits, else the smallest that does.
    #[test]
    fn s14_3_an_occupancy_factor_is_read_back_from_its_planes() {
        assert_eq!(occupancy_factor(&[2, 2, 3, 3], &[16, 24, 24]), Some(DEFAULT_OCCUPANCY_FACTOR));
        assert_eq!(occupancy_factor(&[2, 4, 6, 6], &[16, 24, 24]), Some(4));
        assert_eq!(occupancy_factor(&[1, 16, 24, 24], &[16, 24, 24]), Some(1));
        assert_eq!(occupancy_factor(&[1, 5, 7, 7], &[16, 24, 24]), None);
        assert_eq!(occupancy_factor(&[1, 2, 3], &[16, 24, 24]), None);
    }
}
