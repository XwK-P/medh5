//! §13 --- digests, `content_id` and index currency (E7xx).

use std::collections::BTreeMap;

use super::{loc, str_attr, sub_dataset, Context};
use crate::annotations::encode::normalization_tolerance;
use crate::array::NdArray;
use crate::digest::{parse_digest, DEFAULT_ALGO};
use crate::h5::attrs;
use crate::h5::data::{self, Kind};
use crate::integrity::digest::{dataset_digest_inspected, STREAM_BYTES};
use crate::integrity::{stale_index_entries, verify_root_inspecting};
use crate::json::{py_float, repr_int_tuple, repr_str};
use crate::validate::Diagnostic;
use crate::Result;

pub fn check_integrity(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let root = ctx.root.clone();
    let mut algo_known = true;
    if let Some(algo) = str_attr(&root, "digest_algo")? {
        if parse_digest(&format!("{algo}:00")).is_err() {
            algo_known = false;
            out.push(ctx.err(
                "E703",
                "/",
                format!("unsupported digest_algo {}; §2.1 permits sha256, sha512 and blake2b", repr_str(&algo)),
            ));
        }
    }
    // The stored probabilities are checked as the digest pass reads them.
    let mut probmaps = probmaps(ctx)?;
    let result = verify_root_inspecting(&root, ctx.attr_names.as_ref(), None, algo_known, &mut |path, block| {
        if let Some(check) = probmaps.get_mut(path) {
            check.feed(block);
        }
        Ok(())
    })?;
    for (path, check) in probmaps.iter_mut().filter(|(_, c)| !c.read) {
        // Undigested, so the pass above did not read it.
        let ds = root.dataset(path)?;
        dataset_digest_inspected(
            &ds,
            path,
            DEFAULT_ALGO,
            STREAM_BYTES,
            Some(&mut |block| {
                check.feed(block);
                Ok(())
            }),
        )?;
    }
    for (path, check) in &probmaps {
        out.extend(check.findings(ctx, path));
    }
    if result.checked.is_empty() && result.undigested.is_empty() {
        return Ok(out);
    }
    if !result.undigested.is_empty() && result.checked.is_empty() {
        out.push(ctx.err("W901", "/", "no dataset carries a `digest` attribute"));
    }
    for path in &result.malformed {
        out.push(ctx.err("E703", format!("/{path}"), "malformed digest string"));
    }
    for path in &result.mismatched {
        out.push(ctx.err("E701", format!("/{path}"), "digest does not match the stored data"));
    }
    if result.content_id_ok() == Some(false) && ctx.projection {
        // A later minor may cover root attributes this engine does not know:
        // the root's integrity is unsupported here, not refuted (1.1 §2.2).
        out.push(ctx.err(
            "W913",
            "/",
            format!(
                "`content_id` does not recompute under the rules of MEDH5 {}: the root's integrity is unsupported in \
                 this projection, not verified (per-dataset digests were checked)",
                crate::FORMAT_VERSION
            ),
        ));
    } else if result.content_id_ok() == Some(false) {
        out.push(ctx.err(
            "E702",
            "/",
            format!(
                "`content_id` {} does not match the computed {}",
                result.content_id_declared.clone().unwrap_or_default(),
                result.content_id_computed.clone().unwrap_or_default()
            ),
        ));
    }
    for name in stale_index_entries(&root)? {
        out.push(ctx.err(
            "W905",
            format!("/index/{name}"),
            "index `source_digest` does not match its annotation; readers must ignore this entry and rebuild it",
        ));
    }
    Ok(out)
}

/// What a `probmap`'s stored values must be (§7.5): numbers in [0, 1] and,
/// where it is `normalized`, summing to 1 over its classes at every voxel ---
/// checked on the bytes the integrity pass reads anyway.  Only the threshold
/// was checked, so a map of NaN, or one declared normalised that was not,
/// validated clean (C09 of the 2.0 audit).
struct ProbmapCheck {
    class_ids: Vec<i64>,
    spatial: Vec<usize>,
    normalized: bool,
    tolerance: f64,
    rows: usize,
    sums: Vec<f64>,
    bad: Option<(usize, usize, f64)>,
    read: bool,
}

fn probmaps(ctx: &Context) -> Result<BTreeMap<String, ProbmapCheck>> {
    let mut out = BTreeMap::new();
    for (name, group) in ctx.children("annotations")? {
        if str_attr(loc(&group), "kind")?.as_deref() != Some("probmap") {
            continue;
        }
        let Some(ds) = sub_dataset(&group, "data") else { continue };
        let Kind::Numeric(dtype) = data::kind(&ds)? else { continue };
        let shape = ds.shape();
        let Some((classes, spatial)) = shape.split_first() else { continue };
        let normalized = attrs::get_bool(loc(&group), "normalized")?.unwrap_or(false);
        let voxels: usize = spatial.iter().product();
        out.insert(
            format!("annotations/{name}/data"),
            ProbmapCheck {
                class_ids: attrs::get_i64s(loc(&group), "class_ids")?.unwrap_or_default(),
                spatial: spatial.to_vec(),
                normalized,
                tolerance: normalization_tolerance(dtype, *classes),
                rows: 0,
                sums: if normalized { vec![0.0; voxels] } else { Vec::new() },
                bad: None,
                read: false,
            },
        );
    }
    Ok(out)
}

impl ProbmapCheck {
    fn feed(&mut self, block: &NdArray) {
        self.read = true;
        let voxels = self.spatial.iter().product::<usize>().max(1);
        let values = block.to_f64();
        for (k, v) in values.iter().enumerate() {
            if self.bad.is_none() && !(0.0..=1.0).contains(v) {
                self.bad = Some((self.rows + k / voxels, k % voxels, *v));
            }
            if self.normalized {
                self.sums[k % voxels] += *v;
            }
        }
        self.rows += values.shape().first().copied().unwrap_or(0);
    }

    fn voxel(&self, flat: usize) -> String {
        let mut index = vec![0; self.spatial.len()];
        let mut rest = flat;
        for (axis, extent) in self.spatial.iter().enumerate().rev() {
            let extent = (*extent).max(1);
            index[axis] = rest % extent;
            rest /= extent;
        }
        repr_int_tuple(&index)
    }

    fn findings(&self, ctx: &Context, path: &str) -> Vec<Diagnostic> {
        let mut out = Vec::new();
        if let Some((row, voxel, value)) = self.bad {
            let class = self.class_ids.get(row).map_or_else(|| format!("row {row}"), |c| format!("class {c}"));
            out.push(ctx.err(
                "E411",
                format!("/{path}"),
                format!(
                    "a stored probability is {} ({class}, voxel {}); §7.5 stores values in [0, 1]",
                    py_float(value),
                    self.voxel(voxel)
                ),
            ));
        }
        if self.normalized && self.bad.is_none() {
            if let Some((voxel, sum)) = self.sums.iter().enumerate().find(|(_, s)| (**s - 1.0).abs() > self.tolerance) {
                out.push(ctx.err(
                    "E404",
                    format!("/{path}"),
                    format!(
                        "`normalized` is true, but the classes sum to {} at voxel {}; §7.5 has them sum to 1 at every \
                         voxel",
                        py_float(*sum),
                        self.voxel(voxel)
                    ),
                ));
            }
        }
        out
    }
}
