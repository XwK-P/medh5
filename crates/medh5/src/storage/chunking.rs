//! Cache-aware chunk sizing (spec §14.1).
//!
//! HDF5 decompresses a whole chunk to serve any element of it, so the chunk is
//! the real unit of I/O.  Sizing it to the L3 cache keeps a patch read inside
//! cache after decompression; sizing it near the training patch keeps read
//! amplification low.  Those two pull in opposite directions; this resolves
//! them by starting at the patch, growing toward the cache budget, and stopping
//! before the chunk is much larger than the patch.

use std::sync::OnceLock;

use serde_json::{json, Value};

use crate::geometry::Grid;
use crate::json::repr_int_tuple;
use crate::{Error, Result};

/// Fallback L3 slice, ~1.375 MiB.
pub const DEFAULT_L3_BYTES: u64 = 1_441_792;
pub const MIN_CHUNK_BYTES: f64 = 512.0 * 1024.0;
pub const MAX_CHUNK_BYTES: f64 = 4.0 * 1024.0 * 1024.0;
pub const CACHE_SAFETY: f64 = 0.8;
/// Stop growing once the mean chunk/patch ratio exceeds this.
pub const OVERSHOOT_LIMIT: f64 = 1.5;
pub const DEFAULT_PATCH: usize = 64;

/// Best-effort L3 detection, falling back to [`DEFAULT_L3_BYTES`].
pub fn detect_l3_bytes() -> u64 {
    static L3: OnceLock<u64> = OnceLock::new();
    *L3.get_or_init(|| {
        if let Ok(raw) = std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index3/size") {
            let raw = raw.trim();
            if let Some(k) = raw.strip_suffix('K') {
                if let Ok(v) = k.parse::<u64>() {
                    return v * 1024;
                }
            } else if let Some(m) = raw.strip_suffix('M') {
                if let Ok(v) = m.parse::<u64>() {
                    return v * 1024 * 1024;
                }
            } else if let Ok(v) = raw.parse::<u64>() {
                return v;
            }
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(v) = macos_l3() {
                return v;
            }
        }
        DEFAULT_L3_BYTES
    })
}

#[cfg(target_os = "macos")]
fn macos_l3() -> Option<u64> {
    let out = std::process::Command::new("/usr/sbin/sysctl").args(["-n", "hw.l3cachesize"]).output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let v: u64 = text.trim().parse().ok()?;
    (v > 0).then_some(v)
}

fn budget(l3_bytes: Option<u64>) -> f64 {
    let target = l3_bytes.unwrap_or_else(detect_l3_bytes) as f64 * CACHE_SAFETY;
    target.max(MIN_CHUNK_BYTES).min(MAX_CHUNK_BYTES)
}

fn pow2_ceil(p: usize) -> usize {
    let exp = (p.max(1) as f64).log2().ceil().max(0.0) as u32;
    1usize << exp
}

/// A training patch hint: none, one size for every axis, or per axis.
#[derive(Debug, Clone, PartialEq)]
pub enum Patch {
    Default,
    Uniform(usize),
    PerAxis(Vec<i64>),
}

/// Chunk extents for the spatial axes alone.
pub fn spatial_chunk_for(
    spatial_shape: &[usize],
    patch: &Patch,
    itemsize: usize,
    l3_bytes: Option<u64>,
) -> Result<Vec<usize>> {
    if spatial_shape.is_empty() || spatial_shape.contains(&0) {
        return Err(Error::invalid(format!("spatial shape must be positive, got {}", repr_int_tuple(spatial_shape))));
    }
    let patch_t: Vec<usize> = match patch {
        Patch::Default => spatial_shape.iter().map(|s| DEFAULT_PATCH.min(*s)).collect(),
        Patch::Uniform(p) => spatial_shape.iter().map(|s| (*p).min(*s)).collect(),
        Patch::PerAxis(p) => {
            if p.len() != spatial_shape.len() {
                return Err(Error::invalid(format!(
                    "patch {} does not match spatial shape {}",
                    repr_int_tuple(p),
                    repr_int_tuple(spatial_shape)
                )));
            }
            p.iter().map(|v| (*v).max(0) as usize).collect()
        }
    };
    let patch_t: Vec<usize> = patch_t.iter().zip(spatial_shape).map(|(p, s)| (*p).min(*s).max(1)).collect();
    let mut chunk: Vec<usize> = patch_t.iter().zip(spatial_shape).map(|(p, s)| pow2_ceil(*p).min(*s)).collect();
    let budget = budget(l3_bytes);
    let nbytes = |c: &[usize]| c.iter().map(|v| *v as f64).product::<f64>() * itemsize as f64;
    while nbytes(&chunk) < budget {
        let growable: Vec<usize> = (0..chunk.len()).filter(|i| chunk[*i] < spatial_shape[*i]).collect();
        if growable.is_empty() {
            break;
        }
        // `min(growable, key=ratio)` keeps the first minimum.
        let mut axis = growable[0];
        for &i in &growable[1..] {
            if (chunk[i] as f64 / patch_t[i] as f64) < (chunk[axis] as f64 / patch_t[axis] as f64) {
                axis = i;
            }
        }
        let step = pow2_ceil(patch_t[axis]);
        let mut grown = chunk.clone();
        grown[axis] = (grown[axis] + step).min(spatial_shape[axis]);
        let mean: f64 =
            grown.iter().zip(&patch_t).map(|(c, p)| *c as f64 / *p as f64).sum::<f64>() / grown.len() as f64;
        if mean > OVERSHOOT_LIMIT {
            break;
        }
        chunk = grown;
    }
    Ok(chunk.iter().zip(spatial_shape).map(|(c, s)| (*c).min(*s)).collect())
}

/// Full chunk shape for an array whose axes are described by `axis_kinds`.
///
/// Non-spatial axes get extent 1; `leading` prepends that many extra size-1
/// axes --- the `(1, *spatial_chunk)` requirement of stacked encodings.
pub fn optimize_chunks(
    shape: &[usize],
    axis_kinds: &[String],
    patch: &Patch,
    itemsize: usize,
    l3_bytes: Option<u64>,
    leading: usize,
) -> Result<Vec<usize>> {
    if axis_kinds.len() + leading != shape.len() {
        return Err(Error::invalid(format!(
            "axis_kinds {} (+{leading} leading) does not describe shape {}",
            crate::json::repr_list(axis_kinds),
            repr_int_tuple(shape)
        )));
    }
    let body = &shape[leading..];
    let spatial_idx: Vec<usize> =
        axis_kinds.iter().enumerate().filter(|(_, k)| *k == "spatial").map(|(i, _)| i).collect();
    let spatial_shape: Vec<usize> = spatial_idx.iter().map(|i| body[*i]).collect();
    let spatial_chunk = spatial_chunk_for(&spatial_shape, patch, itemsize, l3_bytes)?;
    let mut out = vec![1usize; body.len()];
    for (slot, i) in spatial_idx.iter().enumerate() {
        out[*i] = spatial_chunk[slot];
    }
    let mut full = vec![1usize; leading];
    full.extend(out);
    Ok(full)
}

/// The chunk shape the writer gives an array laid out on `grid` (§14.1).
pub fn grid_chunks(grid: &Grid, itemsize: usize, leading: usize) -> Result<Vec<usize>> {
    if let Some(hint) = &grid.chunk_hint {
        if leading == 0 {
            return Ok(hint.iter().map(|c| (*c).max(0) as usize).collect());
        }
    }
    let mut shape = vec![1usize; leading];
    shape.extend(grid.shape_usize());
    let patch = match &grid.patch_hint {
        Some(p) => Patch::PerAxis(p.clone()),
        None => Patch::Default,
    };
    optimize_chunks(&shape, &grid.axis_kinds, &patch, itemsize, None, leading)
}

/// `proposed`'s trailing entries clipped to `shape`, or `None`.
pub fn fit_chunks(proposed: &[usize], shape: &[usize]) -> Option<Vec<usize>> {
    let dims = shape.len();
    if proposed.len() < dims {
        return None;
    }
    let tail = &proposed[proposed.len() - dims..];
    Some(tail.iter().zip(shape).map(|(c, s)| (*c).min(*s)).collect())
}

/// Chunks for a displacement field on `grid`: one vector component per chunk.
pub fn field_chunks(grid: &Grid, shape: &[usize], itemsize: usize) -> Result<Option<Vec<usize>>> {
    if shape.len() != grid.n_spatial() + 1 {
        return Ok(None);
    }
    let all = grid_chunks(grid, itemsize, 0)?;
    let spatial = &all[all.len() - grid.n_spatial()..];
    let mut proposed = vec![1usize];
    proposed.extend_from_slice(spatial);
    Ok(fit_chunks(&proposed, shape))
}

/// Diagnostics for `medh5 info`: chunk bytes, count.
pub fn chunk_report(shape: &[usize], chunks: &[usize], itemsize: usize) -> Value {
    let chunk_bytes: usize = chunks.iter().product::<usize>() * itemsize;
    let n_chunks: usize = shape.iter().zip(chunks).map(|(s, c)| s.div_ceil((*c).max(1))).product();
    let mib = crate::geometry::grid::round_to(chunk_bytes as f64 / 1024.0 / 1024.0, 3);
    json!({"chunks": chunks, "chunk_bytes": chunk_bytes, "chunk_mib": mib, "n_chunks": n_chunks})
}

// -- h5py's chunk guess, for datasets chunked without an explicit shape ----------

const CHUNK_BASE: f64 = 16.0 * 1024.0;
const CHUNK_MIN: f64 = 8.0 * 1024.0;
const CHUNK_MAX: f64 = 1024.0 * 1024.0;

/// h5py's `guess_chunk`: a power-of-two fraction of each axis, sized from the
/// dataset's total size.  Used where 1.x passed `chunks=True`, so a dataset
/// is chunked the way it always was.
pub fn guess_chunk(shape: &[usize], typesize: usize) -> Vec<usize> {
    let mut chunks: Vec<f64> = shape.iter().map(|x| if *x == 0 { 1024.0 } else { *x as f64 }).collect();
    let ndims = chunks.len();
    if ndims == 0 {
        return Vec::new();
    }
    let product = |c: &[f64]| c.iter().product::<f64>();
    let dset_size = product(&chunks) * typesize as f64;
    let mut target = CHUNK_BASE * 2f64.powf((dset_size / (1024.0 * 1024.0)).log10());
    if target > CHUNK_MAX {
        target = CHUNK_MAX;
    } else if target < CHUNK_MIN {
        target = CHUNK_MIN;
    }
    let mut idx = 0usize;
    loop {
        let chunk_bytes = product(&chunks) * typesize as f64;
        if (chunk_bytes < target || (chunk_bytes - target).abs() / target < 0.5) && chunk_bytes < CHUNK_MAX {
            break;
        }
        if product(&chunks) == 1.0 {
            break;
        }
        let axis = idx % ndims;
        chunks[axis] = (chunks[axis] / 2.0).ceil();
        idx += 1;
    }
    chunks.iter().map(|c| *c as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spatial_chunks_match_reference() {
        // medh5 1.4.4 on a host whose L3 clamps the budget to 4 MiB.
        let kinds: Vec<String> = vec!["spatial".into(); 3];
        let c =
            optimize_chunks(&[16, 24, 24], &kinds, &Patch::PerAxis(vec![8, 8, 8]), 2, Some(272_629_760), 0).unwrap();
        assert_eq!(c, vec![16, 8, 8]);
    }

    #[test]
    fn h5py_guess_chunk() {
        // Reference values from h5py 3.16's `guess_chunk`.
        assert_eq!(guess_chunk(&[1000], 8), vec![1000]);
        assert_eq!(guess_chunk(&[100, 100, 100], 4), vec![13, 25, 25]);
        assert_eq!(guess_chunk(&[7, 3, 2], 2), vec![7, 3, 2]);
        assert_eq!(guess_chunk(&[5_000_000], 1), vec![39063]);
        assert_eq!(guess_chunk(&[3, 160, 160, 160], 2), vec![1, 20, 20, 40]);
        assert_eq!(guess_chunk(&[41213], 1), vec![10304]);
        assert_eq!(guess_chunk(&[200, 64, 64], 8), vec![25, 8, 16]);
    }
}
