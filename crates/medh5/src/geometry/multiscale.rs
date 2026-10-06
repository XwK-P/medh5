//! Multiscale pyramids (spec §4.3).
//!
//! Every level has its **own grid**, and the geometry of level *l* is derived
//! from level 0 by a rule the validator checks to 1e-3 relative tolerance:
//!
//! ```text
//! spacing'_k = spacing_k * f_k
//! origin'    = origin + direction @ (spacing * ((f - 1) / 2))
//! ```
//!
//! The half-voxel term is the whole point: downsampling by *f* merges *f*
//! voxels into one whose centre sits `(f-1)/2` voxels further along the axis.

use ndarray::Array2;

use super::affine::ORTHONORMAL_TOL;
use super::grid::Grid;
use super::linalg::{allclose, isclose};
use crate::h5::attrs::AttrValue;
use crate::json::{format_g, repr_str};
use crate::{Error, Result};

/// `downsample_method` values (§4.3).
pub const DOWNSAMPLE_METHODS: [&str; 4] = ["mean", "nearest", "gaussian", "max"];
/// Methods that preserve class identity; `mean` on a labelmap invents classes.
pub const LABEL_SAFE_METHODS: [&str; 2] = ["nearest", "max"];
/// The validator's relative tolerance on derived geometry.
pub const GEOMETRY_RTOL: f64 = 1e-3;

/// The declaration on an `images/<id>` group that holds multiple levels.
#[derive(Debug, Clone, PartialEq)]
pub struct Pyramid {
    pub levels: usize,
    /// `(levels, S)` factors, relative to level 0.
    pub downsample_factors: Array2<f64>,
    pub downsample_method: String,
    pub grid_levels: Vec<String>,
}

impl Pyramid {
    /// A validated pyramid declaration.
    pub fn new(
        levels: usize,
        downsample_factors: Array2<f64>,
        downsample_method: impl Into<String>,
        grid_levels: Vec<String>,
    ) -> Result<Self> {
        let p = Pyramid { levels, downsample_factors, downsample_method: downsample_method.into(), grid_levels };
        let (rows, cols) = p.downsample_factors.dim();
        if rows != p.levels {
            return Err(Error::coded(
                "E105",
                format!("downsample_factors must be (levels, S); got ({rows}, {cols}) for levels={}", p.levels),
            ));
        }
        if p.grid_levels.len() != p.levels {
            return Err(Error::coded(
                "E105",
                format!("grid_levels has {} entries for levels={}", p.grid_levels.len(), p.levels),
            ));
        }
        if !DOWNSAMPLE_METHODS.contains(&p.downsample_method.as_str()) {
            return Err(Error::coded(
                "E105",
                format!("unknown downsample_method {}", repr_str(&p.downsample_method)),
            ));
        }
        if p.downsample_factors.iter().any(|v| *v <= 0.0) {
            return Err(Error::coded("E105", "downsample factors must be > 0"));
        }
        if rows > 0 {
            let first: Vec<f64> = p.downsample_factors.row(0).to_vec();
            if !allclose(&first, &vec![1.0; first.len()], 1e-8, 1e-5) {
                return Err(Error::coded("E105", "level 0 must have downsample factor 1 on every axis"));
            }
        }
        Ok(p)
    }

    /// The group attributes (§4.3).
    pub fn attrs(&self) -> Vec<(String, AttrValue)> {
        let (r, c) = self.downsample_factors.dim();
        vec![
            ("levels".into(), AttrValue::Int(self.levels as i64)),
            (
                "downsample_factors".into(),
                AttrValue::matrix(r, c, self.downsample_factors.as_standard_layout().as_slice().unwrap()),
            ),
            ("downsample_method".into(), AttrValue::Str(self.downsample_method.clone())),
            ("grid_levels".into(), AttrValue::strs(&self.grid_levels)),
        ]
    }
}

/// Derive the grid of one pyramid level from level 0 (spec §4.3).
///
/// `shape` defaults to `ceil(shape / f)`, which keeps the full field of view
/// covered; only the spacing/origin relation is normative.
pub fn derive_level_grid(base: &Grid, factors: &[f64], grid_id: &str, shape: Option<&[i64]>) -> Result<Grid> {
    let s = base.n_spatial();
    if factors.len() != s {
        return Err(Error::coded("E105", format!("factors must have {s} entries, got ({},)", factors.len())));
    }
    let spacing: Vec<f64> = base.spacing.iter().zip(factors).map(|(sp, f)| sp * f).collect();
    let half: Vec<f64> = base.spacing.iter().zip(factors).map(|(sp, f)| sp * (f - 1.0) / 2.0).collect();
    let origin: Vec<f64> = (0..s)
        .map(|r| {
            let shift: f64 = (0..s).map(|c| base.direction[[r, c]] * half[c]).sum();
            base.origin[r] + shift
        })
        .collect();
    let full_shape: Vec<i64> = match shape {
        Some(shape) => shape.to_vec(),
        None => {
            let lead = &base.shape[..base.ndim() - s];
            let spatial = base.spatial_shape().iter().zip(factors).map(|(n, f)| ((*n as f64 / f).ceil() as i64).max(1)).collect::<Vec<_>>();
            lead.iter().copied().chain(spatial).collect()
        }
    };
    let mut grid = Grid {
        grid_id: grid_id.to_string(),
        shape: full_shape,
        axis_names: base.axis_names.clone(),
        axis_kinds: base.axis_kinds.clone(),
        spacing,
        origin,
        direction: base.direction.clone(),
        coord_system: base.coord_system.clone(),
        units: base.units.clone(),
        timepoint: base.timepoint.clone(),
        frame_uid: base.frame_uid.clone(),
        time_values: base.time_values.clone(),
        time_units: base.time_units.clone(),
        chunk_hint: None,
        patch_hint: None,
        extra: Vec::new(),
    };
    grid.extra.clear();
    grid.check()?;
    Ok(grid)
}

/// Check every level's geometry against the derivation rule.
///
/// Returns human-readable messages; empty means the pyramid conforms.
pub fn check_pyramid(base: &Grid, levels: &[&Grid], factors: &Array2<f64>, rtol: f64) -> Result<Vec<String>> {
    let mut problems = Vec::new();
    for (level, grid) in levels.iter().enumerate() {
        if level >= factors.nrows() {
            break;
        }
        let f: Vec<f64> = factors.row(level).to_vec();
        let expected = derive_level_grid(base, &f, &grid.grid_id, Some(&grid.shape))?;
        for axis in 0..base.n_spatial() {
            let (got_s, want_s) = (grid.spacing[axis], expected.spacing[axis]);
            if !isclose(got_s, want_s, rtol, 0.0) {
                problems.push(format!(
                    "level {level} axis {axis}: spacing {} != {} (= spacing0 * {})",
                    format_g(got_s, 6),
                    format_g(want_s, 6),
                    format_g(f[axis], 6)
                ));
            }
            let (got_o, want_o) = (grid.origin[axis], expected.origin[axis]);
            let scale = want_o.abs().max(base.spacing[axis]);
            if (got_o - want_o).abs() > rtol * scale {
                problems.push(format!(
                    "level {level} axis {axis}: origin {} != {} (half-voxel shift missing?)",
                    format_g(got_o, 6),
                    format_g(want_o, 6)
                ));
            }
        }
        let same_direction = grid.direction.dim() == base.direction.dim()
            && allclose(
                grid.direction.as_standard_layout().as_slice().unwrap(),
                base.direction.as_standard_layout().as_slice().unwrap(),
                ORTHONORMAL_TOL,
                1e-5,
            );
        if !same_direction {
            problems.push(format!(
                "level {level}: direction differs from level 0; a pyramid level shares its parent's \
                 orientation, only its sampling changes"
            ));
        }
        if grid.axis_kinds != base.axis_kinds {
            problems.push(format!("level {level}: axis_kinds differ from level 0"));
        }
        let base_spatial = base.spatial_shape();
        let grid_spatial = grid.spatial_shape();
        for axis in 0..base.n_spatial() {
            let n = base_spatial[axis] as f64;
            let got_n = grid_spatial.get(axis).copied().unwrap_or(0) as i64;
            let low = ((n / f[axis]).floor() as i64).max(1);
            let high = ((n / f[axis]).ceil() as i64).max(1);
            if !(low <= got_n && got_n <= high) {
                problems.push(format!(
                    "level {level} axis {axis}: extent {got_n} is not a factor-{} resampling of {} \
                     (expected {low} or {high})",
                    format_g(f[axis], 6),
                    base_spatial[axis]
                ));
            }
        }
        if grid.coord_system != base.coord_system || grid.units != base.units {
            problems.push(format!("level {level}: coord_system/units differ from level 0"));
        }
        if grid.frame_uid != base.frame_uid {
            problems.push(format!("level {level}: frame_uid differs from level 0"));
        }
    }
    Ok(problems)
}

/// Recover per-level downsample factors from the levels' spacings.
pub fn pyramid_factors(base: &Grid, levels: &[&Grid]) -> Array2<f64> {
    let s = base.spacing.len();
    let mut out = Array2::zeros((levels.len(), s));
    for (i, g) in levels.iter().enumerate() {
        for k in 0..s {
            out[[i, k]] = g.spacing.get(k).copied().unwrap_or(f64::NAN) / base.spacing[k];
        }
    }
    out
}
