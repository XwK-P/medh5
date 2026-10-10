//! Grids: the named sampling lattices a sample's arrays live on (spec §3.1-§3.2).
//!
//! A grid is an **empty HDF5 group carrying only attributes**.  That makes
//! grids essentially free, which is what lets the format say "two acquisitions
//! with the same lattice in different timepoints are two grids, not one shared
//! grid" --- keeping `timepoint` and `frame_uid` single-valued per grid.

use ndarray::Array2;
use serde_json::{json, Value};

use super::affine::{build_affine, check_orthonormal, index_to_world, world_to_index, ORTHONORMAL_TOL};
use super::linalg::allclose;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::ops;
use crate::ids::validate_id;
use crate::json::{float_repr, repr_int_tuple, repr_str};
use crate::{Error, Result};

/// `axis_kinds` values (§3.1).
pub const AXIS_KINDS: [&str; 4] = ["spatial", "channel", "time", "other"];
/// `units` values (§3.2).
pub const KNOWN_UNITS: [&str; 4] = ["mm", "um", "m", "px"];
/// `time_units` values (§3.2).
pub const TIME_UNITS: [&str; 2] = ["s", "ms"];
/// The spatial dimensionality the spec allows.
pub const MIN_SPATIAL: usize = 2;
pub const MAX_SPATIAL: usize = 3;

/// The grid attributes the spec defines (§3.2), in canonical order.
pub const SPEC_GRID_ATTRS: [&str; 14] = [
    "shape",
    "axis_names",
    "axis_kinds",
    "spacing",
    "origin",
    "direction",
    "coord_system",
    "units",
    "timepoint",
    "frame_uid",
    "time_values",
    "time_units",
    "chunk_hint",
    "patch_hint",
];

/// One discrete sampling lattice with a complete index-world geometry.
#[derive(Debug, Clone)]
pub struct Grid {
    pub grid_id: String,
    pub shape: Vec<i64>,
    pub axis_names: Vec<String>,
    pub axis_kinds: Vec<String>,
    pub spacing: Vec<f64>,
    pub origin: Vec<f64>,
    pub direction: Array2<f64>,
    pub coord_system: String,
    pub units: String,
    pub timepoint: Option<String>,
    pub frame_uid: Option<String>,
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
    pub chunk_hint: Option<Vec<i64>>,
    pub patch_hint: Option<Vec<i64>>,
    /// Attributes outside §3.2, carried through untouched.
    pub extra: Vec<(String, AttrValue)>,
}

/// The fields of a grid before validation.
#[derive(Debug, Clone, Default)]
pub struct GridSpec {
    pub grid_id: String,
    pub shape: Vec<i64>,
    pub axis_names: Option<Vec<String>>,
    pub axis_kinds: Option<Vec<String>>,
    pub spacing: Vec<f64>,
    pub origin: Option<Vec<f64>>,
    pub direction: Option<Array2<f64>>,
    pub coord_system: Option<String>,
    pub units: Option<String>,
    pub timepoint: Option<String>,
    pub frame_uid: Option<String>,
    pub time_values: Option<Vec<f64>>,
    pub time_units: Option<String>,
    pub chunk_hint: Option<Vec<i64>>,
    pub patch_hint: Option<Vec<i64>>,
}

/// Default axis kinds: only 2-D and 3-D have an unambiguous default.
pub fn default_axis_kinds(ndim: usize) -> Result<Vec<String>> {
    if ndim == 2 || ndim == 3 {
        return Ok(vec!["spatial".to_string(); ndim]);
    }
    Err(Error::invalid(format!(
        "a {ndim}-D grid needs explicit axis_kinds: only 2-D and 3-D have an unambiguous default"
    )))
}

/// Default axis names: `z`, `y`, `x` for the spatial axes, the kind's first
/// letter for the others.
pub fn default_axis_names(axis_kinds: &[String]) -> Vec<String> {
    let n_spatial = axis_kinds.iter().filter(|k| *k == "spatial").count();
    let spatial = ["z", "y", "x"];
    let spatial = &spatial[3usize.saturating_sub(n_spatial)..];
    let mut cursor = 0;
    axis_kinds
        .iter()
        .map(|kind| {
            if kind == "spatial" {
                let name = spatial.get(cursor).copied().unwrap_or("s").to_string();
                cursor += 1;
                name
            } else {
                kind.chars().next().map(|c| c.to_string()).unwrap_or_default()
            }
        })
        .collect()
}

impl Grid {
    /// Build and validate a grid from its fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid_id: impl Into<String>,
        shape: Vec<i64>,
        axis_names: Vec<String>,
        axis_kinds: Vec<String>,
        spacing: Vec<f64>,
        origin: Vec<f64>,
        direction: Array2<f64>,
        coord_system: impl Into<String>,
        units: impl Into<String>,
    ) -> Result<Self> {
        let grid = Grid {
            grid_id: grid_id.into(),
            shape,
            axis_names,
            axis_kinds,
            spacing,
            origin,
            direction,
            coord_system: coord_system.into(),
            units: units.into(),
            timepoint: None,
            frame_uid: None,
            time_values: None,
            time_units: None,
            chunk_hint: None,
            patch_hint: None,
            extra: Vec::new(),
        };
        grid.check()?;
        Ok(grid)
    }

    /// A grid from a spec, filling the documented defaults, validated.
    pub fn from_spec(spec: GridSpec) -> Result<Self> {
        let axis_kinds = match spec.axis_kinds {
            Some(kinds) => kinds,
            None => default_axis_kinds(spec.shape.len())?,
        };
        let axis_names = spec.axis_names.unwrap_or_else(|| default_axis_names(&axis_kinds));
        let n_spatial = axis_kinds.iter().filter(|k| *k == "spatial").count();
        let grid = Grid {
            grid_id: spec.grid_id,
            shape: spec.shape,
            axis_names,
            axis_kinds,
            spacing: spec.spacing,
            origin: spec.origin.unwrap_or_else(|| vec![0.0; n_spatial]),
            direction: spec.direction.unwrap_or_else(|| Array2::eye(n_spatial)),
            coord_system: spec.coord_system.unwrap_or_else(|| "LPS".into()),
            units: spec.units.unwrap_or_else(|| "mm".into()),
            timepoint: spec.timepoint,
            frame_uid: spec.frame_uid,
            time_values: spec.time_values,
            time_units: spec.time_units,
            chunk_hint: spec.chunk_hint,
            patch_hint: spec.patch_hint,
            extra: Vec::new(),
        };
        grid.check()?;
        Ok(grid)
    }

    /// Validate every normative rule of spec §3.1-§3.2.
    pub fn check(&self) -> Result<()> {
        validate_id(&self.grid_id, "grid id")?;
        let gid = repr_str(&self.grid_id);
        let ndim = self.shape.len();
        if self.axis_names.len() != ndim || self.axis_kinds.len() != ndim {
            return Err(Error::coded("E109", format!("grid {gid}: axis_names/axis_kinds must have {ndim} entries")));
        }
        if self.shape.iter().any(|s| *s <= 0) {
            return Err(Error::coded(
                "E109",
                format!("grid {gid}: shape {} must be positive", repr_int_tuple(&self.shape)),
            ));
        }
        let mut unknown: Vec<&String> = self.axis_kinds.iter().filter(|k| !AXIS_KINDS.contains(&k.as_str())).collect();
        if !unknown.is_empty() {
            unknown.sort();
            unknown.dedup();
            return Err(Error::coded(
                "E110",
                format!("grid {gid}: unknown axis kinds {}", crate::json::repr_list(&unknown)),
            ));
        }
        let n_spatial = self.n_spatial();
        if !(MIN_SPATIAL..=MAX_SPATIAL).contains(&n_spatial) {
            return Err(Error::coded("E110", format!("grid {gid}: {n_spatial} spatial axes; the spec allows 2 or 3")));
        }
        let count = |k: &str| self.axis_kinds.iter().filter(|v| *v == k).count();
        if count("time") > 1 || count("channel") > 1 {
            return Err(Error::coded("E110", format!("grid {gid}: at most one time and one channel axis")));
        }
        let spatial_idx = self.spatial_axes();
        let expected: Vec<usize> = (ndim - n_spatial..ndim).collect();
        if spatial_idx != expected {
            return Err(Error::coded(
                "E103",
                format!(
                    "grid {gid}: spatial axes {} must be contiguous and trailing (expected {})",
                    repr_int_tuple(&spatial_idx),
                    repr_int_tuple(&expected)
                ),
            ));
        }
        if self.spacing.len() != n_spatial || self.origin.len() != n_spatial {
            return Err(Error::coded("E109", format!("grid {gid}: spacing/origin must have {n_spatial} entries")));
        }
        if self.spacing.iter().any(|v| *v <= 0.0 || v.is_nan()) {
            return Err(Error::coded(
                "E104",
                format!("grid {gid}: spacing {} must be > 0", float_tuple(&self.spacing)),
            ));
        }
        if self.direction.dim() != (n_spatial, n_spatial) {
            let (r, c) = self.direction.dim();
            return Err(Error::coded(
                "E109",
                format!("grid {gid}: direction must be {n_spatial}x{n_spatial}, got ({r}, {c})"),
            ));
        }
        check_orthonormal(&self.direction, ORTHONORMAL_TOL, &format!("grid {gid} direction"))?;
        if let (Some(axis), Some(values)) = (self.time_axis(), &self.time_values) {
            let extent = self.shape[axis];
            if values.len() as i64 != extent {
                return Err(Error::coded(
                    "E109",
                    format!("grid {gid}: time_values has {} entries for a time axis of extent {extent}", values.len()),
                ));
            }
        }
        if let Some(hint) = &self.patch_hint {
            if hint.len() != n_spatial {
                return Err(Error::coded("E109", format!("grid {gid}: patch_hint must have {n_spatial} entries")));
            }
        }
        if let Some(hint) = &self.chunk_hint {
            if hint.len() != ndim {
                return Err(Error::coded("E109", format!("grid {gid}: chunk_hint must have {ndim} entries")));
            }
        }
        Ok(())
    }

    // -- axis views -------------------------------------------------------

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Stored indices of the spatial axes, ascending.
    pub fn spatial_axes(&self) -> Vec<usize> {
        self.axis_kinds.iter().enumerate().filter(|(_, k)| *k == "spatial").map(|(i, _)| i).collect()
    }

    pub fn n_spatial(&self) -> usize {
        self.axis_kinds.iter().filter(|k| *k == "spatial").count()
    }

    /// The full shape as `usize`.
    pub fn shape_usize(&self) -> Vec<usize> {
        self.shape.iter().map(|s| (*s).max(0) as usize).collect()
    }

    pub fn spatial_shape(&self) -> Vec<usize> {
        self.spatial_axes().into_iter().map(|i| self.shape[i].max(0) as usize).collect()
    }

    pub fn spatial_names(&self) -> Vec<String> {
        self.spatial_axes().into_iter().map(|i| self.axis_names[i].clone()).collect()
    }

    pub fn channel_axis(&self) -> Option<usize> {
        self.axis_kinds.iter().position(|k| k == "channel")
    }

    pub fn time_axis(&self) -> Option<usize> {
        self.axis_kinds.iter().position(|k| k == "time")
    }

    /// Number of voxels in the spatial lattice.
    pub fn n_voxels(&self) -> usize {
        self.spatial_shape().iter().product()
    }

    // -- geometry ---------------------------------------------------------

    /// The `(S+1, S+1)` index-to-world affine (spec §3.3).
    pub fn affine(&self) -> Array2<f64> {
        build_affine(&self.spacing, &self.origin, &self.direction).expect("a checked grid has a consistent affine")
    }

    /// Index -> world for flat points, `S` per point.
    pub fn index_to_world(&self, indices: &[f64]) -> Vec<f64> {
        index_to_world(&self.affine(), indices)
    }

    /// World -> index for flat points, `S` per point.
    pub fn world_to_index(&self, points: &[f64]) -> Result<Vec<f64>> {
        world_to_index(&self.affine(), points)
    }

    /// Continuous index bounds `(S, 2)` (flat): every axis spans `[-0.5, n - 0.5]`.
    pub fn extent(&self) -> Vec<f64> {
        self.spatial_shape().iter().flat_map(|n| [-0.5, *n as f64 - 0.5]).collect()
    }

    /// Field of view per spatial axis, in `units`.
    pub fn physical_size(&self) -> Vec<f64> {
        self.spatial_shape().iter().zip(&self.spacing).map(|(n, s)| *n as f64 * s).collect()
    }

    /// The factor that carries this grid's world coordinates into `other`'s
    /// units (§3.5): millimetres per unit of one over the other's.
    ///
    /// E414 when no factor relates their numbers: two grids in different
    /// `coord_system`s, which §3.3 rule 4 does not compare without a transform
    /// --- a code names a convention, and only some codes name one unambiguously
    /// --- or an uncalibrated (`px`) grid against a calibrated one.
    pub fn world_scale_into(&self, other: &Grid) -> Result<f64> {
        if self.coord_system != other.coord_system {
            return Err(Error::coded(
                "E414",
                format!(
                    "grid {} states world coordinates in {} and grid {} in {}; two grids' world coordinates are \
                     compared in one coord_system only (§3.3 rule 4), so write them in one",
                    repr_str(&self.grid_id),
                    self.coord_system,
                    repr_str(&other.grid_id),
                    other.coord_system
                ),
            ));
        }
        if self.units == other.units {
            return Ok(1.0);
        }
        match (mm_per_unit(&self.units), mm_per_unit(&other.units)) {
            (Some(from), Some(to)) => Ok(from / to),
            _ => Err(Error::coded(
                "E414",
                format!(
                    "grid {} is in {} and grid {} in {}; an uncalibrated grid's world coordinates scale to no \
                     other unit (§3.5)",
                    repr_str(&self.grid_id),
                    repr_str(&self.units),
                    repr_str(&other.grid_id),
                    repr_str(&other.units)
                ),
            )),
        }
    }

    /// Flat world points of this grid, in `other`'s units.
    ///
    /// World coordinates are numbers in a grid's units (§3.5), so one frame's
    /// points in millimetres and in metres are different numbers: read as they
    /// were, a point at 1 mm on a grid of 0.001 m voxels went to index 1000,
    /// and two raters' equal boxes scored an F1 of 0 (N10 of the round-3
    /// audit).  E414 when nothing relates the two (`world_scale_into`).
    pub fn world_into(&self, other: &Grid, points: &[f64]) -> Result<Vec<f64>> {
        let scale = self.world_scale_into(other)?;
        if scale == 1.0 {
            return Ok(points.to_vec());
        }
        Ok(points.iter().map(|v| v * scale).collect())
    }

    /// Whether two grids are physically comparable without a transform (§3.3.4).
    ///
    /// A grid without a `frame_uid` shares a frame with nothing --- including
    /// another grid without one.
    pub fn comparable_with(&self, other: &Grid) -> bool {
        self.frame_uid.is_some() && self.frame_uid == other.frame_uid && self.coord_system == other.coord_system
    }

    /// Whether two grids describe the same lattice, within `tol`.
    pub fn is_congruent(&self, other: &Grid, tol: f64) -> bool {
        self.shape == other.shape
            && self.axis_kinds == other.axis_kinds
            && allclose(&self.spacing, &other.spacing, tol, 0.0)
            && allclose(&self.origin, &other.origin, tol, 0.0)
            && self.direction.dim() == other.direction.dim()
            && allclose(
                self.direction.as_standard_layout().as_slice().unwrap(),
                other.direction.as_standard_layout().as_slice().unwrap(),
                tol,
                0.0,
            )
    }

    /// The grid with a different timepoint (a copy).
    pub fn with_timepoint(&self, timepoint: Option<String>) -> Grid {
        let mut g = self.clone();
        g.timepoint = timepoint;
        g
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let tp = match &self.timepoint {
            Some(t) if !t.is_empty() => format!(", timepoint={}", repr_str(t)),
            _ => String::new(),
        };
        let rounded: Vec<f64> = self.spacing.iter().map(|v| round_to(*v, 4)).collect();
        format!(
            "Grid({}, shape={}, spacing={}, {}/{}{tp})",
            repr_str(&self.grid_id),
            repr_int_tuple(&self.shape),
            float_tuple(&rounded),
            self.coord_system,
            self.units
        )
    }

    // -- serialization ----------------------------------------------------

    /// The grid's HDF5 attributes (spec §3.2), in write order.
    pub fn attrs(&self) -> Vec<(String, AttrValue)> {
        let n = self.n_spatial();
        let mut out: Vec<(String, AttrValue)> = vec![
            ("shape".into(), AttrValue::ints(&self.shape)),
            ("axis_names".into(), AttrValue::strs(&self.axis_names)),
            ("axis_kinds".into(), AttrValue::strs(&self.axis_kinds)),
            ("spacing".into(), AttrValue::floats(&self.spacing)),
            ("origin".into(), AttrValue::floats(&self.origin)),
            ("direction".into(), AttrValue::matrix(n, n, self.direction.as_standard_layout().as_slice().unwrap())),
            ("coord_system".into(), AttrValue::Str(self.coord_system.clone())),
            ("units".into(), AttrValue::Str(self.units.clone())),
        ];
        if let Some(v) = &self.timepoint {
            out.push(("timepoint".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.frame_uid {
            out.push(("frame_uid".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.time_values {
            out.push(("time_values".into(), AttrValue::floats(v)));
        }
        if let Some(v) = &self.time_units {
            out.push(("time_units".into(), AttrValue::Str(v.clone())));
        }
        if let Some(v) = &self.chunk_hint {
            out.push(("chunk_hint".into(), AttrValue::ints(v)));
        }
        if let Some(v) = &self.patch_hint {
            out.push(("patch_hint".into(), AttrValue::ints(v)));
        }
        out.extend(self.extra.iter().cloned());
        out
    }

    /// JSON-safe description, for `medh5 info --json`.
    pub fn summary(&self) -> Value {
        json!({
            "id": self.grid_id,
            "shape": self.shape,
            "axis_names": self.axis_names,
            "axis_kinds": self.axis_kinds,
            "spacing": self.spacing,
            "origin": self.origin,
            "direction": self.direction.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>(),
            "coord_system": self.coord_system,
            "units": self.units,
            "timepoint": self.timepoint,
            "frame_uid": self.frame_uid,
            "physical_size": self.physical_size(),
        })
    }

    fn key_eq(&self, other: &Grid) -> bool {
        self.grid_id == other.grid_id
            && self.shape == other.shape
            && self.axis_names == other.axis_names
            && self.axis_kinds == other.axis_kinds
            && bits(&self.spacing) == bits(&other.spacing)
            && bits(&self.origin) == bits(&other.origin)
            && self.direction.dim() == other.direction.dim()
            && bits(self.direction.as_standard_layout().as_slice().unwrap())
                == bits(other.direction.as_standard_layout().as_slice().unwrap())
            && self.coord_system == other.coord_system
            && self.units == other.units
            && self.timepoint == other.timepoint
            && self.frame_uid == other.frame_uid
            && self.time_values.as_ref().map(|v| bits(v)) == other.time_values.as_ref().map(|v| bits(v))
            && self.time_units == other.time_units
    }
}

impl PartialEq for Grid {
    fn eq(&self, other: &Self) -> bool {
        self.key_eq(other)
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// `round(value, ndigits)` with correct decimal rounding.
pub fn round_to(value: f64, ndigits: usize) -> f64 {
    format!("{value:.ndigits$}").parse().unwrap_or(value)
}

/// `(1.5, 0.8)` as Python prints a tuple of floats.
pub fn float_tuple(values: &[f64]) -> String {
    let inner: Vec<String> = values.iter().map(|v| float_repr(*v)).collect();
    if inner.len() == 1 {
        format!("({},)", inner[0])
    } else {
        format!("({})", inner.join(", "))
    }
}

/// Write `grids/<grid_id>` as an empty group carrying the geometry.
pub fn write_grid(grids_group: &hdf5::Group, grid: &Grid) -> Result<hdf5::Group> {
    let group = grids_group.create_group(&grid.grid_id)?;
    for (name, value) in grid.attrs() {
        attrs::write(&group, &name, &value)?;
    }
    Ok(group)
}

fn required(group: &hdf5::Group, name: &str) -> Result<AttrValue> {
    attrs::require(group, name, "E109")
}

fn as_i64s(value: &AttrValue, name: &str) -> Result<Vec<i64>> {
    value.as_i64_vec().ok_or_else(|| Error::coded("E109", format!("attribute {} is not numeric", repr_str(name))))
}

fn as_f64s(value: &AttrValue, name: &str) -> Result<Vec<f64>> {
    value.as_f64_vec().ok_or_else(|| Error::coded("E109", format!("attribute {} is not numeric", repr_str(name))))
}

fn as_strs(value: &AttrValue) -> Vec<String> {
    match value {
        AttrValue::Str(s) => vec![s.clone()],
        AttrValue::Strs(v) => v.clone(),
        other => other.as_f64_vec().unwrap_or_default().iter().map(|f| crate::json::py_float(*f)).collect(),
    }
}

fn as_text(value: &AttrValue) -> String {
    value.as_str().unwrap_or_else(|| attrs::stringify_value(value))
}

/// Read one grid group back into a [`Grid`].
pub fn read_grid(group: &hdf5::Group, grid_id: Option<&str>) -> Result<Grid> {
    let gid = grid_id.map(str::to_string).unwrap_or_else(|| crate::h5::basename(&group.name()).to_string());
    let direction = {
        let value = required(group, "direction")?;
        match value.as_matrix() {
            Some((r, c, values)) => Array2::from_shape_vec((r, c), values).unwrap(),
            None => {
                return Err(Error::coded(
                    "E109",
                    format!("matrix attribute must be stored 2-D, got shape {}", repr_int_tuple(&value.shape())),
                ))
            }
        }
    };
    let mut extra = Vec::new();
    for name in attrs::names(group)? {
        if !SPEC_GRID_ATTRS.contains(&name.as_str()) {
            if let Some(v) = attrs::read(group, &name)? {
                extra.push((name, v));
            }
        }
    }
    let opt = |name: &str| -> Result<Option<AttrValue>> { attrs::read(group, name) };
    let grid = Grid {
        grid_id: gid,
        shape: as_i64s(&required(group, "shape")?, "shape")?,
        axis_names: as_strs(&required(group, "axis_names")?),
        axis_kinds: as_strs(&required(group, "axis_kinds")?),
        spacing: as_f64s(&required(group, "spacing")?, "spacing")?,
        origin: as_f64s(&required(group, "origin")?, "origin")?,
        direction,
        coord_system: as_text(&required(group, "coord_system")?),
        units: as_text(&required(group, "units")?),
        timepoint: opt("timepoint")?.map(|v| as_text(&v)),
        frame_uid: opt("frame_uid")?.map(|v| as_text(&v)),
        time_values: opt("time_values")?.map(|v| as_f64s(&v, "time_values")).transpose()?,
        time_units: opt("time_units")?.map(|v| as_text(&v)),
        chunk_hint: opt("chunk_hint")?.map(|v| as_i64s(&v, "chunk_hint")).transpose()?,
        patch_hint: opt("patch_hint")?.map(|v| as_i64s(&v, "patch_hint")).transpose()?,
        extra,
    };
    grid.check()?;
    Ok(grid)
}

/// Read every grid under `<sample root>/grids`, in name order.
pub fn read_grids(root: &hdf5::Group) -> Result<Vec<(String, Grid)>> {
    let Some(grids) = ops::child_group(root, "grids") else {
        return Err(Error::coded("E008", "sample has no `grids` group"));
    };
    let mut out = Vec::new();
    for name in ops::members(&grids)? {
        if let Some(group) = ops::child_group(&grids, &name) {
            let grid = read_grid(&group, Some(&name))?;
            out.push((name, grid));
        }
    }
    Ok(out)
}

/// Millimetres per unit of a calibrated grid (§3.2); `None` for `px`.
pub fn mm_per_unit(units: &str) -> Option<f64> {
    match units {
        "mm" => Some(1.0),
        "um" => Some(1e-3),
        "m" => Some(1e3),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Grid {
        Grid::from_spec(GridSpec {
            grid_id: "ct".into(),
            shape: vec![16, 24, 24],
            spacing: vec![1.5, 0.8, 0.8],
            origin: Some(vec![-12.0, -9.6, -9.6]),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn grid_defaults_and_repr() {
        let g = grid();
        assert_eq!(g.axis_names, vec!["z", "y", "x"]);
        assert_eq!(g.repr(), "Grid('ct', shape=(16, 24, 24), spacing=(1.5, 0.8, 0.8), LPS/mm)");
        assert_eq!(g.physical_size(), vec![24.0, 19.200000000000003, 19.200000000000003]);
    }

    #[test]
    fn grid_rules() {
        let mut spec =
            GridSpec { grid_id: "g".into(), shape: vec![4, 4], spacing: vec![0.0, 1.0], ..Default::default() };
        assert_eq!(Grid::from_spec(spec.clone()).unwrap_err().code(), Some("E104"));
        spec.spacing = vec![1.0, 1.0];
        spec.direction = Some(Array2::from_shape_vec((2, 2), vec![1.0, 0.1, 0.0, 1.0]).unwrap());
        let err = Grid::from_spec(spec).unwrap_err();
        assert_eq!(err.code(), Some("E102"));
        let spec = GridSpec {
            grid_id: "g".into(),
            shape: vec![4, 4, 4],
            axis_kinds: Some(vec!["spatial".into(), "channel".into(), "spatial".into()]),
            spacing: vec![1.0, 1.0],
            ..Default::default()
        };
        assert_eq!(Grid::from_spec(spec).unwrap_err().code(), Some("E103"));
    }

    fn calibrated(id: &str, coord_system: &str, units: &str) -> Grid {
        Grid::from_spec(GridSpec {
            grid_id: id.into(),
            shape: vec![4, 4, 4],
            spacing: vec![1.0, 1.0, 1.0],
            coord_system: Some(coord_system.into()),
            units: Some(units.into()),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn n10_world_coordinates_carry_into_another_grids_units() {
        let (mm, m, um) = (calibrated("a", "LPS", "mm"), calibrated("b", "LPS", "m"), calibrated("c", "LPS", "um"));
        assert_eq!(mm.world_into(&m, &[1.0, 2.0, 3.0]).unwrap(), vec![1e-3, 2e-3, 3e-3]);
        assert_eq!(m.world_into(&um, &[1.0, 0.0, -2.0]).unwrap(), vec![1e6, 0.0, -2e6]);
        assert_eq!(mm.world_into(&mm, &[1.5, 2.5, 3.5]).unwrap(), vec![1.5, 2.5, 3.5]);
    }

    #[test]
    fn n10_s3_3_rule_4_two_conventions_or_px_relate_by_no_factor() {
        let lps = calibrated("a", "LPS", "mm");
        for other in [calibrated("b", "RAS", "mm"), calibrated("c", "LPS", "px"), calibrated("d", "custom", "mm")] {
            assert_eq!(lps.world_into(&other, &[1.0, 1.0, 1.0]).unwrap_err().code(), Some("E414"));
            assert_eq!(other.world_scale_into(&lps).unwrap_err().code(), Some("E414"));
        }
        // Spelled alike, an uncalibrated grid or a custom convention is its own.
        let (px, custom) = (calibrated("p", "LPS", "px"), calibrated("q", "custom", "mm"));
        assert_eq!(px.world_scale_into(&px).unwrap(), 1.0);
        assert_eq!(custom.world_scale_into(&custom).unwrap(), 1.0);
    }
}
