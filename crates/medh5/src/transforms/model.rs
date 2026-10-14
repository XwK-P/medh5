//! The transform header (§10.1) and one reader type for every kind.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use super::apply::{
    as_points, check_points, folding_fraction, inside_extent, jacobian_determinant, linear_sample, refuse_outside,
    sample_field, to_world_vectors, CubicCache, CubicCoefficients, EXACT_INTEGER, EXTRAPOLATIONS,
};
use crate::array::{Index, NdArray, Slice};
use crate::geometry::grid::{Grid, MAX_SPATIAL, MIN_SPATIAL};
use crate::geometry::linalg::{det, inv};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::{Error, Result};

/// The transform kinds 1.0 defines (§10).
pub const TRANSFORM_KINDS: [&str; 5] = ["identity", "affine", "displacement", "bspline", "composite"];
/// How displacement components are expressed.
pub const VECTOR_SPACES: [&str; 2] = ["world", "index"];
/// Field interpolations.
pub const INTERPOLATIONS: [&str; 2] = ["linear", "cubic"];
/// The transform attributes the spec defines.
pub const SPEC_TRANSFORM_ATTRS: [&str; 18] = [
    "kind",
    "from_frame",
    "to_frame",
    "from_grid",
    "to_grid",
    "units",
    "invertible",
    "inverse_id",
    "prov",
    "metrics",
    "digest",
    "field_grid",
    "vector_space",
    "interpolation",
    "extrapolation",
    "cp_grid",
    "order",
    "components",
];
/// Tolerance on an affine's last row and singularity.
pub const LAST_ROW_TOL: f64 = 1e-9;
/// B-spline orders a reader evaluates.
pub const SUPPORTED_ORDERS: [i64; 2] = [1, 3];
/// The B-spline order when the file does not say.
pub const DEFAULT_ORDER: i64 = 3;
/// Displacement magnitude below which `float16` costs ~5e-4 relative precision.
pub const FLOAT16_SAFE_VOXELS: f64 = 64.0;

/// `grid 'g' is in 'm'` for each of `grids` not in `units`: what E506
/// names, for a transform in `units` relating their frames (§10.1).
pub fn units_disagreeing<'a>(grids: impl IntoIterator<Item = &'a Grid>, units: &str) -> Vec<String> {
    grids
        .into_iter()
        .filter(|g| g.units != units)
        .map(|g| format!("grid {} is in {}", repr_str(&g.grid_id), repr_str(&g.units)))
        .collect()
}

/// `S` of an affine `matrix` of `shape`: `(S+1, S+1)`, for the 2 or 3
/// spatial axes of a grid (§3.1, §10.3), or E504.
///
/// Every use of a matrix indexes it by its row count, so one of another
/// shape was a panic rather than an error: a `(5, 2)` matrix was sliced past
/// its last column, a `(4, 3)` one read past it, and an empty one underflowed.
pub fn affine_dimension(shape: &[usize]) -> Result<usize> {
    match *shape {
        [rows, cols] if rows == cols && (MIN_SPATIAL + 1..=MAX_SPATIAL + 1).contains(&rows) => Ok(rows - 1),
        _ => Err(Error::coded(
            "E504",
            format!(
                "affine `matrix` must be (S+1, S+1) for the {MIN_SPATIAL} or {MAX_SPATIAL} spatial axes of a grid \
                 (§10.3), got {}",
                repr_int_tuple(shape)
            ),
        )),
    }
}

/// Refuse the affine stored in `group` unless its matrix is `(S+1, S+1)` for
/// the spatial axes of every one of `grids`, the grids it relates (§10.3,
/// E504); a transform of another kind passes.
///
/// A `(3, 3)` matrix between two 3-D frames was written and validated, and
/// was applied to 3-D points by regrouping their coordinates in twos.
pub fn check_affine_rank<'a>(
    transform_id: &str,
    group: &hdf5::Group,
    grids: impl IntoIterator<Item = &'a Grid>,
) -> Result<()> {
    if attrs::get_str(group, "kind")?.as_deref() != Some("affine") {
        return Ok(());
    }
    let Some(ds) = ops::child_dataset(group, "matrix") else { return Ok(()) };
    let s = affine_dimension(&ds.shape())?;
    let wrong: Vec<String> = grids
        .into_iter()
        .filter(|g| g.n_spatial() != s)
        .map(|g| format!("grid {} has {} spatial axes", repr_str(&g.grid_id), g.n_spatial()))
        .collect();
    if wrong.is_empty() {
        return Ok(());
    }
    Err(Error::coded(
        "E504",
        format!(
            "transform {}: `matrix` is ({2}, {2}), an affine of {s} spatial axes, and {}: an affine maps a point \
             of the grids it relates, one coordinate per spatial axis (§10.3)",
            repr_str(transform_id),
            wrong.join(", "),
            s + 1
        ),
    ))
}

/// The attribute header every transform carries (spec §10.1).
#[derive(Debug, Clone, PartialEq)]
pub struct TransformHeader {
    pub kind: String,
    pub from_frame: String,
    pub to_frame: String,
    pub units: String,
    pub from_grid: Option<String>,
    pub to_grid: Option<String>,
    pub invertible: Option<bool>,
    pub inverse_id: Option<String>,
    pub prov: Option<String>,
    pub metrics: Option<String>,
    pub extra: Vec<(String, AttrValue)>,
}

impl TransformHeader {
    /// A header with the given kind and frames, every other field defaulted.
    pub fn new(kind: &str, from_frame: &str, to_frame: &str) -> Result<TransformHeader> {
        let header = TransformHeader {
            kind: kind.into(),
            from_frame: from_frame.into(),
            to_frame: to_frame.into(),
            units: "mm".into(),
            from_grid: None,
            to_grid: None,
            invertible: None,
            inverse_id: None,
            prov: None,
            metrics: None,
            extra: Vec::new(),
        };
        header.check()?;
        Ok(header)
    }

    /// Validate §10.1.
    pub fn check(&self) -> Result<()> {
        if !TRANSFORM_KINDS.contains(&self.kind.as_str()) {
            return Err(Error::coded(
                "E502",
                format!(
                    "unknown transform kind {}; expected one of {}",
                    repr_str(&self.kind),
                    repr_list(&TRANSFORM_KINDS)
                ),
            ));
        }
        if self.from_frame.is_empty() || self.to_frame.is_empty() {
            return Err(Error::coded("E502", "a transform requires both `from_frame` and `to_frame`"));
        }
        Ok(())
    }

    /// The attributes this header writes.
    pub fn attrs(&self) -> Vec<(String, AttrValue)> {
        let mut out = vec![
            ("kind".to_string(), AttrValue::Str(self.kind.clone())),
            ("from_frame".to_string(), AttrValue::Str(self.from_frame.clone())),
            ("to_frame".to_string(), AttrValue::Str(self.to_frame.clone())),
            ("units".to_string(), AttrValue::Str(self.units.clone())),
        ];
        for (key, value) in [
            ("from_grid", &self.from_grid),
            ("to_grid", &self.to_grid),
            ("inverse_id", &self.inverse_id),
            ("prov", &self.prov),
            ("metrics", &self.metrics),
        ] {
            if let Some(v) = value {
                out.push((key.to_string(), AttrValue::Str(v.clone())));
            }
        }
        if let Some(v) = self.invertible {
            out.push(("invertible".to_string(), AttrValue::Bool(v)));
        }
        out.extend(self.extra.iter().cloned());
        out
    }

    /// Read the header of a transform group.
    pub fn read(group: &hdf5::Group) -> Result<TransformHeader> {
        let text = |v: AttrValue| v.as_str().unwrap_or_else(|| attrs::stringify_value(&v));
        let header = TransformHeader {
            kind: text(attrs::require(group, "kind", "E502")?),
            from_frame: text(attrs::require(group, "from_frame", "E502")?),
            to_frame: text(attrs::require(group, "to_frame", "E502")?),
            units: attrs::get_str(group, "units")?.unwrap_or_else(|| "mm".into()),
            from_grid: attrs::get_str(group, "from_grid")?,
            to_grid: attrs::get_str(group, "to_grid")?,
            invertible: attrs::get_bool(group, "invertible")?,
            inverse_id: attrs::get_str(group, "inverse_id")?,
            prov: attrs::get_str(group, "prov")?,
            metrics: attrs::get_str(group, "metrics")?,
            extra: Vec::new(),
        };
        header.check()?;
        Ok(header)
    }
}

/// Transform groups by id, for resolving `inverse_id` and composite components.
pub type Siblings = IndexMap<String, hdf5::Group>;

/// What a transform evaluates.
#[derive(Debug, Clone)]
pub enum Body {
    /// A group stored in the file, evaluated by its `kind`.
    Stored,
    /// `T⁻¹` of a transform that can actually be inverted.
    Inverse(Box<Transform>),
    /// An in-memory composition: the result of resolving a multi-hop path.
    Chain(Vec<Transform>),
}

/// A mapping between two frames of reference (§10).
#[derive(Debug, Clone)]
pub struct Transform {
    pub transform_id: String,
    pub group: hdf5::Group,
    pub header: TransformHeader,
    grids: Arc<IndexMap<String, Grid>>,
    siblings: Arc<Siblings>,
    pub body: Body,
    /// A cubic field's prefiltered coefficients, made at its first
    /// evaluation and shared by every clone, within a process-wide budget: a
    /// handle reads a file that cannot change under it (an amendment
    /// replaces the inode).
    cubic: Arc<CubicCache>,
}

fn points_of(points: &ArrayD<f64>, dim: usize) -> Result<Array2<f64>> {
    as_points(points, dim)
}

fn reshape_like(values: Array2<f64>, like: &ArrayD<f64>) -> Result<ArrayD<f64>> {
    Ok(values.into_dyn().into_shape_with_order(IxDyn(like.shape()))?)
}

impl Transform {
    /// Open a stored transform group.
    pub fn open(
        transform_id: &str,
        group: hdf5::Group,
        grids: Arc<IndexMap<String, Grid>>,
        siblings: Arc<Siblings>,
    ) -> Result<Transform> {
        let header = TransformHeader::read(&group)?;
        Ok(Transform {
            transform_id: transform_id.to_string(),
            group,
            header,
            grids,
            siblings,
            body: Body::Stored,
            cubic: Arc::default(),
        })
    }

    /// `T⁻¹` of `inner`, which the caller has checked [`can_invert`].
    pub fn inverse_of(inner: Transform) -> Transform {
        let header = TransformHeader {
            kind: inner.header.kind.clone(),
            from_frame: inner.header.to_frame.clone(),
            to_frame: inner.header.from_frame.clone(),
            units: inner.header.units.clone(),
            from_grid: None,
            to_grid: None,
            invertible: Some(true),
            inverse_id: None,
            prov: None,
            metrics: None,
            extra: Vec::new(),
        };
        Transform {
            transform_id: format!("{}\u{207B}\u{00B9}", inner.transform_id),
            group: inner.group.clone(),
            header,
            grids: inner.grids.clone(),
            siblings: inner.siblings.clone(),
            body: Body::Inverse(Box::new(inner)),
            cubic: Arc::default(),
        }
    }

    /// An in-memory chain of steps, refusing steps in different units or
    /// whose frames do not meet (E501), as a stored composite's are
    /// ([`Transform::check_chain`]).
    ///
    /// Steps given in the wrong order were chained anyway: the chain took
    /// its frames from its first and last steps and mapped points through
    /// a path that does not exist.
    pub fn chain(steps: Vec<Transform>) -> Result<Transform> {
        if steps.is_empty() {
            return Err(Error::invalid("a transform chain needs at least one step"));
        }
        let mut units: Vec<&str> = steps.iter().map(|t| t.units()).collect();
        units.sort_unstable();
        units.dedup();
        if units.len() > 1 {
            return Err(Error::coded(
                "E501",
                format!(
                    "cannot chain transforms in different units: {}",
                    steps
                        .iter()
                        .map(|t| format!("{} in {}", repr_str(&t.transform_id), repr_str(t.units())))
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        }
        let breaks = frame_breaks(&steps);
        if !breaks.is_empty() {
            return Err(Error::coded(
                "E501",
                format!(
                    "cannot chain transforms whose frames do not meet: {}; each step must start in the frame the \
                     step before it ends in",
                    breaks.join("; ")
                ),
            ));
        }
        let first = &steps[0];
        let header = TransformHeader {
            kind: "composite".into(),
            from_frame: first.header.from_frame.clone(),
            to_frame: steps[steps.len() - 1].header.to_frame.clone(),
            units: first.header.units.clone(),
            from_grid: None,
            to_grid: None,
            invertible: None,
            inverse_id: None,
            prov: None,
            metrics: None,
            extra: Vec::new(),
        };
        Ok(Transform {
            transform_id: steps.iter().map(|t| t.transform_id.as_str()).collect::<Vec<_>>().join(" -> "),
            group: first.group.clone(),
            header,
            grids: first.grids.clone(),
            siblings: first.siblings.clone(),
            body: Body::Chain(steps),
            cubic: Arc::default(),
        })
    }

    // -- header passthrough -------------------------------------------------

    pub fn kind(&self) -> &str {
        &self.header.kind
    }

    pub fn from_frame(&self) -> &str {
        &self.header.from_frame
    }

    pub fn to_frame(&self) -> &str {
        &self.header.to_frame
    }

    pub fn units(&self) -> &str {
        &self.header.units
    }

    pub fn prov(&self) -> Option<&str> {
        self.header.prov.as_deref()
    }

    pub fn metrics_key(&self) -> Option<&str> {
        self.header.metrics.as_deref()
    }

    pub fn grids(&self) -> &IndexMap<String, Grid> {
        &self.grids
    }

    /// The reader class name the Python bindings expose.
    pub fn class_name(&self) -> &'static str {
        match &self.body {
            Body::Inverse(_) => "InverseTransform",
            Body::Chain(_) => "ChainTransform",
            Body::Stored => match self.kind() {
                "identity" => "IdentityTransform",
                "affine" => "AffineTransform",
                "displacement" => "DisplacementTransform",
                "bspline" => "BSplineTransform",
                _ => "CompositeTransform",
            },
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        format!(
            "{}({}, {} -> {})",
            self.class_name(),
            repr_str(&self.transform_id),
            repr_str(self.from_frame()),
            repr_str(self.to_frame())
        )
    }

    /// The steps of a resolved chain.
    pub fn steps(&self) -> Option<&[Transform]> {
        match &self.body {
            Body::Chain(steps) => Some(steps),
            _ => None,
        }
    }

    /// Whether an inverse is available --- declared, analytic, or neither.
    pub fn is_invertible(&self) -> bool {
        match &self.body {
            Body::Inverse(_) => true,
            Body::Chain(steps) => steps.iter().all(Transform::is_invertible),
            Body::Stored => match self.kind() {
                "identity" => true,
                "affine" => match self.header.invertible {
                    Some(v) => v,
                    None => match self.linear_det() {
                        Ok(d) => d.abs() > LAST_ROW_TOL,
                        Err(_) => false,
                    },
                },
                "composite" => {
                    if let Some(v) = self.header.invertible {
                        return v;
                    }
                    if self.header.inverse_id.is_some() {
                        return true;
                    }
                    match self.components() {
                        Ok(c) => c.iter().all(Transform::is_invertible),
                        Err(_) => false,
                    }
                }
                _ => self.header.invertible.unwrap_or(self.header.inverse_id.is_some()),
            },
        }
    }

    /// The timepoints this transform relates, from the grids in its frames.
    pub fn timepoints(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for frame in [self.from_frame(), self.to_frame()] {
            for grid in self.grids.values() {
                if grid.frame_uid.as_deref() == Some(frame) {
                    if let Some(tp) = grid.timepoint.as_deref().filter(|t| !t.is_empty()) {
                        if !out.iter().any(|t| t == tp) {
                            out.push(tp.to_string());
                        }
                        break;
                    }
                }
            }
        }
        out
    }

    /// A representative grid living in `frame`, if the sample has one.
    pub fn grid_in(&self, frame: &str) -> Option<&Grid> {
        let named = if frame == self.from_frame() { &self.header.from_grid } else { &self.header.to_grid };
        if let Some(g) = named.as_deref().and_then(|n| self.grids.get(n)) {
            return Some(g);
        }
        self.grids.values().find(|g| g.frame_uid.as_deref() == Some(frame))
    }

    /// The stored inverse transform, when the file carries one.
    pub fn inverse(&self) -> Result<Option<Transform>> {
        let Some(target) = self.header.inverse_id.as_deref() else {
            return Ok(None);
        };
        if !matches!(self.body, Body::Stored) {
            return Ok(None);
        }
        match self.siblings.get(target) {
            None => Ok(None),
            Some(group) => Ok(Some(Transform::open(target, group.clone(), self.grids.clone(), self.siblings.clone())?)),
        }
    }

    /// Map `(..., S)` world points from `from_frame` to `to_frame`.
    pub fn transform_points(&self, points: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        match &self.body {
            Body::Chain(steps) => {
                let mut values = points.clone();
                for step in steps {
                    values = step.transform_points(&values)?;
                }
                Ok(values)
            }
            Body::Inverse(inner) => {
                if inner.kind() == "identity" && matches!(inner.body, Body::Stored) {
                    return Ok(points.clone());
                }
                if let Some(stored) = stored_inverse(inner)? {
                    return stored.transform_points(points);
                }
                if inner.kind() == "affine" && matches!(inner.body, Body::Stored) {
                    return inner.inverse_points(points);
                }
                Err(Error::invalid(format!(
                    "transform {} of kind {} has no analytic inverse and declares no `inverse_id`; approximating \
                     one would report an accuracy nobody measured",
                    repr_str(&inner.transform_id),
                    repr_str(inner.kind())
                )))
            }
            Body::Stored => match self.kind() {
                "identity" => Ok(points.clone()),
                "affine" => self.apply_matrix(&self.matrix()?, points),
                "displacement" | "bspline" => {
                    let displacement = self.displacement_at(points)?;
                    Ok(points + &displacement)
                }
                _ => {
                    let problems = self.check_chain();
                    if !problems.is_empty() {
                        return Err(Error::coded(
                            "E501",
                            format!(
                                "composite {} has a broken frame chain: {}",
                                repr_str(&self.transform_id),
                                problems.join("; ")
                            ),
                        ));
                    }
                    let mut values = points.clone();
                    for component in self.components()? {
                        values = component.transform_points(&values)?;
                    }
                    Ok(values)
                }
            },
        }
    }

    fn require_kind(&self, kind: &str, what: &str) -> Result<()> {
        if self.kind() == kind && matches!(self.body, Body::Stored) {
            Ok(())
        } else {
            Err(Error::Type(format!(
                "{what} is defined on `{kind}` transforms; {} is {}",
                repr_str(&self.transform_id),
                self.class_name()
            )))
        }
    }

    fn missing(&self, what: &str, code: &str) -> Error {
        Error::coded(code, format!("transform {}: {what}", repr_str(&self.transform_id)))
    }

    /// A parameter dataset of the stored transform, or `None` when absent.
    /// A closed file is refused: HDF5 finds no member in one, which read as
    /// a transform without its parameters (E502).
    fn parameters(&self, name: &str) -> Result<Option<hdf5::Dataset>> {
        crate::h5::alive(&self.group)?;
        Ok(ops::child_dataset(&self.group, name))
    }

    // -- affine (§10.3) -------------------------------------------------------

    /// The homogeneous `(S+1, S+1)` world-to-world matrix; one of another
    /// shape is refused (E504, [`affine_dimension`]).
    pub fn matrix(&self) -> Result<Array2<f64>> {
        self.require_kind("affine", "matrix")?;
        let ds =
            self.parameters("matrix")?.ok_or_else(|| self.missing("`affine` requires a `matrix` dataset", "E502"))?;
        let n = affine_dimension(&ds.shape()).map_err(|e| self.missing(e.message(), "E504"))? + 1;
        Ok(data::read(&ds)?.to_f64().into_shape_with_order((n, n))?)
    }

    /// Spatial dimensionality of an affine.
    pub fn n_spatial(&self) -> Result<usize> {
        Ok(self.matrix()?.nrows() - 1)
    }

    fn linear_det(&self) -> Result<f64> {
        let m = self.matrix()?;
        let dim = m.nrows() - 1;
        Ok(det(&m.slice(ndarray::s![..dim, ..dim]).to_owned()))
    }

    fn apply_matrix(&self, matrix: &Array2<f64>, points: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let dim = matrix.nrows() - 1;
        let flat = points_of(points, dim)?;
        let linear = matrix.slice(ndarray::s![..dim, ..dim]);
        let mut out = flat.dot(&linear.t());
        for mut row in out.outer_iter_mut() {
            for (r, v) in row.iter_mut().enumerate() {
                *v += matrix[[r, dim]];
            }
        }
        reshape_like(out, points)
    }

    /// `T⁻¹` as a matrix, computed rather than stored.
    pub fn inverse_matrix(&self) -> Result<Array2<f64>> {
        inv(&self.matrix()?)
    }

    /// Map points through the analytic inverse of an affine.
    pub fn inverse_points(&self, points: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let m = self.inverse_matrix()?;
        self.apply_matrix(&m, points)
    }

    /// An affine's (constant) Jacobian determinant.
    pub fn jacobian_determinant_value(&self) -> Result<f64> {
        self.linear_det()
    }

    // -- displacement (§10.4) ---------------------------------------------------

    /// The `field` dataset of a displacement transform: `(S, *spatial)` on
    /// its field grid (§10.4), or E503.
    ///
    /// Every sampler indexes the field by the grid's axes, so a field of
    /// another shape was a panic rather than an error.
    pub fn field(&self) -> Result<hdf5::Dataset> {
        let ds = self.field_dataset()?;
        let grid = self.field_grid()?;
        let mut expected = vec![grid.n_spatial()];
        expected.extend(grid.spatial_shape());
        let shape = ds.shape();
        if shape != expected {
            return Err(self.missing(
                &format!(
                    "`field` is {}, not the (S, *spatial) {} of field grid {} (§10.4)",
                    repr_int_tuple(&shape),
                    repr_int_tuple(&expected),
                    repr_str(&grid.grid_id)
                ),
                "E503",
            ));
        }
        Ok(ds)
    }

    /// The stored `field` dataset, whatever its shape.
    fn field_dataset(&self) -> Result<hdf5::Dataset> {
        self.require_kind("displacement", "field")?;
        self.parameters("field")?.ok_or_else(|| self.missing("`displacement` requires a `field` dataset", "E502"))
    }

    /// The id of the grid the field is sampled on.
    pub fn field_grid_id(&self) -> Result<String> {
        attrs::get_str(&self.group, "field_grid")?
            .ok_or_else(|| self.missing("`displacement` requires `field_grid`", "E503"))
    }

    /// The grid the field is sampled on.
    pub fn field_grid(&self) -> Result<&Grid> {
        let gid = self.field_grid_id()?;
        self.grids.get(&gid).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("transform {}: field grid {} does not exist", repr_str(&self.transform_id), repr_str(&gid)),
            )
        })
    }

    /// How displacement components are expressed.
    pub fn vector_space(&self) -> Result<String> {
        Ok(attrs::get_str(&self.group, "vector_space")?.unwrap_or_else(|| "world".into()))
    }

    /// The declared interpolation.
    pub fn interpolation(&self) -> Result<String> {
        Ok(attrs::get_str(&self.group, "interpolation")?.unwrap_or_else(|| "linear".into()))
    }

    /// The declared extrapolation.
    pub fn extrapolation(&self) -> Result<String> {
        Ok(attrs::get_str(&self.group, "extrapolation")?.unwrap_or_else(|| "zero".into()))
    }

    /// Read the field, or one component, or one ROI --- in a single call.
    pub fn read_field(&self, roi: Option<&[Slice]>, component: Option<usize>) -> Result<NdArray> {
        let ds = self.field()?;
        let ndim = ds.ndim();
        let window: Vec<Index> = match roi {
            Some(r) => r.iter().map(|s| Index::Slice(*s)).collect(),
            None => vec![Index::Slice(Slice::full()); ndim.saturating_sub(1)],
        };
        let mut index = vec![match component {
            Some(c) => Index::At(c as i64),
            None => Index::Slice(Slice::full()),
        }];
        index.extend(window);
        data::read_region(&ds, &index)
    }

    /// World-space displacement `u(x)` at `(..., S)` world points.
    pub fn displacement_at(&self, points: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        match self.kind() {
            "displacement" => {
                let grid = self.field_grid()?;
                let flat = points_of(points, grid.n_spatial())?;
                let flat_world: Vec<f64> = flat.iter().copied().collect();
                let index = grid.world_to_index(&flat_world)?;
                let indices = Array2::from_shape_vec(flat.dim(), index)?;
                let raw = self.sample_indices(&indices)?;
                reshape_like(to_world_vectors(&raw, grid, &self.vector_space()?)?, points)
            }
            "bspline" => self.bspline_displacement(points),
            _ => Err(Error::Type(format!(
                "displacement_at() is defined on `displacement` and `bspline` transforms; {} is {}",
                repr_str(&self.transform_id),
                self.class_name()
            ))),
        }
    }

    /// Interpolate the stored field at continuous field indices, `(N, S)`,
    /// reading only the window the points touch (linear) --- kilobytes of a
    /// 512³ field rather than all of it --- or the whole field (cubic, whose
    /// spline coefficients are global, so a window would change the answer
    /// near its own edges).  The result equals sampling the whole field.
    /// A cubic field is read and prefiltered once per handle, at its first
    /// evaluation, while the process-wide budget for kept coefficients has
    /// room for it; past the budget, at every evaluation.
    pub fn sample_indices(&self, indices: &Array2<f64>) -> Result<Array2<f64>> {
        let ds = self.field()?;
        let shape = ds.shape();
        let spatial: Vec<usize> = shape[1..].to_vec();
        check_points(&spatial, indices)?;
        let interpolation = self.interpolation()?;
        let extrapolation = self.extrapolation()?;
        if interpolation == "cubic" && indices.nrows() > 0 && EXTRAPOLATIONS.contains(&extrapolation.as_str()) {
            if extrapolation == "error" {
                refuse_outside(&inside_extent(&spatial, indices))?;
            }
            return self.cubic.sample(indices, || CubicCoefficients::new(&data::read(&ds)?.to_f64(), &extrapolation));
        }
        if interpolation != "linear" || indices.nrows() == 0 {
            let field = data::read(&ds)?.to_f64();
            return sample_field(&field, indices, &interpolation, &extrapolation);
        }
        let inside = inside_extent(&spatial, indices);
        if extrapolation == "error" {
            refuse_outside(&inside)?;
        }
        let dim = spatial.len();
        let mut lo = vec![0i64; dim];
        let mut hi = vec![0i64; dim];
        for axis in 0..dim {
            // The window is taken over the points clamped onto the lattice,
            // as `linear_sample` clamps them --- NaN to the first sample, an
            // infinity to the edge --- so it holds every sample the whole
            // field would give them.  Unclamped, `-inf` overflowed the bound
            // or moved the window off the samples a finite point needed.
            let top = spatial[axis] as f64 - 1.0;
            let column = indices.column(axis);
            let clamped = column.iter().map(|v| v.max(0.0).min(top));
            let min = clamped.clone().fold(f64::INFINITY, f64::min);
            let max = clamped.fold(f64::NEG_INFINITY, f64::max);
            let n = spatial[axis] as i64;
            let l = (min.floor() as i64 - 1).clamp(0, (n - 1).max(0));
            let h = (max.ceil() as i64 + 2).clamp(l + 1, n.max(l + 1));
            lo[axis] = l.min((h - 2).max(0));
            hi[axis] = h;
        }
        let mut index = vec![Index::Slice(Slice::full())];
        index.extend(lo.iter().zip(&hi).map(|(a, b)| Index::Slice(Slice::new(*a, *b))));
        let block = data::read_region(&ds, &index)?.to_f64();
        let mut local = indices.clone();
        for mut row in local.outer_iter_mut() {
            for (axis, v) in row.iter_mut().enumerate() {
                *v -= lo[axis] as f64;
            }
        }
        let mut raw = linear_sample(&block, &local, "nearest")?;
        if extrapolation == "zero" {
            for (row, ok) in inside.iter().enumerate() {
                if !ok {
                    raw.row_mut(row).fill(0.0);
                }
            }
        }
        Ok(raw)
    }

    /// `det(I + du/dx)` per voxel --- values `<= 0` mark folding.
    ///
    /// `roi` is resolved as [`Transform::read_field`] resolves it: one slice
    /// per spatial axis of the field grid, whole axes after the last one
    /// given, NumPy's negative and out-of-range bounds.  The determinant is
    /// differenced on the region's own lattice, its edges one-sidedly as the
    /// field's are, so a slice with a step other than 1 is refused --- the
    /// step was ignored, and a strided region reported folding where there
    /// is none --- and so is an axis of fewer than 2 voxels.
    pub fn jacobian_determinant(&self, roi: Option<&[Slice]>) -> Result<ArrayD<f64>> {
        let grid = self.field_grid()?;
        let owner = format!("transform {}", repr_str(&self.transform_id));
        let bounds = roi_bounds(grid, roi.unwrap_or_default(), &owner)?;
        if let Some((axis, _)) = bounds.iter().enumerate().find(|(_, (a, b))| b - a < 2) {
            return Err(Error::Value(format!(
                "{owner}: the {} spans 1 voxel on axis {axis}, and a Jacobian takes a difference along every axis, \
                 which needs at least 2",
                if roi.is_some() { "region" } else { "field" }
            )));
        }
        let window: Vec<Slice> = bounds.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect();
        let field = self.read_field(Some(&window), None)?.to_f64();
        jacobian_determinant(&field, &cropped_grid(grid, &window)?, &self.vector_space()?)
    }

    /// Fraction of voxels that fold.
    pub fn folding_fraction(&self, roi: Option<&[Slice]>) -> Result<f64> {
        Ok(folding_fraction(&self.jacobian_determinant(roi)?))
    }

    /// Largest displacement magnitude, in the field's own component units.
    pub fn max_magnitude(&self) -> Result<f64> {
        let field = data::read(&self.field()?)?.to_f64();
        if field.is_empty() {
            return Ok(0.0);
        }
        let mut squares = ArrayD::<f64>::zeros(IxDyn(&field.shape()[1..]));
        for component in field.outer_iter() {
            squares.zip_mut_with(&component, |s, v| *s += v * v);
        }
        Ok(squares.iter().map(|v| v.sqrt()).fold(f64::NEG_INFINITY, f64::max))
    }

    // -- bspline (§10.5) --------------------------------------------------------

    /// `(S, *cp_shape)` control-point coefficients: `S` components on the
    /// `S` spatial axes of the control-point grid, with a point on every
    /// axis (§10.5), or E503.
    pub fn control_points(&self) -> Result<ArrayD<f64>> {
        let ds = self.control_points_dataset()?;
        let s = self.cp_grid()?.n_spatial();
        let shape = ds.shape();
        if shape.len() != s + 1 || shape[0] != s || shape[1..].contains(&0) {
            return Err(self.missing(
                &format!(
                    "`control_points` is {}, not (S, *cp_shape) with S = {s}, the spatial axes of its grid, and a \
                     point on every axis (§10.5)",
                    repr_int_tuple(&shape)
                ),
                "E503",
            ));
        }
        Ok(data::read(&ds)?.to_f64())
    }

    /// The stored `control_points` dataset, whatever its shape.
    fn control_points_dataset(&self) -> Result<hdf5::Dataset> {
        self.require_kind("bspline", "control_points")?;
        self.parameters("control_points")?.ok_or_else(|| self.missing("`bspline` requires `control_points`", "E502"))
    }

    /// The id of the control-point grid.
    pub fn cp_grid_id(&self) -> Result<String> {
        attrs::get_str(&self.group, "cp_grid")?.ok_or_else(|| self.missing("`bspline` requires `cp_grid`", "E503"))
    }

    /// The control-point grid.
    pub fn cp_grid(&self) -> Result<&Grid> {
        let gid = self.cp_grid_id()?;
        self.grids.get(&gid).ok_or_else(|| {
            Error::coded(
                "E101",
                format!(
                    "transform {}: control-point grid {} does not exist",
                    repr_str(&self.transform_id),
                    repr_str(&gid)
                ),
            )
        })
    }

    /// The B-spline order.
    pub fn order(&self) -> Result<i64> {
        Ok(attrs::get_i64(&self.group, "order")?.unwrap_or(DEFAULT_ORDER))
    }

    fn bspline_displacement(&self, points: &ArrayD<f64>) -> Result<ArrayD<f64>> {
        let grid = self.cp_grid()?;
        let coefficients = self.control_points()?;
        let order = self.order()?;
        // The order sizes every buffer below, so one this reader cannot
        // evaluate is refused before it does (E502), and so is a lattice too
        // small for it, which the writer refuses (E503).
        basis(order, 0.0)?;
        let dim = coefficients.shape()[0];
        let extent: Vec<i64> = coefficients.shape()[1..].iter().map(|n| *n as i64).collect();
        if extent.iter().any(|n| *n < order + 1) {
            return Err(self.missing(
                &format!(
                    "an order-{order} B-spline needs at least {} control points per axis, got {}",
                    order + 1,
                    repr_int_tuple(&coefficients.shape()[1..])
                ),
                "E503",
            ));
        }
        let flat = points_of(points, dim)?;
        let flat_world: Vec<f64> = flat.iter().copied().collect();
        let cp_index = Array2::from_shape_vec(flat.dim(), grid.world_to_index(&flat_world)?)?;
        let n = cp_index.nrows();
        let k = (order + 1) as usize;
        let mut out = Array2::<f64>::zeros((n, dim));
        for row in 0..n {
            let mut first = vec![0i64; dim];
            let mut weights = vec![vec![0.0; k]; dim];
            for axis in 0..dim {
                // Past `EXACT_INTEGER` a coordinate has no fraction to weigh
                // and every tap is beyond the lattice, so the clamp changes
                // nothing but the integer arithmetic, which overflowed.
                let c = cp_index[[row, axis]].clamp(-EXACT_INTEGER, EXACT_INTEGER);
                first[axis] = c.floor() as i64 - (order - 1).div_euclid(2);
                weights[axis] = basis(order, c - c.floor())?;
            }
            for combo in 0..k.pow(dim as u32) {
                let mut rest = combo;
                let mut offsets = vec![0usize; dim];
                for axis in (0..dim).rev() {
                    offsets[axis] = rest % k;
                    rest /= k;
                }
                let mut weight = 1.0;
                let mut index = vec![0usize; dim + 1];
                for axis in 0..dim {
                    weight *= weights[axis][offsets[axis]];
                    index[axis + 1] = (first[axis] + offsets[axis] as i64).clamp(0, extent[axis] - 1) as usize;
                }
                for c in 0..dim {
                    index[0] = c;
                    out[[row, c]] += weight * coefficients[IxDyn(&index)];
                }
            }
        }
        reshape_like(to_world_vectors(&out, grid, &self.vector_space()?)?, points)
    }

    /// Sample the spline onto a grid: the bridge to a dense `(S, *spatial)` field.
    pub fn to_displacement_field(&self, grid: &Grid) -> Result<ArrayD<f32>> {
        self.require_kind("bspline", "to_displacement_field()")?;
        let spatial = grid.spatial_shape();
        let s = grid.n_spatial();
        let total: usize = spatial.iter().product();
        let mut coords = Vec::with_capacity(total * s);
        for flat in 0..total {
            let mut rest = flat;
            let mut idx = vec![0usize; s];
            for axis in (0..s).rev() {
                idx[axis] = rest % spatial[axis];
                rest /= spatial[axis];
            }
            coords.extend(idx.iter().map(|v| *v as f64));
        }
        let world = grid.index_to_world(&coords)?;
        let world = ArrayD::from_shape_vec(IxDyn(&[total, s]), world)?;
        let displacement = self.displacement_at(&world)?;
        let mut shape = vec![s];
        shape.extend(&spatial);
        let mut out = ArrayD::<f32>::zeros(IxDyn(&shape));
        for (flat, row) in displacement.outer_iter().enumerate() {
            let mut rest = flat;
            let mut idx = vec![0usize; s + 1];
            for axis in (0..s).rev() {
                idx[axis + 1] = rest % spatial[axis];
                rest /= spatial[axis];
            }
            for (c, v) in row.iter().enumerate() {
                idx[0] = c;
                out[IxDyn(&idx)] = *v as f32;
            }
        }
        Ok(out)
    }

    // -- composite (§10.5) --------------------------------------------------------

    /// The ordered component ids of a composite.
    pub fn component_ids(&self) -> Result<Vec<String>> {
        Ok(attrs::get_strs(&self.group, "components")?.unwrap_or_default())
    }

    /// The chain, resolved and in application order.
    pub fn components(&self) -> Result<Vec<Transform>> {
        let cycle = composite_cycle(&self.transform_id, &self.siblings)?;
        if !cycle.is_empty() {
            return Err(Error::coded(
                "E501",
                format!(
                    "composite {} contains itself through {}; the chain never ends",
                    repr_str(&self.transform_id),
                    cycle.join(" -> ")
                ),
            ));
        }
        self.component_ids()?
            .iter()
            .map(|name| match self.siblings.get(name) {
                None => Err(Error::coded(
                    "E501",
                    format!(
                        "composite {} names component {}, which does not exist",
                        repr_str(&self.transform_id),
                        repr_str(name)
                    ),
                )),
                Some(group) => Transform::open(name, group.clone(), self.grids.clone(), self.siblings.clone()),
            })
            .collect()
    }

    /// Frame-chaining problems, as messages; empty means the chain is sound.
    pub fn check_chain(&self) -> Vec<String> {
        let chain = match self.components() {
            Ok(c) => c,
            Err(e) => return vec![e.message().to_string()],
        };
        if chain.is_empty() {
            return vec!["composite declares no components".into()];
        }
        let mut problems = Vec::new();
        if chain[0].from_frame() != self.from_frame() {
            problems.push(format!(
                "first component starts in {}, but the composite declares {}",
                repr_str(chain[0].from_frame()),
                repr_str(self.from_frame())
            ));
        }
        let last = &chain[chain.len() - 1];
        if last.to_frame() != self.to_frame() {
            problems.push(format!(
                "last component ends in {}, but the composite declares {}",
                repr_str(last.to_frame()),
                repr_str(self.to_frame())
            ));
        }
        problems.extend(frame_breaks(&chain));
        let mismatched: Vec<String> = chain
            .iter()
            .filter(|t| t.units() != self.units())
            .map(|t| format!("{} in {}", repr_str(&t.transform_id), repr_str(t.units())))
            .collect();
        if !mismatched.is_empty() {
            problems.push(format!(
                "composite declares units {} but {} --- a chain whose legs are in different units does not compose",
                repr_str(self.units()),
                mismatched.join(", ")
            ));
        }
        problems
    }

    // -- summaries ------------------------------------------------------------------

    /// JSON-safe description for `medh5 info`.
    pub fn summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.transform_id));
        out.insert("kind".into(), json!(self.kind()));
        out.insert("from_frame".into(), json!(self.from_frame()));
        out.insert("to_frame".into(), json!(self.to_frame()));
        out.insert("units".into(), json!(self.units()));
        out.insert("timepoints".into(), json!(self.timepoints()));
        out.insert("invertible".into(), json!(self.is_invertible()));
        out.insert("inverse_id".into(), json!(self.header.inverse_id));
        out.insert("metrics".into(), json!(self.metrics_key()));
        out.insert("prov".into(), json!(self.prov()));
        match &self.body {
            Body::Chain(steps) => {
                out.insert("steps".into(), json!(steps.iter().map(|s| s.transform_id.clone()).collect::<Vec<_>>()));
            }
            Body::Inverse(_) => {}
            Body::Stored => match self.kind() {
                "affine" => {
                    out.insert("jacobian_determinant".into(), crate::json::num(self.jacobian_determinant_value()?));
                }
                "displacement" => {
                    let ds = self.field_dataset()?;
                    out.insert("field_grid".into(), json!(self.field_grid_id()?));
                    out.insert("vector_space".into(), json!(self.vector_space()?));
                    out.insert("interpolation".into(), json!(self.interpolation()?));
                    out.insert("field_shape".into(), json!(ds.shape()));
                    out.insert("field_dtype".into(), json!(data::dtype(&ds)?.numpy_str()));
                }
                "bspline" => {
                    out.insert("cp_grid".into(), json!(self.cp_grid_id()?));
                    out.insert("order".into(), json!(self.order()?));
                    out.insert("vector_space".into(), json!(self.vector_space()?));
                    out.insert("cp_shape".into(), json!(self.control_points_dataset()?.shape()));
                }
                "composite" => {
                    out.insert("components".into(), json!(self.component_ids()?));
                    out.insert("chain_ok".into(), json!(self.check_chain().is_empty()));
                }
                _ => {}
            },
        }
        Ok(Value::Object(out))
    }
}

/// `'a' ends in 'M' but 'b' starts in 'F'` for each step of `steps` that
/// does not start in the frame the step before it ends in.
fn frame_breaks(steps: &[Transform]) -> Vec<String> {
    steps
        .windows(2)
        .filter(|pair| pair[0].to_frame() != pair[1].from_frame())
        .map(|pair| {
            format!(
                "{} ends in {} but {} starts in {}",
                repr_str(&pair[0].transform_id),
                repr_str(pair[0].to_frame()),
                repr_str(&pair[1].transform_id),
                repr_str(pair[1].from_frame())
            )
        })
        .collect()
}

/// B-spline basis weights at fractional position `t` in `[0, 1)`.
pub fn basis(order: i64, t: f64) -> Result<Vec<f64>> {
    match order {
        1 => Ok(vec![1.0 - t, t]),
        3 => {
            let t2 = t * t;
            let t3 = t * t * t;
            Ok(vec![
                (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
                (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
                (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
                t3 / 6.0,
            ])
        }
        _ => Err(Error::coded(
            "E502",
            format!(
                "B-spline order {order} is not supported; expected one of {}",
                crate::json::repr_int_list(&SUPPORTED_ORDERS)
            ),
        )),
    }
}

/// The `(start, stop)` voxels `roi` selects on each spatial axis of `grid`,
/// resolved as [`Transform::read_field`] resolves it: NumPy's negative and
/// out-of-range bounds, and whole axes after the last slice given.  A region
/// is a block of the grid's own lattice, so more slices than spatial axes, a
/// step other than 1 and an axis left without a voxel are refused, the error
/// naming `owner`.
fn roi_bounds(grid: &Grid, roi: &[Slice], owner: &str) -> Result<Vec<(usize, usize)>> {
    let spatial = grid.spatial_shape();
    if roi.len() > spatial.len() {
        return Err(Error::Value(format!(
            "{owner}: roi has {} axes; grid {} has {} spatial axes",
            roi.len(),
            repr_str(&grid.grid_id),
            spatial.len()
        )));
    }
    spatial
        .iter()
        .enumerate()
        .map(|(axis, n)| {
            let (a, b, step) = roi.get(axis).copied().unwrap_or_default().resolve(*n)?;
            if step != 1 {
                return Err(Error::Value(format!(
                    "{owner}: a region of interest is a block of the grid's own lattice, so a slice with step \
                     {step} is refused; take the region, then slice the result with the step"
                )));
            }
            if a == b {
                return Err(Error::Value(format!("{owner}: roi selects no voxel on axis {axis}, which has {n}")));
            }
            Ok((a, b))
        })
        .collect()
}

/// The grid of an ROI: same lattice, origin moved to the ROI's first voxel.
///
/// `roi` is resolved as [`Transform::read_field`] resolves it.  Its bounds
/// were taken as written: a negative stop made a grid of negative shape
/// (E109), a negative start put the origin off the grid, a short ROI lost
/// axes and a step was dropped; a step is now refused.
pub fn cropped_grid(grid: &Grid, roi: &[Slice]) -> Result<Grid> {
    let bounds = roi_bounds(grid, roi, &format!("grid {}", repr_str(&grid.grid_id)))?;
    let starts: Vec<f64> = bounds.iter().map(|(a, _)| *a as f64).collect();
    let origin = grid.index_to_world(&starts)?;
    let mut full_shape = grid.shape.clone();
    for (axis, (a, b)) in grid.spatial_axes().into_iter().zip(&bounds) {
        full_shape[axis] = (b - a) as i64;
    }
    let mut out = Grid::new(
        grid.grid_id.clone(),
        full_shape,
        grid.axis_names.clone(),
        grid.axis_kinds.clone(),
        grid.spacing.clone(),
        origin,
        grid.direction.clone(),
        grid.coord_system.clone(),
        grid.units.clone(),
    )?;
    out.timepoint = grid.timepoint.clone();
    out.frame_uid = grid.frame_uid.clone();
    Ok(out)
}

/// The first path by which composite `start` reaches itself, or empty.
///
/// Walked on the `components` attributes alone, so it opens nothing and
/// cannot recurse.
pub fn composite_cycle(start: &str, siblings: &Siblings) -> Result<Vec<String>> {
    cycle_from(start, |name| match siblings.get(name) {
        Some(node) if attrs::get_str(node, "kind")?.as_deref() == Some("composite") => {
            Ok(attrs::get_strs(node, "components")?.unwrap_or_default())
        }
        _ => Ok(Vec::new()),
    })
}

/// The first path by which `start` reaches itself through `components`, or
/// empty: a depth-first walk that remembers how it reached each node rather
/// than copying the path to it, which made a chain quadratic to walk.
fn cycle_from(start: &str, mut components: impl FnMut(&str) -> Result<Vec<String>>) -> Result<Vec<String>> {
    let mut stack = vec![start.to_string()];
    let mut reached_from: HashMap<String, String> = HashMap::new();
    while let Some(name) = stack.pop() {
        for child in components(&name)? {
            if child == start {
                let mut path = vec![child, name];
                while let Some(previous) = reached_from.get(&path[path.len() - 1]) {
                    path.push(previous.clone());
                }
                path.reverse();
                return Ok(path);
            }
            if !reached_from.contains_key(&child) {
                reached_from.insert(child.clone(), name.clone());
                stack.push(child);
            }
        }
    }
    Ok(Vec::new())
}

/// Composites that reach one another up to which each is given its own whole
/// path by [`composite_cycles`].
pub const CYCLE_PATHS: usize = 64;

/// How many names of a path [`composite_cycles`] gives a composite among more
/// than [`CYCLE_PATHS`] that reach one another.
pub const CYCLE_SHOWN: usize = 8;

/// Every composite that reaches itself through `components`, with a path by
/// which it does: `[name, …, name]`.
///
/// `graph` maps each composite to its components.  Whether a composite lies
/// on a cycle is decided for all of them in one walk (Tarjan's strongly
/// connected components): walking from each in turn took a file of a
/// thousand chained composites thirty seconds to validate.  Among at most
/// [`CYCLE_PATHS`] composites that reach one another, each has the path
/// [`composite_cycle`] reports for it.  Among more, a path may be shortened
/// to its start --- at most [`CYCLE_SHOWN`] names --- then `…` and the
/// composite again: every one of them written out was quadratic in the size
/// of the ring.
pub fn composite_cycles(graph: &IndexMap<String, Vec<String>>) -> Result<IndexMap<String, Vec<String>>> {
    let position: HashMap<&str, usize> = graph.keys().enumerate().map(|(i, k)| (k.as_str(), i)).collect();
    let next: Vec<Vec<usize>> = graph
        .values()
        .map(|children| children.iter().filter_map(|c| position.get(c.as_str()).copied()).collect())
        .collect();
    let n = next.len();
    let unseen = usize::MAX;
    let (mut order, mut low, mut scc) = (vec![unseen; n], vec![0; n], vec![0; n]);
    let (mut on_stack, mut stack, mut sizes) = (vec![false; n], Vec::new(), Vec::new());
    let mut counter = 0;
    for root in 0..n {
        if order[root] != unseen {
            continue;
        }
        let mut walk = vec![(root, 0)];
        order[root] = counter;
        low[root] = counter;
        counter += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&(v, i)) = walk.last() {
            if let Some(&w) = next[v].get(i) {
                walk.last_mut().expect("just read").1 += 1;
                if order[w] == unseen {
                    order[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    walk.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(order[w]);
                }
                continue;
            }
            walk.pop();
            if let Some(&(u, _)) = walk.last() {
                low[u] = low[u].min(low[v]);
            }
            if low[v] == order[v] {
                let mut size = 0;
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    scc[w] = sizes.len();
                    size += 1;
                    if w == v {
                        break;
                    }
                }
                sizes.push(size);
            }
        }
    }
    let names: Vec<&String> = graph.keys().collect();
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); sizes.len()];
    for v in 0..n {
        members[scc[v]].push(v);
    }
    let mut paths: Vec<Option<Vec<usize>>> = vec![None; n];
    let elided = usize::MAX;
    // Within a large component, each member's next step toward its first
    // member, found for all of them in one search backwards from it.
    let mut toward = vec![unseen; n];
    let mut before: Vec<Vec<usize>> = vec![Vec::new(); n];
    for v in 0..n {
        for &w in &next[v] {
            if scc[w] == scc[v] {
                before[w].push(v);
            }
        }
    }
    for group in &members {
        let first = group[0];
        if group.len() == 1 && !next[first].contains(&first) {
            continue;
        }
        if group.len() <= CYCLE_PATHS {
            for &v in group {
                paths[v] = Some(cycle_within(v, &next, &scc));
            }
            continue;
        }
        toward[first] = first;
        let mut queue = std::collections::VecDeque::from([first]);
        while let Some(w) = queue.pop_front() {
            for &v in &before[w] {
                if toward[v] == unseen {
                    toward[v] = w;
                    queue.push_back(v);
                }
            }
        }
        for &v in group {
            // Each member reaches the first, and the first reaches it back:
            // the path is shown as far as the first, or for the first itself
            // (or one containing itself directly) as it is.
            let (mut path, mut whole) = if v == first || next[v].contains(&v) {
                (cycle_within(v, &next, &scc), true)
            } else {
                let mut path = vec![v];
                while path.len() <= CYCLE_SHOWN && path[path.len() - 1] != first {
                    path.push(toward[path[path.len() - 1]]);
                }
                (path, false)
            };
            if path.len() > CYCLE_SHOWN {
                path.truncate(CYCLE_SHOWN);
                whole = false;
            }
            if !whole {
                path.extend([elided, v]);
            }
            paths[v] = Some(path);
        }
    }
    let mut out = IndexMap::new();
    for (v, path) in paths.into_iter().enumerate() {
        let Some(path) = path else { continue };
        let shown = path.iter().map(|i| if *i == elided { "…".to_string() } else { names[*i].clone() }).collect();
        out.insert(names[v].clone(), shown);
    }
    Ok(out)
}

/// The first path by which `start` reaches itself along `next` without
/// leaving its component, as [`cycle_from`] finds it: no walk that leaves a
/// strongly connected component returns to it.
fn cycle_within(start: usize, next: &[Vec<usize>], scc: &[usize]) -> Vec<usize> {
    let mut stack = vec![start];
    let mut reached_from: HashMap<usize, usize> = HashMap::new();
    while let Some(v) = stack.pop() {
        for &w in &next[v] {
            if scc[w] != scc[start] {
                continue;
            }
            if w == start {
                let mut path = vec![w, v];
                while let Some(&previous) = reached_from.get(&path[path.len() - 1]) {
                    path.push(previous);
                }
                path.reverse();
                return path;
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = reached_from.entry(w) {
                entry.insert(v);
                stack.push(w);
            }
        }
    }
    Vec::new()
}

/// The transform `inverse_id` names, if it really maps the other way.
///
/// One whose frames are not this transform's reversed is not its inverse
/// whatever the attribute says (E505).
pub fn stored_inverse(transform: &Transform) -> Result<Option<Transform>> {
    let Some(other) = transform.inverse()? else {
        return Ok(None);
    };
    if other.from_frame() != transform.to_frame() || other.to_frame() != transform.from_frame() {
        return Ok(None);
    }
    Ok(Some(other))
}

/// Whether an inverse can be *evaluated*, not merely declared.
pub fn can_invert(inner: &Transform) -> Result<bool> {
    if !inner.is_invertible() {
        return Ok(false);
    }
    if matches!(inner.body, Body::Stored) && (inner.kind() == "identity" || inner.kind() == "affine") {
        return Ok(true);
    }
    Ok(stored_inverse(inner)?.is_some())
}

/// Every transform group under `<sample root>/transforms`, by id.  A closed
/// file is refused, where HDF5 finding no member in it read as a sample
/// without transforms.
pub fn transform_groups(root: &hdf5::Group) -> Result<Siblings> {
    crate::h5::alive(root)?;
    let mut groups = Siblings::new();
    let Some(node) = ops::child_group(root, "transforms") else {
        return Ok(groups);
    };
    for name in ops::members(&node)? {
        if let Some(g) = ops::child_group(&node, &name) {
            groups.insert(name, g);
        }
    }
    Ok(groups)
}

/// Open each of `groups` as a transform: an `inverse_id` or a composite's
/// component resolves among them alone.
pub fn open_transforms(groups: Siblings, grids: Arc<IndexMap<String, Grid>>) -> Result<IndexMap<String, Transform>> {
    let siblings = Arc::new(groups);
    siblings
        .iter()
        .map(|(name, group)| Ok((name.clone(), Transform::open(name, group.clone(), grids.clone(), siblings.clone())?)))
        .collect()
}

/// Validate a transform id.
pub fn check_transform_id(transform_id: &str) -> Result<&str> {
    crate::ids::validate_id(transform_id, "transform id")
}

/// The interpolation and extrapolation values, for writers.
pub fn check_field_options(vector_space: &str, interpolation: &str, extrapolation: &str) -> Result<()> {
    if !VECTOR_SPACES.contains(&vector_space) {
        return Err(Error::coded(
            "E502",
            format!("vector_space {} must be one of {}", repr_str(vector_space), repr_list(&VECTOR_SPACES)),
        ));
    }
    if !INTERPOLATIONS.contains(&interpolation) {
        return Err(Error::coded(
            "E502",
            format!("interpolation {} must be one of {}", repr_str(interpolation), repr_list(&INTERPOLATIONS)),
        ));
    }
    if !EXTRAPOLATIONS.contains(&extrapolation) {
        return Err(Error::coded(
            "E502",
            format!("extrapolation {} must be one of {}", repr_str(extrapolation), repr_list(&EXTRAPOLATIONS)),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s10_3_a_matrix_is_s_plus_one_square_for_two_or_three_axes() {
        assert_eq!(affine_dimension(&[3, 3]).unwrap(), 2);
        assert_eq!(affine_dimension(&[4, 4]).unwrap(), 3);
        // Each of these was a panic or an underflow wherever it was used.
        for shape in [&[5, 2][..], &[4, 3], &[0, 0], &[1, 1], &[2, 2], &[5, 5], &[16], &[]] {
            assert_eq!(affine_dimension(shape).unwrap_err().code(), Some("E504"), "{shape:?}");
        }
    }

    fn graph(edges: &[(&str, &[&str])]) -> IndexMap<String, Vec<String>> {
        edges.iter().map(|(k, v)| (k.to_string(), v.iter().map(|c| c.to_string()).collect())).collect()
    }

    #[test]
    fn s10_5_every_composite_on_a_cycle_is_found_with_its_path() {
        // c1 -> c2 -> c3 -> c1 is a cycle; c0 reaches it without being on it,
        // and `loop` contains itself directly.
        let g = graph(&[
            ("c0", &["a", "c1"]),
            ("c1", &["a", "c2"]),
            ("c2", &["c3"]),
            ("c3", &["b", "c1"]),
            ("loop", &["loop"]),
            ("fine", &["a", "b"]),
        ]);
        let cycles = composite_cycles(&g).unwrap();
        assert_eq!(cycles.keys().collect::<Vec<_>>(), ["c1", "c2", "c3", "loop"]);
        assert_eq!(cycles["c1"], ["c1", "c2", "c3", "c1"]);
        assert_eq!(cycles["c3"], ["c3", "c1", "c2", "c3"]);
        assert_eq!(cycles["loop"], ["loop", "loop"]);
    }

    #[test]
    fn s10_5_a_long_chain_of_composites_is_walked_once() {
        // Walked from each composite in turn, copying the path at every step,
        // this was cubic: a thousand chained composites took thirty seconds
        // to validate.  A hundred thousand are one walk.
        let n = 100_000;
        let g: IndexMap<String, Vec<String>> =
            (0..n).map(|i| (format!("c{i}"), vec!["leg".to_string(), format!("c{}", i + 1)])).collect();
        let started = std::time::Instant::now();
        assert!(composite_cycles(&g).unwrap().is_empty());
        // Bounds for the walks this guards against, not for the machine: the
        // whole test has taken 5.6 s unoptimised on a shared Windows runner.
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "{:?}", started.elapsed());
        // Closed into a ring, every one of them is on the cycle, and up to
        // `CYCLE_PATHS` of them each has its whole path.
        let ring = |n: usize, chord: bool| -> IndexMap<String, Vec<String>> {
            (0..n)
                .map(|i| {
                    let mut components = vec![format!("c{}", (i + 1) % n)];
                    if chord && i == 1 {
                        components.push("c0".into());
                    }
                    (format!("c{i}"), components)
                })
                .collect()
        };
        let cycles = composite_cycles(&ring(50, false)).unwrap();
        assert_eq!(cycles.len(), 50);
        assert_eq!(cycles["c7"].len(), 51);
        assert_eq!((cycles["c7"][0].as_str(), cycles["c7"][1].as_str()), ("c7", "c8"));
        // Each of a hundred thousand written out was ten billion names.  Past
        // `CYCLE_PATHS`, a path is where it starts, and how it closes.
        let started = std::time::Instant::now();
        let cycles = composite_cycles(&ring(n, false)).unwrap();
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "{:?}", started.elapsed());
        assert_eq!(cycles.len(), n);
        assert!(cycles.values().all(|path| path.len() <= CYCLE_SHOWN + 2 && path[path.len() - 2] == "…"));
        assert_eq!(cycles["c7"][..3], ["c7", "c8", "c9"]);
        assert_eq!(cycles["c7"][CYCLE_SHOWN..], ["…", "c7"]);
        assert_eq!(cycles["c99999"], ["c99999", "c0", "…", "c99999"]);
        // A short cycle is shown whole wherever it is.
        let cycles = composite_cycles(&ring(1000, true)).unwrap();
        assert_eq!(cycles["c0"], ["c0", "c1", "c0"]);
    }
}
