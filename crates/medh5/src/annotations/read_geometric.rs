//! Readers for the geometric kinds (§8) and `classification` (§9).
//!
//! Geometric coordinates live in one space --- `index` (the annotation's own
//! grid) or `world` --- and are read on another grid only through a shared
//! frame of reference (§3.3).  Classification answers the questions callers
//! ask --- `value`, `state`, `positives` --- without collapsing several
//! assertions about one class into one silently.

use std::collections::BTreeMap;

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis, IxDyn};
use serde_json::{json, Map, Value};

use super::encode_geometric::{check_scope, check_slice_index, check_space, degenerate_axis, Polygon, CONTOUR_ROLES};
use super::read::{Annotation, GridRef};
use crate::geometry::affine::{box_corners, box_to_slices};
use crate::geometry::grid::Grid;
use crate::h5::attrs;
use crate::json::{format_g, repr_list, repr_str};
use crate::labels::{ClassKey, Skeleton};
use crate::{Error, Result};

/// `repr()` of an optional string, as Python prints `None`.
fn repr_opt(value: Option<&str>) -> String {
    value.map(repr_str).unwrap_or_else(|| "None".into())
}

/// Apply `f` to every point of a `(..., S)` array.
fn map_points(coords: &ArrayD<f64>, f: impl Fn(&[f64]) -> Result<Vec<f64>>) -> Result<ArrayD<f64>> {
    let shape = coords.shape().to_vec();
    let flat: Vec<f64> = coords.iter().copied().collect();
    let out = f(&flat)?;
    Ok(ArrayD::from_shape_vec(IxDyn(&shape), out)?)
}

/// `(S, 2)` bounds of a set of `(M, S)` corners.
fn bounds(corners: &Array2<f64>) -> Array2<f64> {
    let s = corners.ncols();
    let mut out = Array2::zeros((s, 2));
    for k in 0..s {
        let column = corners.column(k);
        out[[k, 0]] = column.iter().copied().fold(f64::INFINITY, f64::min);
        out[[k, 1]] = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    out
}

/// One label assertion: a class, its value, and what it is about (§9).
#[derive(Debug, Clone, PartialEq)]
pub struct Assertion {
    pub class_id: i64,
    pub value: f64,
    pub scope_id: Option<i64>,
    pub scheme: Option<String>,
    pub scheme_value: Option<String>,
}

impl Assertion {
    pub fn is_positive(&self) -> bool {
        self.value > 0.0
    }

    pub fn is_negative(&self) -> bool {
        self.value == 0.0
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> String {
        let scheme = match &self.scheme {
            Some(s) if !s.is_empty() => format!(", {s}={}", repr_opt(self.scheme_value.as_deref())),
            _ => String::new(),
        };
        let target = self.scope_id.map(|s| format!(" @{s}")).unwrap_or_default();
        format!("Assertion({}={}{target}{scheme})", self.class_id, format_g(self.value, 6))
    }
}

impl Annotation {
    // -- coordinate space -----------------------------------------------------

    /// `index` or `world` (§8.1); `index` when the file does not say.
    pub fn space(&self) -> Result<String> {
        let space = self.header.space.as_deref().unwrap_or("index");
        check_space(space)?;
        Ok(space.to_string())
    }

    /// The frame of reference: declared, or the grid's.
    pub fn frame_uid(&self) -> Option<String> {
        if let Some(f) = &self.header.frame_uid {
            return Some(f.clone());
        }
        self.header.grid.as_deref().and_then(|g| self.grids().get(g)).and_then(|g| g.frame_uid.clone())
    }

    /// Spatial dimensionality of the annotation's grid.
    pub fn n_spatial(&self) -> Result<usize> {
        Ok(self.grid()?.n_spatial())
    }

    /// The grid a call names.
    pub fn resolve_grid<'a>(&'a self, grid: GridRef<'a>) -> Result<&'a Grid> {
        match grid {
            GridRef::Grid(g) => Ok(g),
            GridRef::Id(id) => self.grids().get(id).ok_or_else(|| {
                Error::coded(
                    "E101",
                    format!("annotation {}: grid {} does not exist", repr_str(&self.ann_id), repr_str(id)),
                )
            }),
            GridRef::Own => self.grid(),
        }
    }

    /// Whether `grid` is the annotation's own.
    pub fn is_own(&self, grid: &Grid) -> bool {
        match self.header.grid.as_deref().and_then(|g| self.grids().get(g)) {
            Some(own) => own == grid,
            None => false,
        }
    }

    /// Refuse a grid these coordinates cannot be read on (E414).
    ///
    /// The annotation's own grid always relates; any other grid only through a
    /// shared frame of reference, and a grid without a `frame_uid` shares
    /// nothing, another frame-less grid included.
    pub fn require_related(&self, target: &Grid) -> Result<()> {
        if self.is_own(target) {
            return Ok(());
        }
        let frame = self.frame_uid();
        if frame.is_some() && frame == target.frame_uid {
            return Ok(());
        }
        Err(Error::coded(
            "E414",
            format!(
                "annotation {} is in frame {} and grid {} is in {}; a transform is required to relate them",
                repr_str(&self.ann_id),
                repr_opt(frame.as_deref()),
                repr_str(&target.grid_id),
                repr_opt(target.frame_uid.as_deref())
            ),
        ))
    }

    /// Map `(..., S)` coordinates from this annotation's space to world.
    ///
    /// Index coordinates count the voxels of the annotation's **own** grid,
    /// whichever grid is named: the grid says whose world the caller wants,
    /// which is the same world only when the frames match --- and then in that
    /// grid's units.
    pub fn to_world(&self, coords: &ArrayD<f64>, grid: GridRef) -> Result<ArrayD<f64>> {
        let target = match grid {
            GridRef::Own => None,
            other => Some(self.resolve_grid(other)?),
        };
        if let Some(target) = target {
            self.require_related(target)?;
        }
        let world = if self.space()? == "world" {
            coords.clone()
        } else {
            let own = self.grid()?;
            map_points(coords, |flat| own.index_to_world(flat))?
        };
        match (target, self.world_grid()) {
            (Some(target), Some(source)) => map_points(&world, |flat| source.world_into(target, flat)),
            _ => Ok(world),
        }
    }

    /// Map `(..., S)` coordinates from this annotation's space to `grid`'s
    /// continuous index, through world when the grids differ.
    ///
    /// World coordinates are carried into the target grid's units first: a
    /// point at 1 mm on a grid of 0.001 m voxels was index 1000 (N10 of the
    /// round-3 audit).  A target in another `coord_system` is refused (E414),
    /// where an LPS point used to be read as RAS unflipped.
    pub fn to_index(&self, coords: &ArrayD<f64>, grid: GridRef) -> Result<ArrayD<f64>> {
        let target = self.resolve_grid(grid)?;
        self.require_related(target)?;
        let mut values = coords.clone();
        if self.space()? == "index" {
            if self.is_own(target) {
                return Ok(values);
            }
            let own = self.grid()?;
            values = map_points(&values, |flat| own.index_to_world(flat))?;
        }
        if let Some(source) = self.world_grid() {
            values = map_points(&values, |flat| source.world_into(target, flat))?;
        }
        map_points(&values, |flat| target.world_to_index(flat))
    }

    /// The grid whose convention and units this annotation's world
    /// coordinates are in: its own (§3.5), or `None` for a world annotation
    /// that names no grid --- whose numbers are taken as they are.
    pub fn world_grid(&self) -> Option<&Grid> {
        self.header.grid.as_deref().and_then(|g| self.grids().get(g))
    }

    /// Per-object free-form JSON, decoded.
    pub fn attributes(&self) -> Result<Option<Vec<Value>>> {
        let Some(values) = self.read_strings_optional("attributes")? else {
            return Ok(None);
        };
        values
            .iter()
            .map(|v| Ok(crate::json::loads_lenient(if v.is_empty() { "{}" } else { v })?.0))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// The number of objects (§8): boxes, oriented boxes, keypoint sets,
    /// points, polygons or faces.
    pub fn n_items(&self) -> Result<usize> {
        let rows = |name: &str| -> Result<usize> { Ok(self.dataset_shape(name)?.first().copied().unwrap_or(0)) };
        match self.kind() {
            "obb" => rows("centers"),
            "keypoints" | "points" => rows("points"),
            "contours" => Ok(rows("contour_offsets")?.saturating_sub(1)),
            "mesh" => rows("faces"),
            "classification" => rows("class_ids"),
            "boxes" => rows("class_ids"),
            "instances" => self.n_objects(),
            _ => Err(Error::Type(format!(
                "annotation {} of kind {} has no length",
                repr_str(&self.ann_id),
                repr_str(self.kind())
            ))),
        }
    }

    // -- boxes (§8.2) ---------------------------------------------------------

    /// Spatial dimensionality, from the stored `(N, S, 2)` shape.
    pub fn box_ndim(&self) -> Result<usize> {
        Ok(self.dataset_shape("boxes")?.get(1).copied().unwrap_or(0))
    }

    /// The plane each 2-D box was drawn on, when stored.
    pub fn slice_index(&self) -> Result<Option<Vec<i32>>> {
        Ok(self.read_optional::<i32>("slice_index")?.map(|a| a.iter().copied().collect()))
    }

    fn boxes_f64(&self) -> Result<Vec<Vec<f64>>> {
        let boxes = self.boxes()?;
        let n = boxes.shape().first().copied().unwrap_or(0);
        Ok((0..n).map(|i| boxes.index_axis(Axis(0), i).iter().map(|v| f64::from(*v)).collect()).collect())
    }

    /// Each box as index slices on `grid`, clipped to it.
    ///
    /// A box carrying `slice_index` with a degenerate axis is §8.2's "2D box
    /// on slice k": the named slice is given one voxel of thickness, where a
    /// zero-thickness slice would select no voxels at all.
    pub fn as_slices(&self, grid: GridRef) -> Result<Vec<Vec<(i64, i64)>>> {
        let target = self.resolve_grid(grid)?;
        let boxes: Vec<Vec<f64>> = if self.space()? == "index" && self.is_own(target) {
            self.boxes_f64()?
        } else {
            self.boxes_in_index(target)?.outer_iter().map(|b| b.iter().copied().collect()).collect()
        };
        let planes = self.slice_index()?;
        if planes.is_some() && !self.is_own(target) {
            return Err(Error::coded(
                "E414",
                format!(
                    "annotation {}: `slice_index` names planes of grid {}, not of {}; read these boxes on their own grid",
                    repr_str(&self.ann_id),
                    repr_opt(self.grid_id()),
                    repr_str(&target.grid_id)
                ),
            ));
        }
        let shape = target.spatial_shape();
        if let Some(planes) = &planes {
            let planes64: Vec<i64> = planes.iter().map(|p| i64::from(*p)).collect();
            if let Some(problem) = check_slice_index(&planes64, boxes.len(), Some(&boxes), Some(&shape)) {
                return Err(Error::coded("E405", format!("annotation {}: {problem}", repr_str(&self.ann_id))));
            }
        }
        let mut out = Vec::with_capacity(boxes.len());
        for (i, bbox) in boxes.iter().enumerate() {
            let mut slices = box_to_slices(bbox, Some(&shape))?;
            if let Some(planes) = &planes {
                if let Some(axis) = degenerate_axis(bbox) {
                    let plane = i64::from(planes[i]);
                    let extent = shape[axis] as i64;
                    if !(0 <= plane && plane < extent) {
                        return Err(Error::coded(
                            "E405",
                            format!("slice_index names slice {plane} on an axis {extent} voxels deep"),
                        ));
                    }
                    slices[axis] = (plane, plane + 1);
                }
            }
            out.push(slices);
        }
        Ok(out)
    }

    /// `(N, S, 2)` boxes in `grid`'s index space (enclosing bounds).
    pub fn boxes_in_index(&self, grid: &Grid) -> Result<ndarray::Array3<f64>> {
        let s = self.box_ndim()?;
        let boxes = self.boxes_f64()?;
        let mut out = ndarray::Array3::zeros((boxes.len(), s, 2));
        for (i, bbox) in boxes.iter().enumerate() {
            let corners = corners_array(bbox)?;
            let index = self.to_index(&corners.into_dyn(), GridRef::Grid(grid))?;
            let b = bounds(&index.into_dimensionality::<ndarray::Ix2>()?);
            out.index_axis_mut(Axis(0), i).assign(&b);
        }
        Ok(out)
    }

    /// Exact `(N, 2**S, S)` world corners of every box.
    pub fn world_corners(&self, grid: GridRef) -> Result<ndarray::Array3<f64>> {
        let s = self.box_ndim()?;
        let boxes = self.boxes_f64()?;
        if s > crate::geometry::affine::MAX_CORNER_AXES {
            return Err(Error::Value(format!("a box with {s} axes has too many corners to enumerate")));
        }
        let mut out = ndarray::Array3::zeros((boxes.len(), 1 << s, s));
        for (i, bbox) in boxes.iter().enumerate() {
            let world = self.to_world(&corners_array(bbox)?.into_dyn(), grid)?;
            out.index_axis_mut(Axis(0), i).assign(&world.into_dimensionality::<ndarray::Ix2>()?);
        }
        Ok(out)
    }

    /// `(N, S, 2)` **enclosing** world bounds.
    pub fn as_world(&self, grid: GridRef) -> Result<ndarray::Array3<f64>> {
        if self.space()? == "world" {
            let boxes = self.boxes()?;
            let n = boxes.shape().first().copied().unwrap_or(0);
            let s = self.box_ndim()?;
            return Ok(boxes.mapv(f64::from).into_shape_with_order((n, s, 2))?);
        }
        let corners = self.world_corners(grid)?;
        let (n, _, s) = corners.dim();
        let mut out = ndarray::Array3::zeros((n, s, 2));
        for i in 0..n {
            out.index_axis_mut(Axis(0), i).assign(&bounds(&corners.index_axis(Axis(0), i).to_owned()));
        }
        Ok(out)
    }

    // -- obb (§8.3) -----------------------------------------------------------

    /// `(N, 2**S, S)` corners: `center + R @ (size/2 * s)` for every sign.
    pub fn obb_corners(&self) -> Result<ndarray::Array3<f64>> {
        let centers = self.read_as::<f32>("centers")?.mapv(f64::from);
        let sizes = self.read_as::<f32>("sizes")?.mapv(f64::from);
        let rotations = self.read_as::<f32>("rotations")?.mapv(f64::from);
        let n = centers.shape().first().copied().unwrap_or(0);
        let dim = centers.shape().get(1).copied().unwrap_or(0);
        let m = 1usize << dim;
        let mut out = ndarray::Array3::zeros((n, m, dim));
        for i in 0..n {
            for corner in 0..m {
                // Odometer order: the first axis varies slowest.
                let signed: Vec<f64> = (0..dim)
                    .map(|k| {
                        let sign = if (corner >> (dim - 1 - k)) & 1 == 1 { 1.0 } else { -1.0 };
                        sign * (sizes[[i, k]] / 2.0)
                    })
                    .collect();
                for r in 0..dim {
                    let mut offset = 0.0;
                    for (c, v) in signed.iter().enumerate() {
                        offset += v * rotations[[i, r, c]];
                    }
                    out[[i, corner, r]] = centers[[i, r]] + offset;
                }
            }
        }
        Ok(out)
    }

    /// `(N, S, 2)` axis-aligned bounds enclosing each oriented box.
    pub fn obb_as_aabb(&self) -> Result<ndarray::Array3<f64>> {
        let corners = self.obb_corners()?;
        let (n, _, s) = corners.dim();
        let mut out = ndarray::Array3::zeros((n, s, 2));
        for i in 0..n {
            out.index_axis_mut(Axis(0), i).assign(&bounds(&corners.index_axis(Axis(0), i).to_owned()));
        }
        Ok(out)
    }

    /// Volume of each oriented box.
    pub fn obb_volumes(&self) -> Result<Vec<f64>> {
        let sizes = self.read_as::<f32>("sizes")?.mapv(f64::from);
        Ok(sizes.outer_iter().map(|row| row.iter().product()).collect())
    }

    // -- keypoints (§8.4) -------------------------------------------------------

    /// `(N, K)` visibility; every keypoint visible when not stored.
    pub fn visibility(&self) -> Result<ArrayD<u8>> {
        match self.read_optional::<u8>("visibility")? {
            Some(v) => Ok(v),
            None => {
                let shape = self.dataset_shape("points")?;
                Ok(ArrayD::from_elem(IxDyn(&shape[..2.min(shape.len())]), 2u8))
            }
        }
    }

    /// The skeleton id this annotation names.
    pub fn skeleton_id(&self) -> Result<Option<String>> {
        attrs::get_str(&self.group, "skeleton")
    }

    /// The skeleton this annotation names, resolved from the label set.
    pub fn skeleton(&self) -> Result<Option<Skeleton>> {
        let Some(name) = self.skeleton_id()? else {
            return Ok(None);
        };
        match self.label_set() {
            None => Err(Error::coded(
                "E413",
                format!(
                    "annotation {} names skeleton {} but the sample has no label set to resolve it",
                    repr_str(&self.ann_id),
                    repr_str(&name)
                ),
            )),
            Some(ls) => Ok(Some(ls.skeleton(&name)?.clone())),
        }
    }

    /// `(N, K)` mask of keypoints that were actually annotated.
    pub fn labelled(&self) -> Result<ArrayD<bool>> {
        Ok(self.visibility()?.mapv(|v| v > 0))
    }

    // -- points (§8.5) ----------------------------------------------------------

    /// Point names, for landmark sets.
    pub fn point_names(&self) -> Result<Option<Vec<String>>> {
        self.read_strings_optional("names")
    }

    /// The correspondence group these points belong to.
    pub fn correspondence(&self) -> Result<Option<String>> {
        attrs::get_str(&self.group, "correspondence")
    }

    /// `name -> point`, for landmark sets.
    pub fn named_points(&self) -> Result<IndexMap<String, Vec<f32>>> {
        let Some(names) = self.point_names()? else {
            return Ok(IndexMap::new());
        };
        let points = self.read_as::<f32>("points")?;
        if names.len() != points.shape().first().copied().unwrap_or(0) {
            return Err(Error::Value(format!(
                "zip() argument 2 is shorter than argument 1 ({} names, {} points)",
                names.len(),
                points.shape().first().copied().unwrap_or(0)
            )));
        }
        Ok(names.into_iter().zip(points.outer_iter().map(|p| p.iter().copied().collect())).collect())
    }

    // -- contours (§8.6) --------------------------------------------------------

    /// The `n + 1` offsets into `vertices`.
    pub fn contour_offsets(&self) -> Result<Vec<i64>> {
        Ok(self.read_as::<i64>("contour_offsets")?.iter().copied().collect())
    }

    /// `(n, 2)` `(axis, index)` per polygon; `(-1, -1)` when not stored.
    pub fn contour_planes(&self) -> Result<Array2<i32>> {
        match self.read_optional::<i32>("contour_plane")? {
            Some(p) => {
                let n = p.shape().first().copied().unwrap_or(0);
                Ok(p.into_shape_with_order((n, 2))?)
            }
            None => Ok(Array2::from_elem((self.n_items()?, 2), -1)),
        }
    }

    /// `outer` or `hole` per polygon.
    pub fn contour_roles(&self) -> Result<Vec<String>> {
        match self.read_optional::<u8>("contour_role")? {
            None => Ok(vec!["outer".to_string(); self.n_items()?]),
            Some(roles) => roles
                .iter()
                .map(|v| {
                    CONTOUR_ROLES.get(*v as usize).map(|r| r.to_string()).ok_or_else(|| {
                        Error::Index(format!("contour role {v} is not one of {}", repr_list(&CONTOUR_ROLES)))
                    })
                })
                .collect(),
        }
    }

    /// One polygon's `(V, S)` vertices.
    pub fn polygon(&self, index: usize) -> Result<ArrayD<f32>> {
        let offsets = self.contour_offsets()?;
        if index + 1 >= offsets.len() {
            return Err(Error::Index(format!(
                "polygon {index} is out of range for {} polygons",
                offsets.len().saturating_sub(1)
            )));
        }
        let vertices = self.read_as::<f32>("vertices")?;
        let (a, b) = (offsets[index].max(0) as usize, offsets[index + 1].max(0) as usize);
        let n = vertices.shape()[0];
        Ok(vertices.slice_axis(Axis(0), ndarray::Slice::from(a.min(n)..b.min(n).max(a.min(n)))).to_owned())
    }

    /// Every polygon, decoded.
    pub fn polygons(&self) -> Result<Vec<Polygon>> {
        let classes = self.object_class_ids()?;
        let planes = self.contour_planes()?;
        let roles = self.contour_roles()?;
        (0..self.n_items()?)
            .map(|i| {
                Ok(Polygon {
                    vertices: self.polygon(i)?.mapv(f64::from),
                    class_id: i64::from(classes[i]),
                    plane: (i64::from(planes[[i, 0]]), i64::from(planes[[i, 1]])),
                    role: roles[i].clone(),
                })
            })
            .collect()
    }

    /// Plane -> polygon indices, which is how RTSTRUCT data is consumed.
    pub fn by_plane(&self) -> Result<IndexMap<(i32, i32), Vec<usize>>> {
        let mut out: IndexMap<(i32, i32), Vec<usize>> = IndexMap::new();
        for (i, plane) in self.contour_planes()?.outer_iter().enumerate() {
            out.entry((plane[0], plane[1])).or_default().push(i);
        }
        Ok(out)
    }

    // -- mesh (§8.7) ------------------------------------------------------------

    /// The number of sub-meshes.
    pub fn n_submeshes(&self) -> Result<usize> {
        Ok(match self.optional_dataset("mesh_offsets") {
            None => 1,
            Some(ds) => ds.shape().first().copied().unwrap_or(0).saturating_sub(1),
        })
    }

    /// `(3, 2)` axis-aligned bounds of the surface.
    pub fn mesh_bounds(&self) -> Result<Array2<f64>> {
        let v = self.read_as::<f32>("vertices")?.mapv(f64::from);
        let n = v.shape().first().copied().unwrap_or(0);
        let s = v.shape().get(1).copied().unwrap_or(0);
        Ok(bounds(&v.into_shape_with_order((n, s))?))
    }

    pub(super) fn geometric_summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.ann_id));
        out.insert("kind".into(), json!(self.kind()));
        out.insert("task".into(), json!(self.task()));
        out.insert("grid".into(), json!(self.grid_id()));
        out.insert("space".into(), json!(self.space()?));
        out.insert("frame_uid".into(), json!(self.frame_uid()));
        out.insert("timepoints".into(), json!(self.timepoints()));
        out.insert("objects".into(), json!(self.n_items()?));
        out.insert("classes".into(), json!(self.class_ids().len()));
        out.insert("annotated_classes".into(), json!(self.annotated_class_ids().len()));
        out.insert("fully_covered".into(), json!(self.is_fully_covered()));
        out.insert("quality".into(), json!(self.quality_key()));
        out.insert("prov".into(), json!(self.prov()));
        Ok(Value::Object(out))
    }

    // -- classification (§9) ----------------------------------------------------

    /// What the assertions are about (§9).
    pub fn scope(&self) -> Result<String> {
        let value = attrs::get_str(&self.group, "scope")?.ok_or_else(|| {
            Error::coded("E412", format!("annotation {}: `classification` requires `scope`", repr_str(&self.ann_id)))
        })?;
        check_scope(&value)?;
        Ok(value)
    }

    /// Whether several classes may be positive per scope unit.
    pub fn multilabel(&self) -> Result<bool> {
        Ok(attrs::get_bool(&self.group, "multilabel")?.unwrap_or(true))
    }

    /// The asserted class of each row.
    pub fn asserted_class_ids(&self) -> Result<Vec<u16>> {
        Ok(self.read_as::<u16>("class_ids")?.iter().copied().collect())
    }

    /// The asserted value of each row.
    pub fn assertion_values(&self) -> Result<Vec<f32>> {
        Ok(self.read_as::<f32>("values")?.iter().copied().collect())
    }

    /// The scope unit of each row, when stored.
    pub fn scope_ids(&self) -> Result<Option<Vec<i64>>> {
        Ok(self.read_optional::<i64>("scope_ids")?.map(|a| a.iter().copied().collect()))
    }

    /// The named ordinal scheme of each row, when stored.
    pub fn schemes(&self) -> Result<Option<Vec<String>>> {
        self.read_strings_optional("schemes")
    }

    /// The value under the named scheme of each row, when stored.
    pub fn scheme_values(&self) -> Result<Option<Vec<String>>> {
        self.read_strings_optional("scheme_values")
    }

    /// Every assertion, in stored order.
    pub fn assertions(&self) -> Result<Vec<Assertion>> {
        let classes = self.asserted_class_ids()?;
        let values = self.assertion_values()?;
        let units = self.scope_ids()?;
        let schemes = self.schemes()?;
        let scheme_values = self.scheme_values()?;
        Ok((0..classes.len())
            .map(|i| Assertion {
                class_id: i64::from(classes[i]),
                value: f64::from(values[i]),
                scope_id: units.as_ref().and_then(|u| u.get(i).copied()),
                scheme: schemes.as_ref().and_then(|s| s.get(i).cloned()),
                scheme_value: scheme_values.as_ref().and_then(|s| s.get(i).cloned()),
            })
            .collect())
    }

    fn duplicated_classes(&self) -> Result<BTreeMap<i64, usize>> {
        let mut counts: IndexMap<i64, usize> = IndexMap::new();
        for c in self.asserted_class_ids()? {
            *counts.entry(i64::from(c)).or_default() += 1;
        }
        Ok(counts.into_iter().filter(|(_, n)| *n > 1).collect())
    }

    fn require_unambiguous(&self, what: &str, target: Option<i64>) -> Result<()> {
        let mut repeated = self.duplicated_classes()?;
        if let Some(t) = target {
            repeated.retain(|c, _| *c == t);
        }
        if repeated.is_empty() {
            return Ok(());
        }
        let named = repeated
            .iter()
            .map(|(c, n)| format!("{} ({n} assertions)", repr_str(&self.class_key(*c))))
            .collect::<Vec<_>>()
            .join(", ");
        Err(Error::coded(
            "E412",
            format!(
                "annotation {}: {what} collapses one value per class, but {named} with scope {}. §9 makes that \
                 ordinary --- `scope_ids` is per assertion --- so there is no single answer to give. Pass \
                 scope_id=, or read `assertions()` / `by_scope_id()`.",
                repr_str(&self.ann_id),
                repr_str(&self.scope()?)
            ),
        ))
    }

    /// `key -> value` for every assertion, keyed by label-set key when known.
    ///
    /// Only well defined when each class is asserted once; a class asserted
    /// for several scope units is an error rather than a silent last-wins.
    pub fn labels(&self) -> Result<IndexMap<String, f64>> {
        self.require_unambiguous("labels", None)?;
        let classes = self.asserted_class_ids()?;
        let values = self.assertion_values()?;
        Ok(classes.iter().zip(&values).map(|(c, v)| (self.class_key(i64::from(*c)), f64::from(*v))).collect())
    }

    /// Keys of the positive classes.
    pub fn positives(&self) -> Result<Vec<String>> {
        Ok(self.labels()?.into_iter().filter(|(_, v)| *v > 0.0).map(|(k, _)| k).collect())
    }

    /// The asserted value, or `None` when the class was not asserted.
    pub fn value(&self, class: &ClassKey, scope_id: Option<i64>) -> Result<Option<f64>> {
        let target = self.resolve_class(class)?;
        if scope_id.is_none() {
            self.require_unambiguous("value()", Some(target))?;
        }
        let units = self.scope_ids()?;
        let classes = self.asserted_class_ids()?;
        let values = self.assertion_values()?;
        for (i, (c, v)) in classes.iter().zip(&values).enumerate() {
            if i64::from(*c) != target {
                continue;
            }
            if let Some(wanted) = scope_id {
                match &units {
                    Some(u) if u[i] == wanted => {}
                    _ => continue,
                }
            }
            return Ok(Some(f64::from(*v)));
        }
        Ok(None)
    }

    /// `positive`, `negative` or `unknown` for one class (§9).
    ///
    /// `unknown` whenever the class is outside `annotated_class_ids`: nobody
    /// looked, so its absence carries no information.
    pub fn state(&self, class: &ClassKey, scope_id: Option<i64>) -> Result<&'static str> {
        let target = self.resolve_class(class)?;
        if let Some(value) = self.value(&ClassKey::Id(target), scope_id)? {
            return Ok(if value > 0.0 { "positive" } else { "negative" });
        }
        Ok(if self.is_annotated(&ClassKey::Id(target))? { "negative" } else { "unknown" })
    }

    /// The ordinal value recorded under a named scheme, e.g. `"BI-RADS"`.
    pub fn scheme(&self, name: &str) -> Result<Option<String>> {
        let (Some(schemes), Some(values)) = (self.schemes()?, self.scheme_values()?) else {
            return Ok(None);
        };
        Ok(schemes.iter().zip(values).find(|(s, _)| s.as_str() == name).map(|(_, v)| v))
    }

    /// Assertions grouped by scope unit --- per-slice, per-visit, per-lesion.
    pub fn by_scope_id(&self) -> Result<IndexMap<Option<i64>, Vec<Assertion>>> {
        let mut out: IndexMap<Option<i64>, Vec<Assertion>> = IndexMap::new();
        for a in self.assertions()? {
            out.entry(a.scope_id).or_default().push(a);
        }
        Ok(out)
    }

    /// Whether this asserts something *about a set of timepoints* (§9).
    pub fn is_change_label(&self) -> Result<bool> {
        Ok(self.scope()? == "sample" && self.header.timepoints.as_ref().map(Vec::len).unwrap_or(0) > 1)
    }

    /// The timepoints a change label compares.
    pub fn compared_timepoints(&self) -> Result<Vec<String>> {
        Ok(attrs::get_strs(&self.group, "timepoints")?.unwrap_or_default())
    }

    fn summary_labels(&self) -> Result<Value> {
        if self.duplicated_classes()?.is_empty() {
            let labels = self.labels()?;
            return Ok(Value::Object(labels.into_iter().map(|(k, v)| (k, crate::json::num(v))).collect()));
        }
        let mut out: Map<String, Value> = Map::new();
        for a in self.assertions()? {
            let key = self.class_key(a.class_id);
            let unit = a.scope_id.map(|s| s.to_string()).unwrap_or_else(|| "-".into());
            let entry = out.entry(key).or_insert_with(|| Value::Object(Map::new()));
            if let Value::Object(m) = entry {
                m.insert(unit, crate::json::num(a.value));
            }
        }
        Ok(Value::Object(out))
    }

    pub(super) fn classification_summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("id".into(), json!(self.ann_id));
        out.insert("kind".into(), json!(self.kind()));
        out.insert("task".into(), json!(self.task()));
        out.insert("grid".into(), json!(self.grid_id()));
        out.insert("scope".into(), json!(self.scope()?));
        out.insert("multilabel".into(), json!(self.multilabel()?));
        out.insert("timepoints".into(), json!(self.timepoints()));
        out.insert("change_label".into(), json!(self.is_change_label()?));
        out.insert("labels".into(), self.summary_labels()?);
        out.insert("assertions".into(), json!(self.n_items()?));
        out.insert("classes".into(), json!(self.class_ids().len()));
        out.insert("annotated_classes".into(), json!(self.annotated_class_ids().len()));
        out.insert("fully_covered".into(), json!(self.is_fully_covered()));
        out.insert("quality".into(), json!(self.quality_key()));
        out.insert("prov".into(), json!(self.prov()));
        Ok(Value::Object(out))
    }
}

/// The `(2**S, S)` corners of a flat `(S, 2)` box, odometer order.
fn corners_array(bbox: &[f64]) -> Result<Array2<f64>> {
    let corners = box_corners(bbox)?;
    let s = bbox.len() / 2;
    let flat: Vec<f64> = corners.into_iter().flatten().collect();
    Ok(Array2::from_shape_vec((flat.len() / s.max(1), s), flat)?)
}
