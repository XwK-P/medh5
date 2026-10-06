//! Images: dense arrays defined on exactly one grid (spec §4).
//!
//! Reads go through [`Image::read`], which slices in a **single** HDF5 call:
//! `d[(k, *roi)]`, never `d[k][roi]` (§14.5).

use std::sync::Arc;

use ndarray::Array2;
use serde_json::{json, Value};

use crate::annotations::Grids;
use crate::array::{DType, Index, NdArray, Slice};
use crate::geometry::grid::Grid;
use crate::geometry::multiscale::Pyramid;
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, ops};
use crate::json::{repr_int_tuple, repr_list, repr_str};
use crate::{Error, Result};

/// `value_type` values (§4.1).
pub const VALUE_TYPES: [&str; 6] = ["intensity", "quantitative", "rgb", "probability", "displacement", "mask"];

/// The image attributes the spec defines.
pub const SPEC_IMAGE_ATTRS: [&str; 16] = [
    "grid",
    "modality",
    "value_type",
    "channel_names",
    "rescale_slope",
    "rescale_intercept",
    "value_units",
    "window_center",
    "window_width",
    "valid_mask",
    "digest",
    "prov",
    "levels",
    "downsample_factors",
    "downsample_method",
    "grid_levels",
];

/// The stored form of an image: one dataset, or a multiscale group.
#[derive(Debug, Clone)]
pub enum ImageNode {
    Single(hdf5::Dataset),
    Multiscale(hdf5::Group),
}

/// Lazy access to one image and its geometry.
#[derive(Debug, Clone)]
pub struct Image {
    pub image_id: String,
    pub node: ImageNode,
    grids: Arc<Grids>,
    /// The pyramid level this view reads, for a level of a multiscale image.
    level: Option<usize>,
}

/// Validate a `value_type` (E203).
pub fn check_value_type(value_type: &str) -> Result<()> {
    if !VALUE_TYPES.contains(&value_type) {
        return Err(Error::coded(
            "E203",
            format!("unknown value_type {}; expected one of {}", repr_str(value_type), repr_list(&VALUE_TYPES)),
        ));
    }
    Ok(())
}

impl Image {
    /// Open `images/<id>`, a dataset or a multiscale group.
    pub fn open(image_id: &str, parent: &hdf5::Group, grids: Arc<Grids>) -> Result<Image> {
        let node = match ops::node_kind(parent, image_id) {
            Some(ops::NodeKind::Group) => ImageNode::Multiscale(parent.group(image_id)?),
            Some(ops::NodeKind::Dataset) => ImageNode::Single(parent.dataset(image_id)?),
            _ => return Err(Error::Key(repr_str(image_id))),
        };
        Ok(Image { image_id: image_id.to_string(), node, grids, level: None })
    }

    fn location(&self) -> &hdf5::Location {
        match &self.node {
            ImageNode::Single(d) => d,
            ImageNode::Multiscale(g) => g,
        }
    }

    pub fn is_multiscale(&self) -> bool {
        matches!(self.node, ImageNode::Multiscale(_))
    }

    /// The number of pyramid levels (1 for a single-scale image).
    pub fn levels(&self) -> Result<usize> {
        match &self.node {
            ImageNode::Single(_) => Ok(1),
            ImageNode::Multiscale(g) => match attrs::get_i64(g, "levels")? {
                Some(n) => Ok(n.max(0) as usize),
                None => Ok(ops::members(g)?.len()),
            },
        }
    }

    /// The pyramid level this view reads (0 for the image itself).
    pub fn level_index(&self) -> usize {
        self.level.unwrap_or(0)
    }

    /// The dataset this view reads: level 0, or the view's level.
    pub fn dataset(&self) -> Result<hdf5::Dataset> {
        match &self.node {
            ImageNode::Single(d) => Ok(d.clone()),
            ImageNode::Multiscale(g) => Ok(g.dataset(&self.level.unwrap_or(0).to_string())?),
        }
    }

    /// A view of one pyramid level, sharing this image's attributes.
    pub fn level(&self, index: usize) -> Result<Image> {
        if !self.is_multiscale() {
            if index != 0 {
                return Err(Error::invalid(format!(
                    "image {} is single-scale; level {index} does not exist",
                    repr_str(&self.image_id)
                )));
            }
            return Ok(self.clone());
        }
        Ok(Image { level: Some(index), ..self.clone() })
    }

    /// The pyramid declaration of a multiscale image.
    pub fn pyramid(&self) -> Result<Option<Pyramid>> {
        let ImageNode::Multiscale(g) = &self.node else {
            return Ok(None);
        };
        let levels = attrs::require(g, "levels", "E105")?.as_i64().unwrap_or(0).max(0) as usize;
        let (rows, cols, values) = attrs::require(g, "downsample_factors", "E105")?
            .as_matrix()
            .ok_or_else(|| Error::coded("E105", "downsample_factors must be a 2-D array"))?;
        let factors = Array2::from_shape_vec((rows, cols), values)?;
        let method = attrs::get_str(g, "downsample_method")?.unwrap_or_default();
        let grid_levels = attrs::get_strs(g, "grid_levels")?.unwrap_or_default();
        Ok(Some(Pyramid { levels, downsample_factors: factors, downsample_method: method, grid_levels }))
    }

    /// An attribute of the image (the group's, for a multiscale image).
    pub fn attr(&self, name: &str) -> Result<Option<AttrValue>> {
        attrs::read(self.location(), name)
    }

    /// Every attribute name of the image.
    pub fn attr_names(&self) -> Result<Vec<String>> {
        attrs::names(self.location())
    }

    /// The id of the grid this view's data lies on.
    pub fn grid_id(&self) -> Result<String> {
        if let (Some(level), ImageNode::Multiscale(g)) = (self.level, &self.node) {
            let levels = attrs::get_strs(g, "grid_levels")?.unwrap_or_default();
            return levels.get(level).cloned().ok_or_else(|| Error::Index("tuple index out of range".into()));
        }
        let value = attrs::require(self.location(), "grid", "E205")?;
        Ok(value.as_str().unwrap_or_else(|| attrs::stringify_value(&value)))
    }

    /// The grid this view's data lies on.
    pub fn grid(&self) -> Result<&Grid> {
        let gid = self.grid_id()?;
        self.grids.get(&gid).ok_or_else(|| {
            Error::coded(
                "E101",
                format!("image {} names grid {}, which does not exist", repr_str(&self.image_id), repr_str(&gid)),
            )
        })
    }

    /// Inherited from the grid --- an image never declares its own (§3.7).
    pub fn timepoint(&self) -> Result<Option<String>> {
        Ok(self.grid()?.timepoint.clone())
    }

    fn required_text(&self, name: &str) -> Result<String> {
        let value = attrs::require(self.location(), name, "E205")?;
        Ok(value.as_str().unwrap_or_else(|| attrs::stringify_value(&value)))
    }

    pub fn modality(&self) -> Result<String> {
        self.required_text("modality")
    }

    pub fn value_type(&self) -> Result<String> {
        self.required_text("value_type")
    }

    pub fn value_units(&self) -> Result<Option<String>> {
        attrs::get_str(self.location(), "value_units")
    }

    pub fn channel_names(&self) -> Result<Option<Vec<String>>> {
        attrs::get_strs(self.location(), "channel_names")
    }

    /// `(slope, intercept)`, defaulting to the identity rescale.
    pub fn rescale(&self) -> Result<(f64, f64)> {
        Ok((
            attrs::get_f64(self.location(), "rescale_slope")?.unwrap_or(1.0),
            attrs::get_f64(self.location(), "rescale_intercept")?.unwrap_or(0.0),
        ))
    }

    pub fn is_rescaled(&self) -> Result<bool> {
        let (slope, intercept) = self.rescale()?;
        Ok(slope != 1.0 || intercept != 0.0)
    }

    /// `(centers, widths)` of the display window, when declared.
    pub fn window(&self) -> Result<Option<(Vec<f64>, Vec<f64>)>> {
        let loc = self.location();
        if !attrs::has(loc, "window_center") || !attrs::has(loc, "window_width") {
            return Ok(None);
        }
        Ok(Some((
            attrs::get_f64s(loc, "window_center")?.unwrap_or_default(),
            attrs::get_f64s(loc, "window_width")?.unwrap_or_default(),
        )))
    }

    pub fn valid_mask(&self) -> Result<Option<String>> {
        attrs::get_str(self.location(), "valid_mask")
    }

    pub fn prov(&self) -> Result<Option<String>> {
        attrs::get_str(self.location(), "prov")
    }

    /// The level-0 (or view-level) dataset's digest.
    pub fn digest(&self) -> Result<Option<String>> {
        let ds = self.dataset()?;
        attrs::get_str(&ds, "digest")
    }

    pub fn shape(&self) -> Result<Vec<usize>> {
        Ok(self.dataset()?.shape())
    }

    pub fn dtype(&self) -> Result<DType> {
        data::dtype(&self.dataset()?)
    }

    pub fn nbytes(&self) -> Result<usize> {
        data::nbytes(&self.dataset()?)
    }

    pub fn chunks(&self) -> Result<Option<Vec<usize>>> {
        Ok(data::chunks(&self.dataset()?))
    }

    /// The full selection for a region of interest: all axes, or only the
    /// spatial ones (the leading channel/time axes are then taken whole).
    pub fn index(&self, roi: Option<&[Slice]>) -> Result<Vec<Index>> {
        let grid = self.grid()?;
        let ndim = grid.ndim();
        let Some(roi) = roi else {
            return Ok(vec![Index::Slice(Slice::full()); ndim]);
        };
        if roi.len() == ndim {
            return Ok(roi.iter().map(|s| Index::Slice(*s)).collect());
        }
        if roi.len() == grid.n_spatial() {
            let mut out = vec![Index::Slice(Slice::full()); ndim - grid.n_spatial()];
            out.extend(roi.iter().map(|s| Index::Slice(*s)));
            return Ok(out);
        }
        Err(Error::invalid(format!(
            "roi has {} axes; image {} has {ndim} ({} spatial)",
            roi.len(),
            repr_str(&self.image_id),
            grid.n_spatial()
        )))
    }

    /// Read the image, or a region of it.
    ///
    /// `physical` applies `stored * slope + intercept` (in `dtype`, default
    /// `float32`).  The rescale is never applied silently.
    pub fn read(&self, roi: Option<&[Slice]>, physical: bool, dtype: Option<DType>) -> Result<NdArray> {
        let index = self.index(roi)?;
        self.read_index(&index, physical, dtype)
    }

    /// Read with an explicit selection (indices may drop axes).
    pub fn read_index(&self, index: &[Index], physical: bool, dtype: Option<DType>) -> Result<NdArray> {
        let block = data::read_region(&self.dataset()?, index)?;
        if physical && self.is_rescaled()? {
            let (slope, intercept) = self.rescale()?;
            let out = dtype.unwrap_or(DType::F32);
            return Ok(rescaled(&block, slope, intercept, out));
        }
        Ok(match dtype {
            Some(d) => block.astype(d),
            None => block,
        })
    }

    pub fn summary(&self) -> Result<Value> {
        let (slope, intercept) = self.rescale()?;
        let mut out = serde_json::Map::new();
        out.insert("id".into(), json!(self.image_id));
        out.insert("grid".into(), json!(self.grid_id()?));
        out.insert("timepoint".into(), json!(self.timepoint()?));
        out.insert("modality".into(), json!(self.modality()?));
        out.insert("value_type".into(), json!(self.value_type()?));
        out.insert("value_units".into(), json!(self.value_units()?));
        out.insert("shape".into(), json!(self.shape()?));
        out.insert("dtype".into(), json!(self.dtype()?.numpy_str()));
        out.insert("chunks".into(), json!(self.chunks()?));
        out.insert("levels".into(), json!(self.levels()?));
        out.insert("rescale".into(), json!([crate::json::num(slope), crate::json::num(intercept)]));
        out.insert("nbytes".into(), json!(self.nbytes()?));
        Ok(Value::Object(out))
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> Result<String> {
        let shape = repr_int_tuple(&self.shape()?);
        if let Some(level) = self.level {
            return Ok(format!("Image({} level {level}, shape={shape})", repr_str(&self.image_id)));
        }
        Ok(format!(
            "Image({}, {}, shape={shape}, dtype={}, grid={})",
            repr_str(&self.image_id),
            self.modality()?,
            self.dtype()?.numpy_str(),
            repr_str(&self.grid_id()?)
        ))
    }
}

/// `stored * slope + intercept`, computed in `out` as NumPy computes it: a
/// `float32` result rounds after the multiply and after the add.
pub fn rescaled(block: &NdArray, slope: f64, intercept: f64, out: DType) -> NdArray {
    match out {
        DType::F32 => {
            let (s, i) = (slope as f32, intercept as f32);
            NdArray::from(block.cast::<f32>().mapv(|v| v * s + i))
        }
        DType::F16 => {
            let (s, i) = (half::f16::from_f64(slope), half::f16::from_f64(intercept));
            NdArray::from(block.cast::<half::f16>().mapv(|v| v * s + i))
        }
        DType::F64 => NdArray::from(block.to_f64().mapv(|v| v * slope + intercept)),
        other => NdArray::from(block.to_f64().mapv(|v| v * slope + intercept)).astype(other),
    }
}

/// Whether a float array would survive `int16` storage unchanged (W907).
pub fn lossless_as_int16(array: &NdArray) -> bool {
    if !array.dtype().is_float() {
        return false;
    }
    let values = array.to_f64();
    let mut finite = values.iter().filter(|v| v.is_finite()).peekable();
    if finite.peek().is_none() {
        return false;
    }
    finite.all(|v| *v == v.round_ties_even() && *v >= i16::MIN as f64 && *v <= i16::MAX as f64)
}

/// Whether every value lies in `[0, 1]` (and there is at least one).
pub fn is_probability(array: &NdArray) -> bool {
    if array.dtype().is_float() && array.to_f64().iter().any(|v| v.is_nan()) {
        return false;
    }
    match array.min_max() {
        Some((lo, hi)) => lo >= 0.0 && hi <= 1.0,
        None => false,
    }
}

