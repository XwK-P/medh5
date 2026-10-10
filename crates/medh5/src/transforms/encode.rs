//! Transform encoders: arrays and parameters in, a [`Payload`] out (§10.3-§10.5).

use ndarray::ArrayD;

use super::model::{check_field_options, LAST_ROW_TOL, SUPPORTED_ORDERS, VECTOR_SPACES};
use crate::annotations::payload::{Payload, PayloadData};
use crate::array::{DType, NdArray};
use crate::geometry::linalg::{allclose, det};
use crate::h5::attrs::AttrValue;
use crate::json::{py_float, repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::{Error, Result};

/// The identity transform, which stores nothing but its endpoints.
pub fn encode_identity() -> Payload {
    Payload::new("identity")
}

/// Pack a homogeneous world-to-world `(S+1, S+1)` affine (§10.3).
///
/// Index-space affines are deliberately not storable: one that means something
/// only in one grid's index space silently breaks when the image is resampled.
pub fn encode_affine(matrix: &ArrayD<f64>) -> Result<Payload> {
    let shape = matrix.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(Error::coded(
            "E504",
            format!("affine `matrix` must be square (S+1, S+1), got {}", repr_int_tuple(shape)),
        ));
    }
    let dim = shape[0] - 1;
    let last: Vec<f64> = (0..=dim).map(|c| matrix[[dim, c]]).collect();
    let mut expected = vec![0.0; dim + 1];
    expected[dim] = 1.0;
    if !allclose(&last, &expected, LAST_ROW_TOL, 1e-5) {
        return Err(Error::coded(
            "E504",
            format!(
                "affine last row must be [0 \u{2026} 0 1], got [{}]",
                last.iter().map(|v| py_float(*v)).collect::<Vec<_>>().join(", ")
            ),
        ));
    }
    let linear = ndarray::Array2::from_shape_fn((dim, dim), |(r, c)| matrix[[r, c]]);
    if det(&linear).abs() < LAST_ROW_TOL {
        return Err(Error::coded(
            "E504",
            "affine linear part is singular, so the transform maps space onto a lower-dimensional set",
        ));
    }
    let mut payload = Payload::new("affine");
    payload.datasets.insert("matrix".into(), PayloadData::Array(NdArray::from(matrix.clone())));
    Ok(payload)
}

/// Pack a `(S, *spatial)` displacement field (§10.4), cast to `dtype`.
pub fn encode_displacement(
    field: &NdArray,
    field_grid: &str,
    vector_space: &str,
    interpolation: &str,
    extrapolation: &str,
    dtype: DType,
) -> Result<Payload> {
    check_field_options(vector_space, interpolation, extrapolation)?;
    let array = field.astype(dtype);
    let shape = array.shape();
    if shape.len() < 3 {
        return Err(Error::coded(
            "E503",
            format!("displacement field must be (S, *spatial), got {}", repr_int_tuple(&shape)),
        ));
    }
    if shape[0] != shape.len() - 1 {
        return Err(Error::coded(
            "E503",
            format!(
                "displacement field has {} components on a {}-D lattice; they must match",
                shape[0],
                shape.len() - 1
            ),
        ));
    }
    let mut payload = Payload::new("displacement");
    payload.datasets.insert("field".into(), PayloadData::Array(array));
    payload.attrs = vec![
        ("field_grid".into(), AttrValue::Str(field_grid.into())),
        ("vector_space".into(), AttrValue::Str(vector_space.into())),
        ("interpolation".into(), AttrValue::Str(interpolation.into())),
        ("extrapolation".into(), AttrValue::Str(extrapolation.into())),
    ];
    payload.stacked_axes = 1;
    Ok(payload)
}

/// Pack `(S, *cp_shape)` B-spline control-point coefficients (§10.5).
pub fn encode_bspline(control_points: &ArrayD<f64>, cp_grid: &str, order: i64, vector_space: &str) -> Result<Payload> {
    if !SUPPORTED_ORDERS.contains(&order) {
        return Err(Error::coded(
            "E502",
            format!("B-spline order {order} is not supported; expected one of {}", repr_int_list(&SUPPORTED_ORDERS)),
        ));
    }
    if !VECTOR_SPACES.contains(&vector_space) {
        return Err(Error::coded(
            "E502",
            format!("vector_space {} must be one of {}", repr_str(vector_space), repr_list(&VECTOR_SPACES)),
        ));
    }
    let shape = control_points.shape();
    if shape.len() < 3 || shape[0] != shape.len() - 1 {
        return Err(Error::coded(
            "E503",
            format!("control_points must be (S, *cp_shape) with S components, got {}", repr_int_tuple(shape)),
        ));
    }
    if shape[1..].iter().any(|extent| (*extent as i64) < order + 1) {
        return Err(Error::coded(
            "E503",
            format!(
                "an order-{order} B-spline needs at least {} control points per axis, got {}",
                order + 1,
                repr_int_tuple(&shape[1..])
            ),
        ));
    }
    let mut payload = Payload::new("bspline");
    payload.datasets.insert("control_points".into(), PayloadData::Array(NdArray::from(control_points.clone())));
    payload.attrs = vec![
        ("cp_grid".into(), AttrValue::Str(cp_grid.into())),
        ("order".into(), AttrValue::Int(order)),
        ("vector_space".into(), AttrValue::Str(vector_space.into())),
    ];
    payload.stacked_axes = 1;
    Ok(payload)
}

/// Declare an ordered composition by component id (§10.5).
pub fn encode_composite(components: &[String]) -> Result<Payload> {
    if components.len() < 2 {
        return Err(Error::coded("E501", "a composite transform needs at least two components"));
    }
    let mut payload = Payload::new("composite");
    payload.attrs = vec![("components".into(), AttrValue::strs(components))];
    Ok(payload)
}
