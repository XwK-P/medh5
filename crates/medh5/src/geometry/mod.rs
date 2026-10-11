//! Geometry: grids, the index-world affine, and multiscale pyramids (spec §3, §4.3).

pub mod affine;
pub mod grid;
pub mod linalg;
pub mod multiscale;

pub use grid::{Grid, GridSpec};
