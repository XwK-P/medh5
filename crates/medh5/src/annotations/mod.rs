//! Annotations: one coherent unit of ground truth per group (spec §6–§9).
//!
//! Encoders are pure --- masks, boxes or assertions in, a [`Payload`] out ---
//! so the transcoding matrix is testable without a file; the reader in
//! [`read`] opens a stored group as one [`Annotation`] whose methods dispatch
//! on its kind.

pub mod encode;
pub mod encode_geometric;
pub mod header;
pub mod payload;
pub mod read;
pub mod read_geometric;
pub mod select;

pub use encode::{encode_masks, transcode_payload, EncodeOptions, InstanceInput};
pub use encode_geometric::{Assertions, ObjectColumns, Polygon, SCOPES, SPACES};
pub use header::{
    AnnotationHeader, ANNOTATION_KINDS, GEOMETRIC_KINDS, RESERVED_KINDS, TASKS, VOXEL_KINDS,
};
pub use payload::{Masks, Payload, PayloadData};
pub use read::{Annotation, GridRef, Grids, Instance};
pub use read_geometric::Assertion;
pub use select::{analyse, select_encoding, OverlapStats};
