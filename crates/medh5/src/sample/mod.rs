//! Samples: reading (`Sample`), images, and writing (`SampleWriter`).
//!
//! A sample is **one subject at one or more timepoints**.  Reading is lazy and
//! timepoint-aware; writing is a builder that validates as it goes and commits
//! atomically (§14.4).

pub mod image;
pub mod reader;

pub use image::{Image, ImageNode, SPEC_IMAGE_ATTRS, VALUE_TYPES};
pub use reader::{
    annotation_id, attr_name_map_of, frame_references, open_sample, read_document, read_document_text, require_major,
    Sample, PROFILES, ROOT_DIGEST_ATTRS,
};
pub mod writer;

pub use writer::{amend, create, Annotated, GridOptions, ImageOptions, QualityArg, SampleWriter};
pub mod writer_annotations;

pub use writer_annotations::{
    annotation_to_masks, transcode, AnnotationOptions, ObjectFields, Placement, SegmentationOptions,
    SegmentationSource, TransformSpec,
};
