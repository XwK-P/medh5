//! Conversion infrastructure shared by every importer, and the 0.x migration.
//!
//! The library-specific converters (NIfTI, DICOM, DICOM-SEG, RTSTRUCT,
//! nnU-Net) live in the Python package, because they are built on Python
//! libraries; they record their decisions in the same [`ConversionReport`]
//! and group studies with the same rules as the native migration.

pub mod grouping;
pub mod legacy;
pub mod report;

pub use grouping::{group_by_subject, Occasion, SubjectGroup};
pub use report::{merge_reports, ConversionReport, Note};
