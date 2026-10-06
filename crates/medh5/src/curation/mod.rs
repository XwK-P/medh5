//! Curation records: who produced what, how good it is, and who it is about.
//!
//! Spec §3.7 (timepoints), §11 (provenance, quality, de-identification) and §12
//! (identity, cohorts, splits).  These are the *documents* of the sample
//! document (§2.4); nothing here writes an HDF5 attribute.

pub mod identity;
pub mod provenance;
pub mod quality;
pub mod timeline;

pub use identity::{Cohort, Deidentification, Identity, SplitClaim};
pub use provenance::{Activity, Agent, Provenance};
pub use quality::{Agreement, Issue, QualityRecord};
pub use timeline::{Timeline, Timepoint};
