//! The `clinical` profile (format 1.1): clinical context beside the images.
//!
//! A 1.1 sample may carry, under `clinical/`, the subject's source documents,
//! asynchronous observations and interventions, typed links between them and
//! the imaging objects, and explicit information-availability times
//! (`docs/spec/medh5-1.1.md` §3--§8).  The profile is additive: no image,
//! grid, annotation, transform or `/meta` member changes meaning, a sample
//! still needs its images, and a clinical event needs no imaging timepoint.
//!
//! - [`model`] --- the logical records: descriptor, events, documents, links;
//! - [`columns`] --- the §4 column encoding: primitive columns, packed UTF-8,
//!   validity masks;
//! - [`table`] --- the three tables written and read, and the [`Clinical`]
//!   reader a sample exposes;
//! - [`check`] --- the record rules the writer and the validator share;
//! - [`schema`] --- the descriptor and logical-record JSON Schema;
//! - [`select`] --- revision chains and prospective selection at a cutoff
//!   (§9, the task-and-cache contract §4);
//! - [`augment`] --- adding the profile to an existing sample (§10).

pub mod augment;
pub mod check;
pub mod columns;
pub mod model;
pub mod schema;
pub mod select;
pub mod table;

pub use check::{check_document, check_event, check_link, check_records, Finding, SampleContext};
pub use model::{
    Bounds, ClinicalRecords, Clock, Descriptor, Document, Event, Link, ASSESSMENT_SYSTEM, CLOCK_REFERENCES,
    COMPARATORS, DAY, ENDPOINT_TYPES, EVENT_KINDS, GROUP, HOUR, LESION_PRESENCE, LESION_VALUES, MEDIA_TYPES,
    MIN_VERSION, PROFILE, RELATIONS, SCHEMA, SECOND, STATUSES, TEMPORAL_TYPES,
};
pub use select::{select, Chains, Selected, Selection, SelectionPolicy, POLICIES};
pub use table::{recognised, Clinical, DocumentInfo};
