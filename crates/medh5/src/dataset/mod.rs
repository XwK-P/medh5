//! Cohort tools: manifests, splits, streaming statistics, cross-file checks.
//!
//! Everything above the single file.  A sample is self-describing, but a
//! *cohort* has properties no file can carry: which label set everyone agrees
//! on, which subject is in which partition, what the intensity distribution
//! is, whether a class was examined everywhere or only in some files.  These
//! are computed from metadata alone wherever metadata can answer.

pub mod check;
pub mod manifest;
pub mod split;
pub mod stats;

pub use check::{check, CohortReport, Finding, CHECK_CODES};
pub use manifest::{entries_for, find, scan, Entry, Manifest, GROUPABLE};
pub use split::{make_splits, write_claims, Assignment, Split, SplitOptions};
pub use stats::{compute_stats, ClassStats, DatasetStats, Moments, StatsOptions};
