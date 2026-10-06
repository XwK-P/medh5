//! Content addressing and verification (spec §13).

pub mod digest;
pub mod repair;
pub mod verify;

pub use digest::{
    array_digest, attrs_digest, canonical_attrs, collect_digests, compute_content_id, dataset_digest, group_digest,
    relative_path, stamp_digests, AttrNameMap,
};
pub use verify::{stale_index_entries, subtrees_identical, verify_object, verify_root, VerifyResult};
