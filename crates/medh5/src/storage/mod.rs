//! Storage-layer concerns: chunking, codecs and derived sampling indices (spec §14).

pub mod chunking;
pub mod codecs;
pub mod index;
pub mod recompress;

pub use codecs::{CodecProfile, Role};
