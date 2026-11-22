//! vl-core: vector types, distance kernels, and shared error definitions.
//!
//! This crate is the foundation of VectorLite. It provides the fundamental
//! numeric types and distance functions that every other crate depends on.
//! All public APIs are documented; errors use [`thiserror`] and never panic
//! on user input.

pub mod distance;
pub mod kernels;
pub mod vector;

// Flatten the most-used types to the crate root for convenience.
pub use distance::{CosineDistance, Distance, DistanceKind, InnerProductDistance, L2Distance};
pub use vector::{Vector, VectorId};

/// Top-level error type for the core layer.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CoreError {
    /// The supplied vector dimension did not match what was expected.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The required dimension.
        expected: usize,
        /// The dimension that was actually provided.
        got: usize,
    },
    /// A vector component was non-finite (NaN or infinite).
    #[error("vector contains non-finite value (NaN or infinity)")]
    NonFiniteValue,
    /// Cosine distance is undefined for zero-norm vectors.
    #[error("cannot compute cosine distance for zero-norm vector")]
    ZeroVector,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimension_mismatch_error_formats_correctly() {
        let err = CoreError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("128"),
            "error should mention expected dimension"
        );
        assert!(msg.contains("64"), "error should mention actual dimension");
    }
}
