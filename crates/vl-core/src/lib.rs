//! vl-core: vector types, distance kernels, and shared error definitions.
//!
//! This crate is the foundation of VectorLite. It provides the fundamental
//! numeric types and distance functions that every other crate depends on.
//! All public APIs are documented; errors use [`thiserror`] and never panic
//! on user input.

/// A placeholder for the top-level error type (populated in Phase 1).
#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    /// The supplied vector dimension did not match what was expected.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch {
        /// The required dimension.
        expected: usize,
        /// The dimension that was actually provided.
        got: usize,
    },
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
