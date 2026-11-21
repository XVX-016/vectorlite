//! vl-index: vector index implementations.
//!
//! This crate contains brute-force exact search, HNSW approximate nearest-
//! neighbour search, and quantization (SQ8, PQ). All heavy work lives here;
//! the server and CLI depend on this crate through the `SearchIndex` trait.

/// Placeholder error type for index operations (populated in Phase 1).
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// Wraps a core-layer error.
    #[error("core error: {0}")]
    Core(#[from] vl_core::CoreError),
    /// The index is empty; no search can be performed.
    #[error("index is empty")]
    Empty,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_error_formats_correctly() {
        let err = IndexError::Empty;
        assert_eq!(err.to_string(), "index is empty");
    }
}
