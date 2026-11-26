//! vl-index: vector index implementations.
//!
//! This crate contains brute-force exact search, HNSW approximate nearest-
//! neighbour search, and quantization (SQ8, PQ). All heavy work lives here;
//! the server and CLI depend on this crate through the `SearchIndex` trait.

pub mod brute_force;
pub mod concurrent_hnsw;
pub mod hnsw;

pub use brute_force::BruteForceIndex;
pub use concurrent_hnsw::ConcurrentHnswIndex;
pub use hnsw::{HnswConfig, HnswIndex, SearchScratch, SearchTrace};
use vl_core::{DistanceKind, VectorId};

/// Common interface for vector search indexes.
///
/// All index implementations (brute-force, HNSW, quantized) implement this trait
/// so the benchmark harness, server, and CLI can operate over them uniformly.
pub trait SearchIndex: Send + Sync {
    /// Insert a vector with the given identifier into the index.
    fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError>;

    /// Return the `k` nearest vectors to `query`, sorted by ascending distance.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError>;

    /// The number of vectors currently stored in the index.
    fn len(&self) -> usize;

    /// True if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The expected dimensionality of vectors in this index.
    fn dim(&self) -> usize;

    /// The distance metric used by this index.
    fn metric(&self) -> DistanceKind;
}

/// Error type for index operations.
#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    /// Wraps a core-layer error.
    #[error("core error: {0}")]
    Core(#[from] vl_core::CoreError),
    /// The index is empty; no search can be performed.
    #[error("index is empty")]
    Empty,
    /// The index has reached its maximum node capacity (`u32::MAX`).
    #[error("index capacity exceeded")]
    CapacityExceeded,
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
