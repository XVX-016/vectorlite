//! Brute-force exact nearest-neighbour index.
//!
//! [`BruteForceIndex`] stores every inserted vector in memory and answers
//! top-k queries by scanning all vectors and maintaining a max-heap of size k.
//! It is `O(n · d)` per query, which is acceptable for small datasets and is
//! used as a ground-truth oracle in tests for approximate indexes.
//!
//! # Dependencies
//!
//! - `ordered-float = "4"`: In Rust, primitive `f32` does not implement `Ord` or `Eq`
//!   because IEEE-754 specifies that `NaN != NaN`. Standard library `BinaryHeap` requires
//!   elements to implement `Ord`. Using `NotNan<f32>` guarantees total ordering at
//!   compile-time and prevents panics during sorting.

use std::sync::Arc;

use ordered_float::NotNan;
use vl_core::{CoreError, Distance, VectorId};

use crate::IndexError;

/// An exact nearest-neighbour index backed by a linear scan.
///
/// Validates dimensions, finiteness, and metric constraints (such as non-zero norms for cosine)
/// upon `insert` and at `search` entry, and executes inner distance calculations via the unchecked path.
pub struct BruteForceIndex {
    dim: usize,
    distance: Arc<dyn Distance>,
    vectors: Vec<(VectorId, Vec<f32>)>,
}

impl BruteForceIndex {
    /// Create a new index for vectors of dimension `dim`.
    ///
    /// All vectors inserted into this index must have exactly `dim` components.
    pub fn new(dim: usize, distance: Arc<dyn Distance>) -> Self {
        Self {
            dim,
            distance,
            vectors: Vec::new(),
        }
    }

    /// The number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// True if no vectors have been inserted yet.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// The expected dimensionality of this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Insert a vector into the index.
    ///
    /// # Errors
    ///
    /// * [`IndexError::Core(CoreError::DimensionMismatch)`] if `v.len() != self.dim`.
    /// * [`IndexError::Core(CoreError::NonFiniteValue)`] if any component is NaN or infinite.
    /// * [`IndexError::Core(CoreError::ZeroVector)`] if distance metric is Cosine and vector is zero.
    pub fn insert(&mut self, id: VectorId, v: Vec<f32>) -> Result<(), IndexError> {
        if v.len() != self.dim {
            return Err(IndexError::Core(CoreError::DimensionMismatch {
                expected: self.dim,
                got: v.len(),
            }));
        }
        self.distance
            .validate_vector(&v)
            .map_err(IndexError::Core)?;
        self.vectors.push((id, v));
        Ok(())
    }

    /// Return the `k` nearest vectors to `query`, sorted by ascending distance.
    ///
    /// Validates `query` at entry, then uses unchecked kernel calculations during
    /// the linear scan.
    ///
    /// # Returns
    ///
    /// A `Vec<(VectorId, f32)>` of length `min(k, self.len())`, sorted from
    /// closest to farthest. If vectors have identical distance, ties are
    /// broken deterministically by ascending [`VectorId`].
    ///
    /// # Errors
    ///
    /// * [`IndexError::Empty`] – if no vectors have been inserted.
    /// * [`IndexError::Core(CoreError::DimensionMismatch)`] – if `query.len() != self.dim`.
    /// * [`IndexError::Core(CoreError::NonFiniteValue)`] – if query contains NaN or infinity.
    /// * [`IndexError::Core(CoreError::ZeroVector)`] – if cosine distance is used with a zero query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        if self.vectors.is_empty() {
            return Err(IndexError::Empty);
        }
        if query.len() != self.dim {
            return Err(IndexError::Core(CoreError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            }));
        }
        self.distance
            .validate_vector(query)
            .map_err(IndexError::Core)?;

        if k == 0 {
            return Ok(Vec::new());
        }

        let mut heap: std::collections::BinaryHeap<(NotNan<f32>, VectorId)> =
            std::collections::BinaryHeap::with_capacity(k + 1);

        for (id, v) in &self.vectors {
            // Unchecked call: dimensions, finiteness, and non-zero invariants are guaranteed
            let dist = self.distance.compute_unchecked(query, v);
            let not_nan =
                NotNan::new(dist).map_err(|_| IndexError::Core(CoreError::NonFiniteValue))?;
            let candidate = (not_nan, *id);

            if heap.len() < k {
                heap.push(candidate);
            } else if let Some(&worst) = heap.peek() {
                if candidate < worst {
                    heap.pop();
                    heap.push(candidate);
                }
            }
        }

        let mut results: Vec<(VectorId, f32)> = heap
            .into_iter()
            .map(|(dist, id)| (id, dist.into_inner()))
            .collect();

        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use vl_core::{CosineDistance, L2Distance, VectorId};

    fn build_index(n: usize, dim: usize, seed: u64) -> BruteForceIndex {
        let mut idx = BruteForceIndex::new(dim, Arc::new(L2Distance));
        let mut state = seed;
        for i in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    state = state
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (state >> 40) as f32 / 1e13
                })
                .collect();
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    fn exact_top_k(idx: &BruteForceIndex, query: &[f32], k: usize) -> Vec<VectorId> {
        let mut all: Vec<(VectorId, f32)> = idx
            .vectors
            .iter()
            .map(|(id, v)| (*id, idx.distance.compute(query, v).unwrap()))
            .collect();
        all.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        all.into_iter().take(k).map(|(id, _)| id).collect()
    }

    #[test]
    fn insert_and_len() {
        let mut idx = BruteForceIndex::new(3, Arc::new(L2Distance));
        assert!(idx.is_empty());
        idx.insert(VectorId(1), vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn insert_dimension_mismatch() {
        let mut idx = BruteForceIndex::new(3, Arc::new(L2Distance));
        let err = idx.insert(VectorId(0), vec![1.0, 2.0]).unwrap_err();
        assert!(matches!(
            err,
            IndexError::Core(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn insert_rejects_nan_and_inf() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        let err1 = idx.insert(VectorId(0), vec![f32::NAN, 1.0]).unwrap_err();
        assert!(matches!(err1, IndexError::Core(CoreError::NonFiniteValue)));

        let err2 = idx
            .insert(VectorId(1), vec![f32::INFINITY, 1.0])
            .unwrap_err();
        assert!(matches!(err2, IndexError::Core(CoreError::NonFiniteValue)));

        let err3 = idx
            .insert(VectorId(2), vec![f32::NEG_INFINITY, 1.0])
            .unwrap_err();
        assert!(matches!(err3, IndexError::Core(CoreError::NonFiniteValue)));
    }

    #[test]
    fn cosine_rejects_zero_vector_at_insert_time() {
        let mut idx = BruteForceIndex::new(3, Arc::new(CosineDistance));
        let err = idx.insert(VectorId(0), vec![0.0, 0.0, 0.0]).unwrap_err();
        assert!(matches!(err, IndexError::Core(CoreError::ZeroVector)));

        // Non-zero insert succeeds
        assert!(idx.insert(VectorId(1), vec![0.0, 1.0, 0.0]).is_ok());
    }

    #[test]
    fn search_empty_returns_error() {
        let idx = BruteForceIndex::new(3, Arc::new(L2Distance));
        assert!(matches!(
            idx.search(&[0.0, 0.0, 0.0], 1),
            Err(IndexError::Empty)
        ));
    }

    #[test]
    fn search_dimension_mismatch() {
        let mut idx = BruteForceIndex::new(3, Arc::new(L2Distance));
        idx.insert(VectorId(0), vec![1.0, 2.0, 3.0]).unwrap();
        let err = idx.search(&[1.0, 2.0], 1).unwrap_err();
        assert!(matches!(
            err,
            IndexError::Core(CoreError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn search_rejects_nan_and_inf_query() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx.insert(VectorId(0), vec![1.0, 2.0]).unwrap();

        let err_nan = idx.search(&[f32::NAN, 1.0], 1).unwrap_err();
        assert!(matches!(
            err_nan,
            IndexError::Core(CoreError::NonFiniteValue)
        ));

        let err_inf = idx.search(&[1.0, f32::INFINITY], 1).unwrap_err();
        assert!(matches!(
            err_inf,
            IndexError::Core(CoreError::NonFiniteValue)
        ));
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx.insert(VectorId(0), vec![1.0, 1.0]).unwrap();
        let res = idx.search(&[1.0, 1.0], 0).unwrap();
        assert!(res.is_empty());
    }

    #[test]
    fn search_k_larger_than_n_returns_all() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx.insert(VectorId(0), vec![0.0, 0.0]).unwrap();
        idx.insert(VectorId(1), vec![1.0, 1.0]).unwrap();
        let res = idx.search(&[0.0, 0.0], 10).unwrap();
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn duplicate_vectors_deterministic_tie_breaking() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx.insert(VectorId(42), vec![1.0, 1.0]).unwrap();
        idx.insert(VectorId(5), vec![1.0, 1.0]).unwrap();
        idx.insert(VectorId(17), vec![1.0, 1.0]).unwrap();

        let res = idx.search(&[1.0, 1.0], 3).unwrap();
        assert_eq!(res.len(), 3);
        assert_eq!(res[0].0, VectorId(5));
        assert_eq!(res[1].0, VectorId(17));
        assert_eq!(res[2].0, VectorId(42));
    }

    #[test]
    fn cosine_search_rejects_zero_query() {
        let mut idx = BruteForceIndex::new(2, Arc::new(CosineDistance));
        idx.insert(VectorId(1), vec![1.0, 1.0]).unwrap();
        let err = idx.search(&[0.0, 0.0], 1).unwrap_err();
        assert!(matches!(err, IndexError::Core(CoreError::ZeroVector)));
    }

    #[test]
    fn nearest_neighbour_is_correct() {
        let mut idx = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx.insert(VectorId(0), vec![10.0, 10.0]).unwrap();
        idx.insert(VectorId(1), vec![0.1, 0.1]).unwrap();
        idx.insert(VectorId(2), vec![5.0, 5.0]).unwrap();

        let res = idx.search(&[0.0, 0.0], 1).unwrap();
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].0, VectorId(1));
    }

    #[test]
    fn results_sorted_ascending_distance() {
        let idx = build_index(50, 8, 42);
        let query = vec![0.5f32; 8];
        let res = idx.search(&query, 10).unwrap();
        for w in res.windows(2) {
            assert!(w[0].1 <= w[1].1, "results not sorted: {:?}", res);
        }
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 128,
            ..Default::default()
        })]

        #[test]
        fn search_exact_top_k(
            n in 1usize..=100,
            dim in 1usize..=32,
            k in 0usize..=20,
            seed in 0u64..u64::MAX,
        ) {
            let idx = build_index(n, dim, seed);
            let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
            let k_clamped = k.min(n);

            let got_ids: Vec<VectorId> = idx.search(&query, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let expected_ids = exact_top_k(&idx, &query, k_clamped);

            prop_assert_eq!(
                got_ids,
                expected_ids,
                "top-{} mismatch: n={} dim={}",
                k_clamped,
                n,
                dim
            );
        }
    }
}
