//! Distance trait and concrete distance type implementations.

use crate::{kernels, CoreError};

/// Which distance metric to use when comparing vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceKind {
    /// Squared Euclidean distance: `Σ (a[i] - b[i])²`.
    ///
    /// Smaller is closer.
    L2,
    /// Inner (dot) product: `Σ a[i] * b[i]`.
    ///
    /// Larger means more similar.
    InnerProduct,
    /// Cosine distance: `1 - dot(a,b) / (‖a‖ · ‖b‖)`.
    ///
    /// Ranges from 0 (identical direction) to 2 (opposite direction).
    /// Smaller is closer.
    Cosine,
}

/// A distance metric that can compare two f32 vectors.
///
/// Implementors are `Send + Sync` so they can be shared across threads via
/// `Arc<dyn Distance>`.
pub trait Distance: Send + Sync {
    /// Public checked API: validates dimensions and finiteness, then computes distance.
    ///
    /// # Errors
    ///
    /// * [`CoreError::DimensionMismatch`] if `a.len() != b.len()`.
    /// * [`CoreError::NonFiniteValue`] if any element is NaN or infinite.
    /// * [`CoreError::ZeroVector`] if computing cosine distance with a zero vector.
    fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32, CoreError>;

    /// Unchecked kernel call for index internals.
    ///
    /// Caller guarantees:
    /// - `a.len() == b.len()`
    /// - All elements in `a` and `b` are finite
    /// - Stored vectors for cosine metrics are non-zero
    fn compute_unchecked(&self, a: &[f32], b: &[f32]) -> f32;

    /// Validate a vector intended for storage in an index using this metric.
    ///
    /// # Errors
    ///
    /// * [`CoreError::NonFiniteValue`] if any component is NaN or infinite.
    /// * [`CoreError::ZeroVector`] if this is a cosine metric and vector has zero norm.
    fn validate_vector(&self, v: &[f32]) -> Result<(), CoreError>;

    /// The kind of distance this implementation computes.
    fn kind(&self) -> DistanceKind;
}

// ---------------------------------------------------------------------------
// Concrete implementations
// ---------------------------------------------------------------------------

/// Squared Euclidean distance, with SIMD dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub struct L2Distance;

impl Distance for L2Distance {
    fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32, CoreError> {
        check_dims_and_finite(a, b)?;
        Ok(kernels::compute_unchecked(DistanceKind::L2, a, b))
    }

    #[inline]
    fn compute_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        kernels::compute_unchecked(DistanceKind::L2, a, b)
    }

    fn validate_vector(&self, v: &[f32]) -> Result<(), CoreError> {
        for &x in v {
            if !x.is_finite() {
                return Err(CoreError::NonFiniteValue);
            }
        }
        Ok(())
    }

    fn kind(&self) -> DistanceKind {
        DistanceKind::L2
    }
}

/// Inner (dot) product, with SIMD dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub struct InnerProductDistance;

impl Distance for InnerProductDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32, CoreError> {
        check_dims_and_finite(a, b)?;
        Ok(kernels::compute_unchecked(DistanceKind::InnerProduct, a, b))
    }

    #[inline]
    fn compute_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        kernels::compute_unchecked(DistanceKind::InnerProduct, a, b)
    }

    fn validate_vector(&self, v: &[f32]) -> Result<(), CoreError> {
        for &x in v {
            if !x.is_finite() {
                return Err(CoreError::NonFiniteValue);
            }
        }
        Ok(())
    }

    fn kind(&self) -> DistanceKind {
        DistanceKind::InnerProduct
    }
}

/// Cosine distance, with SIMD dispatch.
#[derive(Debug, Clone, Copy, Default)]
pub struct CosineDistance;

impl Distance for CosineDistance {
    fn compute(&self, a: &[f32], b: &[f32]) -> Result<f32, CoreError> {
        check_dims_and_finite(a, b)?;
        let dist = kernels::compute_unchecked(DistanceKind::Cosine, a, b);
        if dist.is_nan() {
            Err(CoreError::ZeroVector)
        } else {
            Ok(dist)
        }
    }

    #[inline]
    fn compute_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        kernels::compute_unchecked(DistanceKind::Cosine, a, b)
    }

    fn validate_vector(&self, v: &[f32]) -> Result<(), CoreError> {
        let mut norm_sq = 0.0f32;
        for &x in v {
            if !x.is_finite() {
                return Err(CoreError::NonFiniteValue);
            }
            norm_sq += x * x;
        }
        if norm_sq == 0.0 {
            return Err(CoreError::ZeroVector);
        }
        Ok(())
    }

    fn kind(&self) -> DistanceKind {
        DistanceKind::Cosine
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

#[inline]
fn check_dims_and_finite(a: &[f32], b: &[f32]) -> Result<(), CoreError> {
    if a.len() != b.len() {
        return Err(CoreError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    for &x in a {
        if !x.is_finite() {
            return Err(CoreError::NonFiniteValue);
        }
    }
    for &y in b {
        if !y.is_finite() {
            return Err(CoreError::NonFiniteValue);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn l2_dimension_mismatch() {
        let d = L2Distance;
        let err = d.compute(&[1.0, 2.0], &[1.0]).unwrap_err();
        assert!(matches!(err, CoreError::DimensionMismatch { .. }));
    }

    #[test]
    fn cosine_dimension_mismatch() {
        let d = CosineDistance;
        assert!(d.compute(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn ip_dimension_mismatch() {
        let d = InnerProductDistance;
        assert!(d.compute(&[], &[1.0]).is_err());
    }

    #[test]
    fn l2_zero_distance_for_identical() {
        let d = L2Distance;
        let v = vec![3.0f32, 1.0, 4.0, 1.0, 5.0];
        assert!(d.compute(&v, &v).unwrap() < 1e-5);
    }

    #[test]
    fn cosine_zero_distance_for_identical() {
        let d = CosineDistance;
        let v = vec![1.0f32, 2.0, 3.0];
        assert!(d.compute(&v, &v).unwrap() < 1e-5);
    }

    #[test]
    fn ip_known_value() {
        let d = InnerProductDistance;
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        assert!((d.compute(&a, &b).unwrap() - 32.0).abs() < 1e-4);
    }

    #[test]
    fn distance_kind_reported_correctly() {
        assert_eq!(L2Distance.kind(), DistanceKind::L2);
        assert_eq!(CosineDistance.kind(), DistanceKind::Cosine);
        assert_eq!(InnerProductDistance.kind(), DistanceKind::InnerProduct);
    }

    #[test]
    fn cosine_validate_vector_rejects_zero() {
        let d = CosineDistance;
        assert_eq!(d.validate_vector(&[0.0, 0.0]), Err(CoreError::ZeroVector));
        assert!(d.validate_vector(&[1.0, 0.0]).is_ok());
    }
}
