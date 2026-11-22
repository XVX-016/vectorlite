//! Vector identifier and owned vector types.

use crate::CoreError;

/// Opaque 64-bit identifier for a stored vector.
///
/// Identifiers are assigned by the caller; VectorLite treats them as opaque
/// keys and never interprets the bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VectorId(pub u64);

impl std::fmt::Display for VectorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VectorId({})", self.0)
    }
}

/// An owned, dimension-validated f32 vector paired with its identifier.
///
/// The `data` field is a plain `Vec<f32>` so that callers control allocation.
/// Dimension validation happens at construction time via [`Vector::new`].
#[derive(Debug, Clone, PartialEq)]
pub struct Vector {
    /// The unique identifier of this vector.
    pub id: VectorId,
    /// The raw f32 components.
    pub data: Vec<f32>,
}

impl Vector {
    /// Create a new [`Vector`], returning an error if `data` is empty or contains non-finite values.
    ///
    /// We disallow zero-dimensional vectors and NaN/infinite components.
    pub fn new(id: VectorId, data: Vec<f32>) -> Result<Self, CoreError> {
        if data.is_empty() {
            return Err(CoreError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        for &x in &data {
            if !x.is_finite() {
                return Err(CoreError::NonFiniteValue);
            }
        }
        Ok(Self { id, data })
    }

    /// The dimensionality of this vector.
    #[inline]
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_new_rejects_empty() {
        let err = Vector::new(VectorId(0), vec![]).unwrap_err();
        assert!(err.to_string().contains("dimension mismatch"));
    }

    #[test]
    fn vector_new_accepts_nonempty() {
        let v = Vector::new(VectorId(42), vec![1.0, 2.0, 3.0]).unwrap();
        assert_eq!(v.dim(), 3);
        assert_eq!(v.id, VectorId(42));
    }

    #[test]
    fn vector_id_display() {
        assert_eq!(VectorId(7).to_string(), "VectorId(7)");
    }
}
