//! vl-filter: metadata filtering and hybrid search primitives.
//!
//! This crate stores per-vector JSON metadata, indexes equality and range
//! filters using roaring bitmaps, provides a filter expression language
//! (and, or, not, eq, range, in), and implements a BM25 inverted index
//! over text fields for hybrid (vector + keyword) queries.

/// Placeholder error type for filter operations (populated in Phase 6).
#[derive(Debug, thiserror::Error)]
pub enum FilterError {
    /// A filter expression could not be parsed.
    #[error("invalid filter expression: {reason}")]
    InvalidExpression {
        /// Human-readable explanation.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_expression_error_formats_correctly() {
        let err = FilterError::InvalidExpression {
            reason: "unexpected token '?'".into(),
        };
        let msg = err.to_string();
        assert!(
            msg.contains("unexpected token"),
            "error should contain the reason"
        );
    }
}
