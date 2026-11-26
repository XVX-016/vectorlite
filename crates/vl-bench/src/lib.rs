//! vl-bench: benchmark harness for VectorLite.
//!
//! This crate provides readers for the standard `fvecs`/`ivecs` file formats
//! (used by SIFT1M and other ANN benchmark datasets), and a measurement
//! harness that computes `recall@k` and latency percentiles (`p50`, `p95`, `p99`)
//! plus multi-threaded QPS over any index implementing [`vl_index::SearchIndex`].

pub mod harness;
pub mod io;

/// Returns the current benchmark harness version string.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Error type for benchmark operations.
#[derive(Debug, thiserror::Error)]
pub enum BenchError {
    /// The `fvecs`/`ivecs` file has an unexpected format.
    #[error("malformed benchmark file: {reason}")]
    MalformedFile {
        /// Human-readable explanation.
        reason: String,
    },
    /// Invalid benchmark configuration or arguments.
    #[error("invalid benchmark configuration: {0}")]
    InvalidConfig(String),
    /// An underlying index error.
    #[error("index error: {0}")]
    Index(#[from] vl_index::IndexError),
    /// An underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_file_error_formats_correctly() {
        let err = BenchError::MalformedFile {
            reason: "unexpected EOF".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("unexpected EOF"));
    }
}
