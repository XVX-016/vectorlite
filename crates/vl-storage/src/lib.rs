//! vl-storage: durability layer.
//!
//! This crate owns the write-ahead log (WAL), immutable segment files,
//! memory-mapped loading, snapshot creation, and crash recovery. It is
//! designed around the invariant that every acknowledged write survives
//! a `kill -9`.

/// Placeholder error type for storage operations (populated in Phase 4).
#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    /// A WAL record checksum did not match; the tail may be torn.
    #[error("checksum mismatch in WAL record at offset {offset}")]
    ChecksumMismatch {
        /// Byte offset of the offending record.
        offset: u64,
    },
    /// An underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checksum_mismatch_error_formats_correctly() {
        let err = StorageError::ChecksumMismatch { offset: 1024 };
        let msg = err.to_string();
        assert!(msg.contains("1024"), "error should contain the offset");
    }
}
