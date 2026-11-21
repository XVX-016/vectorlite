//! vl-server library: shared server components.
//!
//! This crate exposes the Axum REST and Tonic gRPC handlers, request
//! validation, Prometheus metrics, structured logging, and config loading.
//! The binary entry point in `main.rs` wires these together.

/// Returns the current server version string.
///
/// # Examples
///
/// ```
/// assert!(!vl_server::version().is_empty());
/// ```
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        assert!(!version().is_empty());
    }
}
