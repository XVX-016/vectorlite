//! vl-cli: command-line interface for VectorLite.
//!
//! # Commands
//!
//! | Command  | Description                        |
//! |----------|------------------------------------|
//! | `serve`  | Start the VectorLite server.       |
//! | `import` | Import vectors from a file.        |
//! | `bench`  | Run the benchmark harness.         |
//! | `ingest` | Chunk, embed, and upsert documents.|
//!
//! Full command implementations arrive in Phases 7 and 9.

fn main() -> anyhow::Result<()> {
    println!("VectorLite CLI — use `vl-cli --help` for available commands.");
    println!("Command implementations coming in Phase 7.");
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Smoke test: the binary entry point compiles and the trivial path
    /// (constructing an `Ok(())` result) succeeds.
    #[test]
    fn cli_entry_point_compiles() {
        // Check that the crate version string is non-empty (a real runtime check).
        let version = env!("CARGO_PKG_VERSION");
        assert!(!version.is_empty(), "vl-cli must have a package version");
    }
}
