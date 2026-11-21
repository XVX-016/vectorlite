//! vl-server: entry point for the VectorLite HTTP/gRPC server.
//!
//! Run with:
//! ```text
//! vl-server --config config.toml
//! ```

fn main() -> anyhow::Result<()> {
    println!("VectorLite server v{}", vl_server::version());
    println!("Server implementation coming in Phase 7.");
    Ok(())
}
