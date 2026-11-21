//! vl-bench: benchmark runner entry point.
//!
//! Run with:
//! ```text
//! vl-bench --dataset data/sift1m --index hnsw --ef-search 128
//! ```

fn main() -> anyhow::Result<()> {
    println!("VectorLite benchmark harness v{}", vl_bench::version());
    println!("Harness implementation coming in Phase 2.");
    Ok(())
}
