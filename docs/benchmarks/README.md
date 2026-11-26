# VectorLite Benchmarks

This directory contains reproducible benchmark outputs (JSON/CSV) and summary tables for VectorLite.

## Dataset Provenance

- **SIFT1M** (`data/sift/`): Downloaded from `ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz` via `scripts/get_sift1m.ps1` / `scripts/get_sift1m.sh`.
  - `sift.tar.gz` SHA-256: `92f1270c5e3a0cb46b89983e72b0511e4df065c31a9fa0276d8c9b1fca5bc81a`
- **SIFTsmall** (`data/siftsmall/`): 10k base vectors, 100 queries (`dim = 128`, L2).
- **SIFT100k** (`--limit 100000`): First 100,000 vectors of `sift_base.fvecs` with exact L2 ground truth computed via `BruteForceIndex` and cached to `data/sift100k_gt.ivecs` (`--gt-cache`).

## Phase 2 Dependencies

| Crate | Dependency | Justification |
| :--- | :--- | :--- |
| `vl-index` | `ordered-float = "4"` | Provides `NotNan<f32>` total ordering (`Ord`/`Eq`) for `BinaryHeap` top-k search without NaN panics. |
| `vl-bench` | `clap = "4"` | Derive-based CLI argument parsing for the benchmark runner binary. |
| `vl-bench` | `serde = "1"` | Derive serialization for `BenchmarkResult` metadata and metrics. |
| `vl-bench` | `serde_json = "1"` | Pretty-printed JSON output for benchmark results. |
| `vl-bench` | `csv = "1"` | Single-row CSV output for benchmark results. |
| `vl-bench` | `hdrhistogram = "7"` | High-dynamic-range latency histogram for accurate `p50`/`p95`/`p99` microsecond percentiles. |
| `vl-bench` | `rayon = "1"` | Parallel query execution for multi-threaded QPS and exact ground-truth generation. |
| `vl-bench` | `sysinfo = "0.32"` | Cross-platform CPU model, OS version, and safe process RSS memory sampling. |
| `vl-bench` | `time = "0.3"` | RFC-3339 UTC timestamp formatting for benchmark run metadata. |
| `vl-bench` | `criterion = "0.5"` | Microbenchmarking harness for `benches/distance.rs` (dev-dependency). |
| `vl-bench` | `proptest = "1"` | Property-based testing for `fvecs`/`ivecs` round-trip parsing (dev-dependency). |
| `vl-bench` | `tempfile = "3"` | Temporary directory management in `io.rs` unit tests (dev-dependency). |

## HNSW Recall vs. Latency (`M=16`, `M0=32`, `ef_construction=200`, `k=10`)

### SIFT100k (`N = 100,000`, `Q = 10,000`, `dim = 128`, L2, single-threaded build = `54.06s`)

| `ef_search` | `recall@10` | `recall_tie_aware` | `p50 (us)` | `p95 (us)` | `p99 (us)` | `QPS (32T)` | `peak RSS (MB)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.8512 | 0.8513 | 39.4 | 57.2 | 70.5 | 277511.1 | 152.0 |
| 20 | 0.9363 | 0.9364 | 68.0 | 92.0 | 114.2 | 176279.2 | 157.2 |
| 40 | 0.9812 | 0.9812 | 119.0 | 166.0 | 221.1 | 99897.3 | 157.0 |
| 80 | 0.9963 | 0.9963 | 196.6 | 289.5 | 453.1 | 61929.5 | 157.5 |
| 160 | 0.9993 | 0.9993 | 312.6 | 465.7 | 603.1 | 29435.0 | 157.9 |
| 320 | 0.9999 | 0.9999 | 417.3 | 725.0 | 979.5 | 24002.4 | 158.8 |

### SIFT1M (`N = 1,000,000`, `Q = 10,000`, `dim = 128`, L2, single-threaded build = `718.78s`)

| `ef_search` | `recall@10` | `recall_tie_aware` | `p50 (us)` | `p95 (us)` | `p99 (us)` | `QPS (32T)` | `peak RSS (MB)` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.7694 | 0.7703 | 52.5 | 84.2 | 173.3 | 173319.2 | 1322.0 |
| 20 | 0.8786 | 0.8795 | 78.2 | 120.4 | 252.0 | 120541.8 | 1330.3 |
| 40 | 0.9496 | 0.9504 | 147.1 | 218.0 | 457.0 | 71640.1 | 1330.2 |
| 80 | 0.9838 | 0.9844 | 262.9 | 392.7 | 891.9 | 42384.6 | 1330.7 |
| 160 | 0.9957 | 0.9963 | 481.0 | 689.2 | 1629.2 | 22773.5 | 1330.7 |
| 320 | 0.9989 | 0.9995 | 883.2 | 1283.1 | 2850.8 | 13565.3 | 1330.9 |
