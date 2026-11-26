# Architecture Decision Record: HNSW Index (`vl-index`)

## Context
VectorLite needs an in-memory approximate nearest-neighbor (ANN) index implementing `SearchIndex` with high recall (`>= 0.95` at `k=10` on SIFT) and sub-millisecond single-threaded query latency.

## Single-Threaded Design (`HnswIndex`)

1. **Contiguous Vector Arena (`Vec<f32>`)**
   - Vectors are stored in one flat `Vec<f32>` (`offset = node_id * dim`) with `u32` internal node IDs and a separate `ids: Vec<VectorId>` table.
   - *Trade-off*: Eliminates per-vector heap allocations and pointer chasing during graph traversal, at the cost of `u32::MAX` (~4B) node limit.

2. **Flat Fixed-Capacity Neighbor Lists (No `Vec<Vec<>>`)**
   - Layer 0 neighbors are stored in `l0_neighbors: Vec<u32>` with stride `M0 = 2 * M` and `l0_counts: Vec<u16>`.
   - Upper layers (`1..=level`) are stored in a flat `upper_neighbors: Vec<u32>` arena with stride `M` per layer slot, indexed by `upper_offsets: Vec<u32>`.
   - *Trade-off*: Slightly higher memory reservation for unused slots in sparse nodes, but zero per-node heap allocations and predictable cache-friendly access.

3. **Pre-Resolved SIMD Dispatch & Cosine Normalization**
   - `vl_core::kernels::resolve_kernel` resolves CPU feature detection (AVX2+FMA / NEON / scalar) once at `HnswIndex::new` into a `fn(&[f32], &[f32]) -> f32` pointer.
   - Input validation (`dim`, finiteness, non-zero norm for cosine) happens only at `insert` and `search` entry; inner loops use the unchecked kernel.
   - Cosine vectors are normalized at `insert` (and queries into scratch at `search`), reducing cosine distance to `1.0 - dot(a, b)`.

4. **Reusable `SearchScratch` & Generation-Stamped Visited Array**
   - `SearchScratch` holds `visited: Vec<u32>`, `gen: u32`, `candidates` (min-heap), `results` (max-heap), and `query_buf`.
   - `SearchIndex::search(&self, ...)` uses a `thread_local!` `RefCell<SearchScratch>` for zero-allocation reuse across queries. On `gen == u32::MAX`, `visited` is zeroed and `gen` wraps to `1`.

5. **Heuristic Pruning & Deterministic Seeding**
   - `SELECT-NEIGHBORS-HEURISTIC` (Malkov & Yashunin Algorithm 4, `keepPrunedConnections = true`) is used both for forward edges and when a back-link causes a neighbor list to overflow `M` / `M0`.
   - A pure-Rust `SplitMix64` PRNG seeded by `HnswConfig::seed` ensures deterministic graph construction and reproducible benchmarks.
