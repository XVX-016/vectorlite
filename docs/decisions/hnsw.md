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

---

## Concurrent Design (`ConcurrentHnswIndex`)

### Problem

The single-threaded `HnswIndex` requires exclusive access during inserts because it mutates shared in-place `Vec` neighbor lists. Enabling a parallel build (N inserting threads) and simultaneous read-while-write requires a different ownership model.

### Design Comparison

Two options were evaluated:

| | Option A: Per-Node `Mutex<()>` + Atomic Arrays | Option B: Batch-Parallel Build |
|---|---|---|
| **Concurrent insert + search** | ✅ Yes | ❌ No (snapshot-only) |
| **Locking granularity** | Per node — coarse contention only near high-degree hubs | No locks during build |
| **Recall quality** | Identical to ST index within noise | Degrades during subgraph merge |
| **Incremental inserts** | ✅ Supported | ❌ Requires full rebuild |

**Option A was chosen** because VectorLite requires concurrent reads during inserts and incremental insert semantics.

### Storage Layout

All storage is **pre-allocated** at construction from `max_elements`:

- **Vector arena** (`Vec<AtomicU32>`, length `max_elements * dim`): stores `f32::to_bits()` values so each slot is a single-word atomic write, keeping 100% safe Rust with no `unsafe`.
- **External ID table** (`Vec<AtomicU64>`): maps internal `u32` node IDs to external `VectorId`.
- **Layer-0 neighbor array** (`Vec<AtomicU32>`, length `max_elements * M0`): flat, stride `M0 = 2*M`, sentinel `u32::MAX` for empty slots; `Vec<AtomicU16>` per-node counts.
- **Upper layers** (`Vec<OnceLock<UpperLayerStorage>>`): allocated exactly once per node with `OnceLock`, guaranteeing that after initialization the pointer is stable and readers need no lock.

Pre-allocation eliminates any reallocation during inserts, which would otherwise invalidate concurrent reader pointers.

### Synchronization

Three synchronization mechanisms, one per data granularity:

1. **Per-node `Mutex<()>`** (`node_locks: Vec<Mutex<()>>`): a writer acquires the lock for a neighbor node before rewriting its neighbor list with `select_neighbors_heuristic`. Writers hold at most one node lock at a time (insert order prevents cycles), guaranteeing deadlock freedom.

2. **Lock-free readers on neighbor lists**: readers load the count with `Acquire`, then read slots with `Relaxed`, skipping `u32::MAX` sentinels. A reader that observes a stale count or partial write sees a subset of valid neighbors — it may miss edges temporarily added by a concurrent writer, but it never dereferences an invalid node ID.

3. **Global `RwLock<Option<(u32, usize)>>`** for `entry_point` and `max_level`: write-locked only when a new node's sampled level exceeds the current `max_level`, which happens O(log N) times total. All other insert paths take a cheap read lock.

### Safety (100% Safe Rust)

No `unsafe` blocks are used. The key insight is that `AtomicU32` accepts any bit-pattern — including the bits of a partially-constructed `f32` — so writing `f32::to_bits()` atomically is memory-safe and meaningful, since any observed value is either the initial `0.0` or the fully-written float.

### Determinism

The single-threaded `HnswIndex` remains unchanged and fully deterministic. The parallel `ConcurrentHnswIndex` is **non-deterministic** because rayon thread scheduling varies between runs. Tests assert recall within tolerance (`± 0.01`) rather than exact bit-equality.

### Capacity

Inserting beyond `max_elements` returns `IndexError::CapacityExceeded`. Dynamic growth via arena segments is deferred to a future phase.

### Benchmark Results (SIFT1M, M=16, M0=32, ef_construction=200)

| Threads | Build Time | Speedup | Recall@10 (ef=40) | Recall@10 (ef=160) |
|---------|-----------|---------|-------------------|---------------------|
| 1       | 3685.74 s | 1.00×   | 0.9661            | 0.9983              |
| 8       | 765.83 s  | 4.81×   | 0.9663            | 0.9982              |
| 32      | 588.05 s  | 6.27×   | 0.9658            | 0.9981              |

Recall is stable within noise across all thread counts.
