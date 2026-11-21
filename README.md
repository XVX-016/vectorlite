# VectorLite

**A lightweight, embeddable vector database written in Rust for private, on-device semantic search.**

> ⚠️ This project is under active development. See the [roadmap](#roadmap) below.

---

## What and Why

Organizations that cannot send documents to the cloud — law firms, clinics, banks, government offices — still want semantic search and RAG. Existing options are either cloud-first or heavy to run, and most tooling is English-first.

VectorLite is:

- **Single binary, fully offline** — with quantization to keep memory low.
- **Filtered + hybrid search** — vector similarity and BM25 keyword search as first-class features.
- **Multilingual** — designed with Telugu, Hindi, and English in mind.

**Non-goals:** distributed clustering, replication, full SQL, or beating Qdrant/Milvus at billion-scale. VectorLite is intentionally scoped to the laptop/edge/private deployment use case.

---

## Quickstart

> Detailed quickstart will be added in Phase 7 (Server) and Phase 8 (Docker packaging).

```bash
# Build from source (requires Rust stable ≥ 1.75)
cargo build --release

# Start the server (Phase 7)
./target/release/vl-server --config config.toml

# Run the CLI (Phase 7)
./target/release/vl-cli serve
```

---

## Architecture

```
vectorlite/
  Cargo.toml             workspace
  proto/vectorlite.proto gRPC schema
  crates/
    vl-core/     vector types, distance kernels (SIMD), errors
    vl-index/    brute force, HNSW, quantizers (SQ8, PQ)
    vl-storage/  WAL, segment files, mmap, snapshots, recovery
    vl-filter/   metadata store, filter bitmaps, BM25 inverted index
    vl-server/   Axum REST, Tonic gRPC, metrics, config
    vl-bench/    dataset loaders, recall and latency harness
    vl-cli/      serve, import, bench commands
  clients/python/
  web/           React frontend
  docs/          architecture, decisions, benchmark results
```

**Write path:** request → validate → WAL append → in-memory index insert → periodic segment flush → WAL truncate.

**Read path:** request → parse filter into bitmap → HNSW search with ef_search → optional rerank → return results + timing.

---

## Features (Roadmap)

| Phase | Feature | Status |
|-------|---------|--------|
| 0 | Workspace, CI, project structure | ✅ Done |
| 1 | SIMD distance kernels, brute-force index | 🔜 Next |
| 2 | Benchmark harness, SIFT1M recall | 🔜 |
| 3 | HNSW index | 🔜 |
| 4 | WAL, crash-safe persistence | 🔜 |
| 5 | SQ8 / PQ quantization | 🔜 |
| 6 | Metadata filtering, BM25, hybrid search | 🔜 |
| 7 | REST + gRPC server, Prometheus metrics | 🔜 |
| 8 | Python client, Docker | 🔜 |
| 9 | Multilingual embedding pipeline | 🔜 |
| 10 | React dashboard and search UI | 🔜 |
| 11 | Competitor benchmarks, v0.1.0 release | 🔜 |

---

## Benchmarks

> Benchmark numbers will be added after Phase 2 and updated through each subsequent phase.
> All numbers will come from reproducible commands stored in `docs/benchmarks/`.

---

## Limitations

- Not distributed — single node only.
- No replication or clustering.
- Not a full SQL database.
- Performance numbers are hardware-dependent (see `docs/benchmarks/`).

---

## Development

```bash
# Run all checks (required before every commit)
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test --all
```

---

## License

Licensed under the [MIT License](LICENSE-MIT).
