//! vl-bench: benchmark runner entry point.
//!
//! Example:
//! ```text
//! vl-bench --dataset sift1m --base data/sift/sift_base.fvecs \
//!   --query data/sift/sift_query.fvecs \
//!   --groundtruth data/sift/sift_groundtruth.ivecs \
//!   --metric l2 --k 10 \
//!   --out-json docs/benchmarks/sift1m_brute_force.json \
//!   --out-csv docs/benchmarks/sift1m_brute_force.csv
//! ```

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use vl_bench::harness::{
    compute_recall_at_k, compute_recall_tie_aware, format_console_table, run_benchmark, write_csv,
    write_json, BenchmarkConfig,
};
use vl_bench::io::{read_fvecs_limit, read_groundtruth_ivecs, write_groundtruth_ivecs};
use vl_core::{CosineDistance, Distance, DistanceKind, InnerProductDistance, L2Distance, VectorId};
use vl_index::{
    BruteForceIndex, ConcurrentHnswIndex, HnswConfig, HnswIndex, IndexError, SearchIndex,
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum IndexArg {
    BruteForce,
    Hnsw,
    ConcurrentHnsw,
}

struct HnswEfView<'a> {
    index: &'a HnswIndex,
    ef_search: usize,
}

impl SearchIndex for HnswEfView<'_> {
    fn insert(&mut self, _id: VectorId, _vector: Vec<f32>) -> Result<(), IndexError> {
        unreachable!("read-only benchmark view")
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        self.index.search_with_ef(query, k, self.ef_search)
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn dim(&self) -> usize {
        self.index.dim()
    }

    fn metric(&self) -> DistanceKind {
        self.index.metric()
    }
}

struct ConcurrentHnswEfView<'a> {
    index: &'a ConcurrentHnswIndex,
    ef_search: usize,
}

impl SearchIndex for ConcurrentHnswEfView<'_> {
    fn insert(&mut self, _id: VectorId, _vector: Vec<f32>) -> Result<(), IndexError> {
        unreachable!("read-only benchmark view")
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        self.index.search_with_ef(query, k, self.ef_search)
    }

    fn len(&self) -> usize {
        self.index.len()
    }

    fn dim(&self) -> usize {
        self.index.dim()
    }

    fn metric(&self) -> DistanceKind {
        self.index.metric()
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum MetricArg {
    L2,
    Cosine,
    Ip,
}

impl MetricArg {
    fn to_distance(self) -> Arc<dyn Distance> {
        match self {
            MetricArg::L2 => Arc::new(L2Distance),
            MetricArg::Cosine => Arc::new(CosineDistance),
            MetricArg::Ip => Arc::new(InnerProductDistance),
        }
    }
}

#[derive(Debug, Parser)]
#[command(name = "vl-bench", version, about = "VectorLite ANN benchmark harness")]
struct Cli {
    /// Dataset name recorded in output metadata.
    #[arg(long, default_value = "sift1m")]
    dataset: String,

    /// Path to base vectors (.fvecs).
    #[arg(long)]
    base: PathBuf,

    /// Path to query vectors (.fvecs).
    #[arg(long)]
    query: PathBuf,

    /// Path to ground-truth neighbor IDs (.ivecs). Evaluated only when metric is L2.
    #[arg(long)]
    groundtruth: Option<PathBuf>,

    /// Distance metric.
    #[arg(long, value_enum, default_value_t = MetricArg::L2)]
    metric: MetricArg,

    /// Top-k neighbors to retrieve.
    #[arg(long, default_value_t = 10)]
    k: usize,

    /// Optional limit on the number of base vectors to load.
    #[arg(long)]
    limit: Option<usize>,

    /// Number of warmup queries before latency measurement.
    #[arg(long, default_value_t = 100)]
    warmup: usize,

    /// Number of threads for multi-threaded QPS measurement (0 = all logical cores).
    #[arg(long, default_value_t = 0)]
    threads: usize,

    /// Optional path to write JSON benchmark output.
    #[arg(long)]
    out_json: Option<PathBuf>,

    /// Optional path to write CSV benchmark output.
    #[arg(long)]
    out_csv: Option<PathBuf>,

    /// Path to read/write cached exact ground truth (.ivecs) for dataset subsets.
    #[arg(long)]
    gt_cache: Option<PathBuf>,

    /// Run only parallel recall & tie-check verification (skip timing passes).
    #[arg(long, default_value_t = false)]
    recall_only: bool,

    /// Index implementation to benchmark.
    #[arg(long, value_enum, default_value_t = IndexArg::BruteForce)]
    index: IndexArg,

    /// HNSW M parameter (max neighbors per upper-layer node; M0 = 2*M).
    #[arg(long, default_value_t = 16)]
    m: usize,

    /// HNSW ef_construction parameter.
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,

    /// HNSW ef_search parameter (used when --ef-sweep is not set).
    #[arg(long, default_value_t = 50)]
    ef_search: usize,

    /// Comma-separated list of ef_search values to sweep for HNSW (e.g. 10,20,40,80,160,320).
    #[arg(long, value_delimiter = ',')]
    ef_sweep: Option<Vec<usize>>,

    /// Deterministic RNG seed for HNSW construction.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn compute_exact_groundtruth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    dim: usize,
    k: usize,
) -> anyhow::Result<Vec<Vec<VectorId>>> {
    let mut bf = BruteForceIndex::new(dim, Arc::new(L2Distance));
    for (id, vec) in base.iter().enumerate() {
        bf.insert(VectorId(id as u64), vec.clone())?;
    }
    let gt: Vec<Vec<VectorId>> = queries
        .par_iter()
        .map(|q| {
            bf.search(q, k)
                .map(|res| res.into_iter().map(|(id, _)| id).collect())
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(gt)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("Loading base vectors from {} ...", cli.base.display());
    let base = read_fvecs_limit(&cli.base, cli.limit)?;
    let dim = base[0].len();
    println!("Loaded {} base vectors (dim={}).", base.len(), dim);

    println!("Loading query vectors from {} ...", cli.query.display());
    let queries = read_fvecs_limit(&cli.query, None)?;
    println!("Loaded {} query vectors.", queries.len());

    let groundtruth = if let Some(cache_path) = &cli.gt_cache {
        if cache_path.exists() {
            println!(
                "Loading cached ground truth from {} ...",
                cache_path.display()
            );
            Some(read_groundtruth_ivecs(cache_path, None)?)
        } else {
            println!(
                "Computing exact L2 ground truth (k={}) for {} base vectors ...",
                cli.k,
                base.len()
            );
            let t0 = Instant::now();
            let gt = compute_exact_groundtruth(&base, &queries, dim, cli.k)?;
            write_groundtruth_ivecs(cache_path, &gt)?;
            println!(
                "Computed and cached ground truth to {} in {:.2}s.",
                cache_path.display(),
                t0.elapsed().as_secs_f64()
            );
            Some(gt)
        }
    } else if let Some(path) = &cli.groundtruth {
        if cli.limit.is_some() {
            println!(
                "Warning: --limit is set with --groundtruth; provided GT may be invalid for subsets. Consider --gt-cache."
            );
        }
        println!("Loading ground truth from {} ...", path.display());
        Some(read_groundtruth_ivecs(path, None)?)
    } else {
        None
    };

    let build_start = Instant::now();
    match cli.index {
        IndexArg::BruteForce => {
            let mut index = BruteForceIndex::new(dim, cli.metric.to_distance());
            for (id, vec) in base.iter().enumerate() {
                index.insert(VectorId(id as u64), vec.clone())?;
            }
            let build_time_secs = build_start.elapsed().as_secs_f64();
            let index_memory_bytes = index.memory_bytes();

            if cli.recall_only {
                let gt = groundtruth.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("--recall-only requires --groundtruth or --gt-cache")
                })?;
                println!("Running parallel recall verification (k={}) ...", cli.k);
                let results: Vec<Vec<(VectorId, f32)>> = queries
                    .par_iter()
                    .map(|q| index.search(q, cli.k))
                    .collect::<Result<Vec<_>, _>>()?;
                let exact_recall = compute_recall_at_k(&results, gt, cli.k)?;
                let (tie_recall, stats) = compute_recall_tie_aware(
                    &results,
                    &queries,
                    gt,
                    &base,
                    DistanceKind::L2,
                    cli.k,
                    0.0,
                )?;
                println!("recall@{}:          {exact_recall:.10}", cli.k);
                println!("recall_tie_aware:  {tie_recall:.10}");
                println!(
                    "queries with recall < 1: {} / {}",
                    stats.queries_with_exact_mismatch, stats.total_queries
                );
                println!("non-tie mismatches: {}", stats.non_tie_mismatches);
                return Ok(());
            }

            let mut index_params = BTreeMap::new();
            index_params.insert("type".into(), "exact_linear_scan".into());

            let config = BenchmarkConfig {
                dataset: cli.dataset,
                index_name: "brute_force".into(),
                index_params,
                k: cli.k,
                warmup_queries: cli.warmup,
                qps_threads: cli.threads,
                build_time_secs,
                index_memory_bytes,
                qps_duration_secs: 10.0,
            };

            println!("Running benchmark (warmup={}, k={}) ...", cli.warmup, cli.k);
            let result = run_benchmark(
                &index as &dyn SearchIndex,
                &queries,
                groundtruth.as_deref(),
                Some(&base),
                &config,
            )?;

            println!("\n{}", format_console_table(&result));

            if let Some(json_path) = &cli.out_json {
                write_json(json_path, &result)?;
                println!("Saved JSON to {}", json_path.display());
            }
            if let Some(csv_path) = &cli.out_csv {
                write_csv(csv_path, &result)?;
                println!("Saved CSV to {}", csv_path.display());
            }
        }
        IndexArg::Hnsw => {
            let ef_list = cli.ef_sweep.clone().unwrap_or_else(|| vec![cli.ef_search]);
            let cfg = HnswConfig::new(cli.m, cli.ef_construction, ef_list[0], cli.seed);
            println!(
                "Building HNSW index (M={}, M0={}, ef_construction={}, seed={}) ...",
                cfg.m, cfg.m0, cfg.ef_construction, cfg.seed
            );
            let mut index = HnswIndex::new(dim, cli.metric.to_distance(), cfg);
            for (id, vec) in base.iter().enumerate() {
                index.insert(VectorId(id as u64), vec.clone())?;
            }
            let build_time_secs = build_start.elapsed().as_secs_f64();
            let index_memory_bytes = index.memory_bytes();
            println!(
                "Built HNSW index in {build_time_secs:.2}s (max_level={}).",
                index.max_level()
            );

            for ef in ef_list {
                let mut index_params = BTreeMap::new();
                index_params.insert("M".into(), cfg.m.to_string());
                index_params.insert("M0".into(), cfg.m0.to_string());
                index_params.insert("ef_construction".into(), cfg.ef_construction.to_string());
                index_params.insert("ef_search".into(), ef.to_string());
                index_params.insert("seed".into(), cfg.seed.to_string());

                let config = BenchmarkConfig {
                    dataset: cli.dataset.clone(),
                    index_name: "hnsw".into(),
                    index_params,
                    k: cli.k,
                    warmup_queries: cli.warmup,
                    qps_threads: cli.threads,
                    build_time_secs,
                    index_memory_bytes,
                    qps_duration_secs: 10.0,
                };

                println!(
                    "Running HNSW benchmark (ef_search={ef}, warmup={}, k={}) ...",
                    cli.warmup, cli.k
                );
                let view = HnswEfView {
                    index: &index,
                    ef_search: ef,
                };
                let result = run_benchmark(
                    &view as &dyn SearchIndex,
                    &queries,
                    groundtruth.as_deref(),
                    Some(&base),
                    &config,
                )?;

                println!("\n{}", format_console_table(&result));

                if let Some(json_path) = &cli.out_json {
                    let path = if cli.ef_sweep.is_some() {
                        with_ef_suffix(json_path, ef)
                    } else {
                        json_path.clone()
                    };
                    write_json(&path, &result)?;
                    println!("Saved JSON to {}", path.display());
                }
                if let Some(csv_path) = &cli.out_csv {
                    let path = if cli.ef_sweep.is_some() {
                        with_ef_suffix(csv_path, ef)
                    } else {
                        csv_path.clone()
                    };
                    write_csv(&path, &result)?;
                    println!("Saved CSV to {}", path.display());
                }
            }
        }
        IndexArg::ConcurrentHnsw => {
            let ef_list = cli.ef_sweep.clone().unwrap_or_else(|| vec![cli.ef_search]);
            let cfg = HnswConfig::new(cli.m, cli.ef_construction, ef_list[0], cli.seed);
            let pool = if cli.threads > 0 {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cli.threads)
                    .build()?
            } else {
                rayon::ThreadPoolBuilder::new().build()?
            };
            let build_threads = pool.current_num_threads();
            println!(
                "Building ConcurrentHnswIndex (M={}, M0={}, ef_construction={}, threads={build_threads}) ...",
                cfg.m, cfg.m0, cfg.ef_construction
            );
            let index = ConcurrentHnswIndex::new(dim, cli.metric.to_distance(), cfg, base.len());
            let items: Vec<(VectorId, Vec<f32>)> = base
                .iter()
                .enumerate()
                .map(|(id, vec)| (VectorId(id as u64), vec.clone()))
                .collect();
            let par_build_start = Instant::now();
            pool.install(|| index.insert_batch_parallel(&items))?;
            let build_time_secs = par_build_start.elapsed().as_secs_f64();
            let index_memory_bytes = index.memory_bytes();
            println!("Built ConcurrentHnswIndex in {build_time_secs:.2}s.");

            for ef in ef_list {
                let mut index_params = BTreeMap::new();
                index_params.insert("M".into(), cfg.m.to_string());
                index_params.insert("M0".into(), cfg.m0.to_string());
                index_params.insert("ef_construction".into(), cfg.ef_construction.to_string());
                index_params.insert("ef_search".into(), ef.to_string());
                index_params.insert("build_threads".into(), build_threads.to_string());

                let config = BenchmarkConfig {
                    dataset: cli.dataset.clone(),
                    index_name: "concurrent_hnsw".into(),
                    index_params,
                    k: cli.k,
                    warmup_queries: cli.warmup,
                    qps_threads: cli.threads,
                    build_time_secs,
                    index_memory_bytes,
                    qps_duration_secs: 10.0,
                };

                let view = ConcurrentHnswEfView {
                    index: &index,
                    ef_search: ef,
                };
                let result = run_benchmark(
                    &view as &dyn SearchIndex,
                    &queries,
                    groundtruth.as_deref(),
                    Some(&base),
                    &config,
                )?;
                println!("\n{}", format_console_table(&result));

                if let Some(json_path) = &cli.out_json {
                    let path = if cli.ef_sweep.is_some() {
                        with_ef_suffix(json_path, ef)
                    } else {
                        json_path.clone()
                    };
                    write_json(&path, &result)?;
                }
                if let Some(csv_path) = &cli.out_csv {
                    let path = if cli.ef_sweep.is_some() {
                        with_ef_suffix(csv_path, ef)
                    } else {
                        csv_path.clone()
                    };
                    write_csv(&path, &result)?;
                }
            }
        }
    }

    Ok(())
}

fn with_ef_suffix(path: &std::path::Path, ef: usize) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("bench");
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let file_name = if ext.is_empty() {
        format!("{stem}_ef{ef}")
    } else {
        format!("{stem}_ef{ef}.{ext}")
    };
    path.with_file_name(file_name)
}
