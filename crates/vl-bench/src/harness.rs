//! Benchmark harness: recall@k, single-threaded latency percentiles, and multi-threaded QPS.

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

use hdrhistogram::Histogram;
use rayon::prelude::*;
use serde::Serialize;
use sysinfo::System;
use time::format_description::well_known::Rfc3339;
use time::OffsetDateTime;
use vl_core::kernels::resolve_kernel;
use vl_core::{DistanceKind, VectorId};
use vl_index::SearchIndex;

use crate::BenchError;

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Human-readable dataset name (e.g., `"siftsmall"` or `"sift1m"`).
    pub dataset: String,
    /// Index name (e.g., `"brute_force"`).
    pub index_name: String,
    /// Index-specific parameters recorded in the JSON output.
    pub index_params: BTreeMap<String, String>,
    /// Number of nearest neighbors to retrieve (`k` in `recall@k`).
    pub k: usize,
    /// Number of warmup queries run before latency measurement.
    pub warmup_queries: usize,
    /// Number of threads for the multi-threaded QPS measurement (0 = Rayon default).
    pub qps_threads: usize,
    /// Time spent building/loading the index, in seconds.
    pub build_time_secs: f64,
    /// Memory owned by the index itself, in bytes.
    pub index_memory_bytes: usize,
    /// Target duration for the multi-threaded QPS pass, in seconds.
    pub qps_duration_secs: f64,
}

/// Complete output of a benchmark run, serialized to JSON and CSV.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkResult {
    /// Short git commit hash of the repository at run time (`-dirty` if uncommitted changes exist).
    pub git_commit: String,
    /// `rustc --version` string.
    pub rustc_version: String,
    /// Host CPU model string.
    pub cpu_model: String,
    /// Host operating system name and version.
    pub os: String,
    /// Dataset name.
    pub dataset: String,
    /// Distance metric (`L2`, `InnerProduct`, or `Cosine`).
    pub metric: String,
    /// Index implementation name.
    pub index: String,
    /// Index parameters.
    pub index_params: BTreeMap<String, String>,
    /// UTC ISO-8601 timestamp.
    pub timestamp: String,
    /// Number of indexed base vectors.
    pub num_vectors: usize,
    /// Number of queries evaluated.
    pub num_queries: usize,
    /// Vector dimensionality.
    pub dim: usize,
    /// Top-k cutoff.
    pub k: usize,
    /// Mean `recall@k` against ground truth (`None` when metric is not L2 or no GT provided).
    pub recall_at_k: Option<f64>,
    /// Mean tie-aware `recall@k` against ground truth (`dist <= d_k + tol`).
    pub recall_tie_aware: Option<f64>,
    /// Single-threaded 50th percentile query latency in microseconds.
    pub p50_us: f64,
    /// Single-threaded 95th percentile query latency in microseconds.
    pub p95_us: f64,
    /// Single-threaded 99th percentile query latency in microseconds.
    pub p99_us: f64,
    /// Multi-threaded queries per second.
    pub qps: f64,
    /// Thread count used in the multi-threaded QPS pass.
    pub qps_threads: usize,
    /// Index build time in seconds.
    pub build_time_secs: f64,
    /// Total elapsed time for the single-threaded latency pass in seconds.
    pub single_thread_total_secs: f64,
    /// Memory owned by the index itself, in bytes.
    pub index_memory_bytes: usize,
    /// OS-reported peak resident set size of the process, in bytes.
    pub peak_rss_bytes: u64,
}

/// Summary statistics from tie-aware recall verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TieCheckStats {
    /// Total number of queries evaluated.
    pub total_queries: usize,
    /// Number of queries where exact ID recall@k was `< 1.0`.
    pub queries_with_exact_mismatch: usize,
    /// Total number of returned items whose distance exceeded `d_k + tol`.
    pub non_tie_mismatches: usize,
}

/// Return the OS-reported peak resident set size (RSS) of the current process in bytes.
pub fn os_peak_rss_bytes() -> u64 {
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        let mut pmc = std::mem::MaybeUninit::<PROCESS_MEMORY_COUNTERS>::zeroed();
        let size = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
        // SAFETY: `GetCurrentProcess()` returns a valid pseudo-handle for the current process,
        // `pmc.as_mut_ptr()` points to a properly aligned `PROCESS_MEMORY_COUNTERS` struct of `size` bytes.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), pmc.as_mut_ptr(), size) };
        if ok != 0 {
            // SAFETY: `K32GetProcessMemoryInfo` succeeded and initialized `pmc`.
            let pmc = unsafe { pmc.assume_init() };
            return pmc.PeakWorkingSetSize as u64;
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    if let Some(kb_str) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(target_os = "macos")]
    {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
        // SAFETY: `RUSAGE_SELF` is valid and `usage.as_mut_ptr()` points to a `libc::rusage` buffer.
        let rc = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
        if rc == 0 {
            // SAFETY: `getrusage` returned 0 and initialized `usage`.
            let usage = unsafe { usage.assume_init() };
            return usage.ru_maxrss as u64;
        }
        0
    }
    #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

/// Compute mean `recall@k` across all queries.
///
/// For each query, compares the top-`k` predicted IDs against the first `k`
/// ground-truth IDs.
///
/// # Errors
///
/// Returns [`BenchError::InvalidConfig`] if `k == 0`, `results.len() != groundtruth.len()`,
/// or any ground-truth row is empty.
pub fn compute_recall_at_k(
    results: &[Vec<(VectorId, f32)>],
    groundtruth: &[Vec<VectorId>],
    k: usize,
) -> Result<f64, BenchError> {
    if k == 0 {
        return Err(BenchError::InvalidConfig(
            "k must be greater than 0 for recall@k".into(),
        ));
    }
    if results.len() != groundtruth.len() {
        return Err(BenchError::InvalidConfig(format!(
            "query count ({}) does not match ground-truth count ({})",
            results.len(),
            groundtruth.len()
        )));
    }
    if results.is_empty() {
        return Err(BenchError::InvalidConfig("no queries to evaluate".into()));
    }

    let mut total_recall = 0.0_f64;
    for (idx, (pred, gt)) in results.iter().zip(groundtruth.iter()).enumerate() {
        let gt_k = &gt[..k.min(gt.len())];
        if gt_k.is_empty() {
            return Err(BenchError::InvalidConfig(format!(
                "ground-truth row {idx} is empty"
            )));
        }
        let hits = pred
            .iter()
            .take(k)
            .filter(|(id, _)| gt_k.contains(id))
            .count();
        total_recall += hits as f64 / gt_k.len() as f64;
    }

    Ok(total_recall / results.len() as f64)
}

/// Compute mean tie-aware `recall@k` and mismatch statistics.
///
/// A returned ID is counted as correct if `dist(query, base[id]) <= d_k + tol`,
/// where `d_k` is the distance from `query` to the `k`-th ground-truth neighbor.
///
/// # Errors
///
/// Returns [`BenchError::InvalidConfig`] on mismatched lengths, `k == 0`,
/// empty ground-truth rows, or out-of-bounds ground-truth IDs.
pub fn compute_recall_tie_aware(
    results: &[Vec<(VectorId, f32)>],
    queries: &[Vec<f32>],
    groundtruth: &[Vec<VectorId>],
    base: &[Vec<f32>],
    metric: DistanceKind,
    k: usize,
    tol: f32,
) -> Result<(f64, TieCheckStats), BenchError> {
    if k == 0 {
        return Err(BenchError::InvalidConfig(
            "k must be greater than 0 for recall@k".into(),
        ));
    }
    if results.len() != groundtruth.len() || results.len() != queries.len() {
        return Err(BenchError::InvalidConfig(format!(
            "counts do not match: results={}, queries={}, groundtruth={}",
            results.len(),
            queries.len(),
            groundtruth.len()
        )));
    }
    if results.is_empty() {
        return Err(BenchError::InvalidConfig("no queries to evaluate".into()));
    }

    let kernel = resolve_kernel(metric);
    let mut total_recall = 0.0_f64;
    let mut queries_with_exact_mismatch = 0usize;
    let mut non_tie_mismatches = 0usize;

    for (q_idx, ((pred, gt), query)) in results
        .iter()
        .zip(groundtruth.iter())
        .zip(queries.iter())
        .enumerate()
    {
        let gt_k = &gt[..k.min(gt.len())];
        if gt_k.is_empty() {
            return Err(BenchError::InvalidConfig(format!(
                "ground-truth row {q_idx} is empty"
            )));
        }
        let kth_id = gt_k[gt_k.len() - 1].0 as usize;
        let kth_vec = base.get(kth_id).ok_or_else(|| {
            BenchError::InvalidConfig(format!(
                "ground-truth ID {kth_id} at query {q_idx} out of bounds (base len={})",
                base.len()
            ))
        })?;
        let kth_dist = kernel(query, kth_vec);

        let mut tie_hits = 0usize;
        let mut query_had_mismatch = false;

        for &(pred_id, _) in pred.iter().take(k) {
            let is_exact_id = gt_k.contains(&pred_id);
            if !is_exact_id {
                query_had_mismatch = true;
            }
            if is_exact_id {
                tie_hits += 1;
            } else {
                let pred_vec = base.get(pred_id.0 as usize).ok_or_else(|| {
                    BenchError::InvalidConfig(format!(
                        "predicted ID {} at query {q_idx} out of bounds (base len={})",
                        pred_id.0,
                        base.len()
                    ))
                })?;
                let d = kernel(query, pred_vec);
                if d <= kth_dist + tol {
                    tie_hits += 1;
                } else {
                    non_tie_mismatches += 1;
                }
            }
        }

        if query_had_mismatch {
            queries_with_exact_mismatch += 1;
        }
        total_recall += tie_hits as f64 / gt_k.len() as f64;
    }

    Ok((
        total_recall / results.len() as f64,
        TieCheckStats {
            total_queries: results.len(),
            queries_with_exact_mismatch,
            non_tie_mismatches,
        },
    ))
}

/// Run the benchmark harness against any [`SearchIndex`].
///
/// 1. Executes a single-threaded warmup pass.
/// 2. Executes a single-threaded one-query-at-a-time pass, recording per-query
///    nanosecond timings in an HDR histogram to compute `p50`/`p95`/`p99` (µs).
/// 3. Computes `recall@k` and `recall_tie_aware` if `groundtruth` is provided and `index.metric() == DistanceKind::L2`.
/// 4. Executes a separate multi-threaded pass using Rayon to measure QPS.
pub fn run_benchmark(
    index: &dyn SearchIndex,
    queries: &[Vec<f32>],
    groundtruth: Option<&[Vec<VectorId>]>,
    base: Option<&[Vec<f32>]>,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult, BenchError> {
    if queries.is_empty() {
        return Err(BenchError::InvalidConfig("queries cannot be empty".into()));
    }
    if config.k == 0 {
        return Err(BenchError::InvalidConfig("k must be > 0".into()));
    }

    // 1. Warmup pass (single-threaded)
    let warmup_n = config.warmup_queries.min(queries.len());
    for q in &queries[..warmup_n] {
        let _ = index.search(q, config.k)?;
    }

    // 2. Single-threaded per-query latency pass
    let mut hist = Histogram::<u64>::new(3).map_err(|e| {
        BenchError::InvalidConfig(format!("failed to initialize HDR histogram: {e}"))
    })?;
    let mut all_results = Vec::with_capacity(queries.len());
    let single_start = Instant::now();
    for q in queries {
        let t0 = Instant::now();
        let res = index.search(q, config.k)?;
        let elapsed_ns = t0.elapsed().as_nanos().max(1) as u64;
        hist.record(elapsed_ns)
            .map_err(|e| BenchError::InvalidConfig(format!("histogram record failed: {e}")))?;
        all_results.push(res);
    }
    let single_thread_total_secs = single_start.elapsed().as_secs_f64();

    let p50_us = hist.value_at_quantile(0.50) as f64 / 1_000.0;
    let p95_us = hist.value_at_quantile(0.95) as f64 / 1_000.0;
    let p99_us = hist.value_at_quantile(0.99) as f64 / 1_000.0;

    // 3. Recall@k (L2 only)
    let recall_at_k = match (index.metric(), groundtruth) {
        (DistanceKind::L2, Some(gt)) => Some(compute_recall_at_k(&all_results, gt, config.k)?),
        _ => None,
    };
    let recall_tie_aware = match (index.metric(), groundtruth, base) {
        (DistanceKind::L2, Some(gt), Some(base_vecs)) => Some(
            compute_recall_tie_aware(
                &all_results,
                queries,
                gt,
                base_vecs,
                DistanceKind::L2,
                config.k,
                0.0,
            )?
            .0,
        ),
        _ => None,
    };

    // 4. Multi-threaded QPS pass
    let pool = if config.qps_threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.qps_threads)
            .build()
            .map_err(|e| BenchError::InvalidConfig(format!("failed to build thread pool: {e}")))?
    } else {
        rayon::ThreadPoolBuilder::new()
            .build()
            .map_err(|e| BenchError::InvalidConfig(format!("failed to build thread pool: {e}")))?
    };
    let actual_threads = pool.current_num_threads();

    let target_dur = Duration::from_secs_f64(config.qps_duration_secs.max(0.01));
    let qps_start = Instant::now();
    let mut total_qps_queries: u64 = 0;
    pool.install(|| -> Result<(), BenchError> {
        loop {
            queries
                .par_iter()
                .try_for_each(|q| index.search(q, config.k).map(|_| ()))?;
            total_qps_queries += queries.len() as u64;
            if qps_start.elapsed() >= target_dur {
                break;
            }
        }
        Ok(())
    })?;
    let qps_elapsed = qps_start.elapsed().as_secs_f64();
    let qps = total_qps_queries as f64 / qps_elapsed.max(f64::EPSILON);
    let peak_rss_bytes = os_peak_rss_bytes();

    let (cpu_model, os) = collect_system_info();

    Ok(BenchmarkResult {
        git_commit: collect_git_commit(),
        rustc_version: collect_rustc_version(),
        cpu_model,
        os,
        dataset: config.dataset.clone(),
        metric: format!("{:?}", index.metric()),
        index: config.index_name.clone(),
        index_params: config.index_params.clone(),
        timestamp: collect_timestamp(),
        num_vectors: index.len(),
        num_queries: queries.len(),
        dim: index.dim(),
        k: config.k,
        recall_at_k,
        recall_tie_aware,
        p50_us,
        p95_us,
        p99_us,
        qps,
        qps_threads: actual_threads,
        build_time_secs: config.build_time_secs,
        single_thread_total_secs,
        index_memory_bytes: config.index_memory_bytes,
        peak_rss_bytes,
    })
}

/// Format a [`BenchmarkResult`] as a human-readable console table.
pub fn format_console_table(res: &BenchmarkResult) -> String {
    let recall_str = res
        .recall_at_k
        .map_or_else(|| "N/A".to_string(), |r| format!("{r:.4}"));
    let tie_str = res
        .recall_tie_aware
        .map_or_else(|| "N/A".to_string(), |r| format!("{r:.4}"));
    let rss_mb = res.peak_rss_bytes as f64 / (1024.0 * 1024.0);
    let mut out = String::new();
    out.push_str(&format!(
        "Dataset: {} | Metric: {} | Index: {} | N={} Q={} dim={} k={}\n",
        res.dataset, res.metric, res.index, res.num_vectors, res.num_queries, res.dim, res.k
    ));
    out.push_str(
        "+-------------+-------------+-----------+-----------+-----------+-----------+------------+--------------+--------------+\n",
    );
    out.push_str(&format!(
        "| {:<11} | {:<11} | {:>9} | {:>9} | {:>9} | {:>9} | {:>10} | {:>12} | {:>12} |\n",
        format!("recall@{}", res.k),
        "tie_recall",
        "p50 (us)",
        "p95 (us)",
        "p99 (us)",
        "QPS",
        "threads",
        "build (s)",
        "peak RSS(MB)"
    ));
    out.push_str(
        "+-------------+-------------+-----------+-----------+-----------+-----------+------------+--------------+--------------+\n",
    );
    out.push_str(&format!(
        "| {:<11} | {:<11} | {:>9.1} | {:>9.1} | {:>9.1} | {:>9.1} | {:>10} | {:>12.2} | {:>12.1} |\n",
        recall_str,
        tie_str,
        res.p50_us,
        res.p95_us,
        res.p99_us,
        res.qps,
        res.qps_threads,
        res.build_time_secs,
        rss_mb
    ));
    out.push_str(
        "+-------------+-------------+-----------+-----------+-----------+-----------+------------+--------------+--------------+",
    );
    out
}

/// Write a [`BenchmarkResult`] to a pretty-printed JSON file.
pub fn write_json(path: &Path, result: &BenchmarkResult) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let json = serde_json::to_string_pretty(result)
        .map_err(|e| BenchError::InvalidConfig(format!("failed to serialize JSON: {e}")))?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write a [`BenchmarkResult`] as a single-row CSV file (with header).
pub fn write_csv(path: &Path, result: &BenchmarkResult) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut writer = csv::Writer::from_path(path)
        .map_err(|e| BenchError::InvalidConfig(format!("failed to create CSV writer: {e}")))?;
    writer
        .write_record([
            "timestamp",
            "git_commit",
            "rustc_version",
            "cpu_model",
            "os",
            "dataset",
            "metric",
            "index",
            "num_vectors",
            "num_queries",
            "dim",
            "k",
            "recall_at_k",
            "recall_tie_aware",
            "p50_us",
            "p95_us",
            "p99_us",
            "qps",
            "qps_threads",
            "build_time_secs",
            "single_thread_total_secs",
            "index_memory_bytes",
            "peak_rss_bytes",
        ])
        .map_err(|e| BenchError::InvalidConfig(format!("CSV header error: {e}")))?;
    writer
        .write_record([
            result.timestamp.clone(),
            result.git_commit.clone(),
            result.rustc_version.clone(),
            result.cpu_model.clone(),
            result.os.clone(),
            result.dataset.clone(),
            result.metric.clone(),
            result.index.clone(),
            result.num_vectors.to_string(),
            result.num_queries.to_string(),
            result.dim.to_string(),
            result.k.to_string(),
            result
                .recall_at_k
                .map_or_else(String::new, |r| format!("{r:.6}")),
            result
                .recall_tie_aware
                .map_or_else(String::new, |r| format!("{r:.6}")),
            format!("{:.3}", result.p50_us),
            format!("{:.3}", result.p95_us),
            format!("{:.3}", result.p99_us),
            format!("{:.3}", result.qps),
            result.qps_threads.to_string(),
            format!("{:.3}", result.build_time_secs),
            format!("{:.3}", result.single_thread_total_secs),
            result.index_memory_bytes.to_string(),
            result.peak_rss_bytes.to_string(),
        ])
        .map_err(|e| BenchError::InvalidConfig(format!("CSV row error: {e}")))?;
    writer
        .flush()
        .map_err(|e| BenchError::InvalidConfig(format!("CSV flush error: {e}")))?;
    Ok(())
}

fn collect_git_commit() -> String {
    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    if hash == "unknown" {
        return hash;
    }
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .is_some_and(|o| !o.stdout.is_empty());
    if dirty {
        format!("{hash}-dirty")
    } else {
        hash
    }
}

fn collect_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

fn collect_system_info() -> (String, String) {
    let mut sys = System::new();
    sys.refresh_cpu_all();
    let cpu = sys
        .cpus()
        .first()
        .map(|c| c.brand().trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into());
    let os_name = System::name().unwrap_or_else(|| std::env::consts::OS.to_string());
    let os_ver = System::os_version().unwrap_or_default();
    let os = format!("{os_name} {os_ver}").trim().to_string();
    (cpu, os)
}

fn collect_timestamp() -> String {
    OffsetDateTime::now_utc()
        .format(&Rfc3339)
        .unwrap_or_else(|_| "unknown".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use vl_core::{CosineDistance, L2Distance};
    use vl_index::BruteForceIndex;

    #[test]
    fn recall_exact_and_partial() {
        let preds = vec![
            vec![(VectorId(1), 0.1), (VectorId(2), 0.2)],
            vec![(VectorId(3), 0.1), (VectorId(99), 0.5)],
        ];
        let gt = vec![
            vec![VectorId(1), VectorId(2), VectorId(5)],
            vec![VectorId(3), VectorId(4), VectorId(5)],
        ];
        let r = compute_recall_at_k(&preds, &gt, 2).unwrap();
        assert!((r - 0.75).abs() < 1e-9);
    }

    #[test]
    fn harness_runs_and_skips_gt_for_non_l2() {
        let mut idx_l2 = BruteForceIndex::new(2, Arc::new(L2Distance));
        idx_l2.insert(VectorId(0), vec![0.0, 0.0]).unwrap();
        idx_l2.insert(VectorId(1), vec![1.0, 1.0]).unwrap();

        let queries = vec![vec![0.1, 0.1], vec![0.9, 0.9]];
        let gt = vec![
            vec![VectorId(0), VectorId(1)],
            vec![VectorId(1), VectorId(0)],
        ];

        let cfg = BenchmarkConfig {
            dataset: "unit".into(),
            index_name: "brute_force".into(),
            index_params: BTreeMap::new(),
            k: 1,
            warmup_queries: 1,
            qps_threads: 2,
            build_time_secs: 0.01,
            index_memory_bytes: 1024,
            qps_duration_secs: 0.05,
        };

        let res_l2 = run_benchmark(&idx_l2, &queries, Some(&gt), None, &cfg).unwrap();
        assert_eq!(res_l2.recall_at_k, Some(1.0));
        assert!(res_l2.p50_us > 0.0);
        assert!(res_l2.qps > 0.0);
        assert_eq!(res_l2.index_memory_bytes, 1024);
        assert!(res_l2.peak_rss_bytes > 0);

        let mut idx_cos = BruteForceIndex::new(2, Arc::new(CosineDistance));
        idx_cos.insert(VectorId(0), vec![1.0, 0.0]).unwrap();
        idx_cos.insert(VectorId(1), vec![0.0, 1.0]).unwrap();
        let res_cos = run_benchmark(&idx_cos, &queries, Some(&gt), None, &cfg).unwrap();
        assert_eq!(res_cos.recall_at_k, None);
    }

    #[test]
    fn tie_aware_recall_counts_distance_ties() {
        let base = vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![-1.0, 0.0]];
        let queries = vec![vec![0.0, 0.0]];
        // Vectors 1 and 2 are both at squared L2 distance 1.0 from query [0, 0]
        let gt = vec![vec![VectorId(0), VectorId(1)]];
        let preds = vec![vec![(VectorId(0), 0.0), (VectorId(2), 1.0)]];

        let exact = compute_recall_at_k(&preds, &gt, 2).unwrap();
        assert!((exact - 0.5).abs() < 1e-9);

        let (tie_aware, stats) =
            compute_recall_tie_aware(&preds, &queries, &gt, &base, DistanceKind::L2, 2, 0.0)
                .unwrap();
        assert!((tie_aware - 1.0).abs() < 1e-9);
        assert_eq!(stats.queries_with_exact_mismatch, 1);
        assert_eq!(stats.non_tie_mismatches, 0);
    }
}
