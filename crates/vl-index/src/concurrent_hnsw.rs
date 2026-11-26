//! Concurrent HNSW index supporting parallel builds and simultaneous readers/writers.
//!
//! Design:
//! - Pre-allocated fixed-capacity arena (`max_elements` set at construction) so
//!   storage never reallocates while readers traverse the graph.
//! - Vectors stored as `Vec<AtomicU32>` (`f32::to_bits()`) and external IDs as
//!   `Vec<AtomicU64>` for 100% safe concurrent reads/writes without `unsafe`.
//! - Neighbor slots stored as `AtomicU32` (`u32::MAX` sentinel) with `AtomicU16`
//!   per-node-layer counts: readers are lock-free (`Acquire`/`Relaxed`), while
//!   writers synchronize per node via `Mutex<()>` (at most one node locked at a time,
//!   guaranteeing deadlock freedom).
//! - Global `entry_point` and `max_level` are guarded by a `RwLock`.

use std::cell::RefCell;
use std::cmp::Reverse;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use ordered_float::NotNan;
use rayon::prelude::*;
use vl_core::kernels::{resolve_kernel, DistanceKernelFn};
use vl_core::{CoreError, Distance, DistanceKind, VectorId};

use crate::hnsw::{HnswConfig, SearchScratch};
use crate::{IndexError, SearchIndex};

const EMPTY_NEIGHBOR: u32 = u32::MAX;

thread_local! {
    static CONCURRENT_TLS_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

struct UpperLayerStorage {
    neighbors: Box<[AtomicU32]>,
    counts: Box<[AtomicU16]>,
}

/// Concurrent HNSW index with fixed `max_elements` capacity.
pub struct ConcurrentHnswIndex {
    dim: usize,
    max_elements: usize,
    distance: Arc<dyn Distance>,
    metric: DistanceKind,
    kernel: DistanceKernelFn,
    config: HnswConfig,

    next_id: AtomicUsize,
    // Flat pre-allocated vector arena (`max_elements * dim`)
    vectors: Vec<AtomicU32>,
    ids: Vec<AtomicU64>,
    node_levels: Vec<AtomicU8>,

    // Layer 0 flat neighbor arena (`max_elements * m0`)
    l0_neighbors: Vec<AtomicU32>,
    l0_counts: Vec<AtomicU16>,

    // Upper layers (1..=level) allocated once per node when level > 0
    upper_layers: Vec<OnceLock<UpperLayerStorage>>,

    // Per-node writer lock
    node_locks: Vec<Mutex<()>>,

    // Global entry point and max_level
    entry_point: RwLock<Option<(u32, usize)>>,
}

impl ConcurrentHnswIndex {
    /// Create a new concurrent HNSW index pre-allocated for up to `max_elements` vectors.
    pub fn new(
        dim: usize,
        distance: Arc<dyn Distance>,
        config: HnswConfig,
        max_elements: usize,
    ) -> Self {
        let metric = distance.kind();
        let kernel_kind = match metric {
            DistanceKind::Cosine => DistanceKind::InnerProduct,
            other => other,
        };
        let kernel = resolve_kernel(kernel_kind);

        let vec_len = max_elements * dim;
        let mut vectors = Vec::with_capacity(vec_len);
        vectors.resize_with(vec_len, || AtomicU32::new(0));

        let mut ids = Vec::with_capacity(max_elements);
        ids.resize_with(max_elements, || AtomicU64::new(0));

        let mut node_levels = Vec::with_capacity(max_elements);
        node_levels.resize_with(max_elements, || AtomicU8::new(0));

        let l0_len = max_elements * config.m0;
        let mut l0_neighbors = Vec::with_capacity(l0_len);
        l0_neighbors.resize_with(l0_len, || AtomicU32::new(EMPTY_NEIGHBOR));

        let mut l0_counts = Vec::with_capacity(max_elements);
        l0_counts.resize_with(max_elements, || AtomicU16::new(0));

        let mut upper_layers = Vec::with_capacity(max_elements);
        upper_layers.resize_with(max_elements, OnceLock::new);

        let mut node_locks = Vec::with_capacity(max_elements);
        node_locks.resize_with(max_elements, || Mutex::new(()));

        Self {
            dim,
            max_elements,
            distance,
            metric,
            kernel,
            config,
            next_id: AtomicUsize::new(0),
            vectors,
            ids,
            node_levels,
            l0_neighbors,
            l0_counts,
            upper_layers,
            node_locks,
            entry_point: RwLock::new(None),
        }
    }

    /// Number of vectors currently inserted.
    pub fn len(&self) -> usize {
        self.next_id.load(Ordering::Acquire).min(self.max_elements)
    }

    /// True if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance metric.
    pub fn metric(&self) -> DistanceKind {
        self.metric
    }

    /// Maximum element capacity.
    pub fn max_elements(&self) -> usize {
        self.max_elements
    }

    /// Approximate heap and struct memory owned by this index, in bytes.
    pub fn memory_bytes(&self) -> usize {
        let upper_bytes: usize = self
            .upper_layers
            .iter()
            .filter_map(|cell| cell.get())
            .map(|u| {
                u.neighbors.len() * std::mem::size_of::<AtomicU32>()
                    + u.counts.len() * std::mem::size_of::<AtomicU16>()
            })
            .sum();

        std::mem::size_of::<Self>()
            + self.vectors.capacity() * std::mem::size_of::<AtomicU32>()
            + self.ids.capacity() * std::mem::size_of::<AtomicU64>()
            + self.node_levels.capacity() * std::mem::size_of::<AtomicU8>()
            + self.l0_neighbors.capacity() * std::mem::size_of::<AtomicU32>()
            + self.l0_counts.capacity() * std::mem::size_of::<AtomicU16>()
            + self.upper_layers.capacity() * std::mem::size_of::<OnceLock<UpperLayerStorage>>()
            + self.node_locks.capacity() * std::mem::size_of::<Mutex<()>>()
            + upper_bytes
    }

    /// Build the index in parallel using `rayon`.
    pub fn insert_batch_parallel(&self, items: &[(VectorId, Vec<f32>)]) -> Result<(), IndexError> {
        items
            .par_iter()
            .try_for_each(|(id, vec)| self.insert(*id, vec.clone()))
    }

    #[inline]
    fn read_vec_into(&self, node_id: u32, buf: &mut Vec<f32>) {
        buf.clear();
        let start = node_id as usize * self.dim;
        for a in &self.vectors[start..start + self.dim] {
            buf.push(f32::from_bits(a.load(Ordering::Relaxed)));
        }
    }

    #[inline]
    fn dist_to_node(&self, query: &[f32], node_id: u32, buf: &mut Vec<f32>) -> f32 {
        self.read_vec_into(node_id, buf);
        let raw = (self.kernel)(query, buf);
        if self.metric == DistanceKind::Cosine {
            (1.0 - raw).max(0.0)
        } else {
            raw
        }
    }

    #[inline]
    fn dist_between_nodes(
        &self,
        a: u32,
        b: u32,
        buf_a: &mut Vec<f32>,
        buf_b: &mut Vec<f32>,
    ) -> f32 {
        self.read_vec_into(a, buf_a);
        self.read_vec_into(b, buf_b);
        let raw = (self.kernel)(buf_a, buf_b);
        if self.metric == DistanceKind::Cosine {
            (1.0 - raw).max(0.0)
        } else {
            raw
        }
    }

    fn read_neighbors(&self, node_id: u32, layer: usize, out: &mut Vec<u32>) {
        out.clear();
        let idx = node_id as usize;
        if layer == 0 {
            let count = (self.l0_counts[idx].load(Ordering::Acquire) as usize).min(self.config.m0);
            let start = idx * self.config.m0;
            for slot in &self.l0_neighbors[start..start + count] {
                let nb = slot.load(Ordering::Relaxed);
                if nb != EMPTY_NEIGHBOR {
                    out.push(nb);
                }
            }
        } else if let Some(upper) = self.upper_layers[idx].get() {
            let count =
                (upper.counts[layer - 1].load(Ordering::Acquire) as usize).min(self.config.m);
            let start = (layer - 1) * self.config.m;
            for slot in &upper.neighbors[start..start + count] {
                let nb = slot.load(Ordering::Relaxed);
                if nb != EMPTY_NEIGHBOR {
                    out.push(nb);
                }
            }
        }
    }

    fn write_neighbors_locked(&self, node_id: u32, layer: usize, neighbors: &[u32]) {
        let idx = node_id as usize;
        if layer == 0 {
            let len = neighbors.len().min(self.config.m0);
            let start = idx * self.config.m0;
            for (i, &nb) in neighbors[..len].iter().enumerate() {
                self.l0_neighbors[start + i].store(nb, Ordering::Relaxed);
            }
            self.l0_counts[idx].store(len as u16, Ordering::Release);
        } else if let Some(upper) = self.upper_layers[idx].get() {
            let len = neighbors.len().min(self.config.m);
            let start = (layer - 1) * self.config.m;
            for (i, &nb) in neighbors[..len].iter().enumerate() {
                upper.neighbors[start + i].store(nb, Ordering::Relaxed);
            }
            upper.counts[layer - 1].store(len as u16, Ordering::Release);
        }
    }

    fn sample_level(&self, node_id: u32) -> u8 {
        // Mix node_id + seed + thread/atomic entropy for non-deterministic parallel level assignment
        let mut z = self
            .config
            .seed
            .wrapping_add((node_id as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let v = (z >> 11) + 1;
        let u = (v as f64) * (1.0 / ((1u64 << 53) as f64));
        let lvl = (-u.ln() * self.config.ml).floor() as usize;
        lvl.min(u8::MAX as usize) as u8
    }

    /// Insert a vector concurrently using `&self`.
    pub fn insert(&self, id: VectorId, mut vector: Vec<f32>) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::Core(CoreError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            }));
        }
        self.distance
            .validate_vector(&vector)
            .map_err(IndexError::Core)?;
        if self.metric == DistanceKind::Cosine {
            normalize_in_place(&mut vector);
        }

        let slot = self.next_id.fetch_add(1, Ordering::AcqRel);
        if slot >= self.max_elements {
            self.next_id.fetch_sub(1, Ordering::AcqRel);
            return Err(IndexError::CapacityExceeded);
        }
        let node_id = slot as u32;
        let level = self.sample_level(node_id);

        // Write vector, id, level, and upper storage before linking into graph
        let v_start = slot * self.dim;
        for (i, &val) in vector.iter().enumerate() {
            self.vectors[v_start + i].store(val.to_bits(), Ordering::Relaxed);
        }
        self.ids[slot].store(id.0, Ordering::Release);
        self.node_levels[slot].store(level, Ordering::Release);

        if level > 0 {
            let num_upper = level as usize;
            let mut nbs = Vec::with_capacity(num_upper * self.config.m);
            nbs.resize_with(num_upper * self.config.m, || AtomicU32::new(EMPTY_NEIGHBOR));
            let mut cnts = Vec::with_capacity(num_upper);
            cnts.resize_with(num_upper, || AtomicU16::new(0));
            let _ = self.upper_layers[slot].set(UpperLayerStorage {
                neighbors: nbs.into_boxed_slice(),
                counts: cnts.into_boxed_slice(),
            });
        }

        let ep_snapshot = *self.entry_point.read().unwrap_or_else(|e| e.into_inner());
        let Some((curr_obj, max_lvl)) = ep_snapshot else {
            let mut ep_guard = self.entry_point.write().unwrap_or_else(|e| e.into_inner());
            if ep_guard.is_none() {
                *ep_guard = Some((node_id, level as usize));
                return Ok(());
            }
            let (ep, ml) = ep_guard.unwrap();
            drop(ep_guard);
            return self.insert_into_existing(node_id, level as usize, &vector, ep, ml);
        };

        self.insert_into_existing(node_id, level as usize, &vector, curr_obj, max_lvl)
    }

    fn insert_into_existing(
        &self,
        node_id: u32,
        level: usize,
        q_vec: &[f32],
        mut curr_obj: u32,
        max_lvl: usize,
    ) -> Result<(), IndexError> {
        let mut buf = Vec::with_capacity(self.dim);
        let mut nb_buf = Vec::new();
        let mut scratch = SearchScratch::new();

        if max_lvl > level {
            let mut curr_dist = self.dist_to_node(q_vec, curr_obj, &mut buf);
            for lc in (level + 1..=max_lvl).rev() {
                curr_obj = self.greedy_search_layer(
                    q_vec,
                    curr_obj,
                    &mut curr_dist,
                    lc,
                    &mut buf,
                    &mut nb_buf,
                );
            }
        }

        let top_connect = level.min(max_lvl);
        let num_nodes = self.len();

        for lc in (0..=top_connect).rev() {
            let candidates = self.search_layer_internal(
                q_vec,
                curr_obj,
                self.config.ef_construction,
                lc,
                num_nodes,
                &mut scratch,
                &mut buf,
                &mut nb_buf,
            );
            if let Some(&(_, closest)) = candidates.first() {
                curr_obj = closest;
            }
            let m_max = if lc == 0 {
                self.config.m0
            } else {
                self.config.m
            };
            let selected = self.select_neighbors_heuristic(&candidates, m_max);

            {
                let _guard = self.node_locks[node_id as usize]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                self.write_neighbors_locked(node_id, lc, &selected);
            }

            let mut existing_nbs = Vec::with_capacity(m_max + 1);
            let mut nb_vec = Vec::with_capacity(self.dim);
            for &nb in &selected {
                let _guard = self.node_locks[nb as usize]
                    .lock()
                    .unwrap_or_else(|e| e.into_inner());
                self.read_neighbors(nb, lc, &mut existing_nbs);
                if existing_nbs.contains(&node_id) {
                    continue;
                }
                if existing_nbs.len() < m_max {
                    existing_nbs.push(node_id);
                    self.write_neighbors_locked(nb, lc, &existing_nbs);
                } else {
                    self.read_vec_into(nb, &mut nb_vec);
                    let mut combined: Vec<(NotNan<f32>, u32)> =
                        Vec::with_capacity(existing_nbs.len() + 1);
                    for &ex in &existing_nbs {
                        let d = self.dist_to_node(&nb_vec, ex, &mut buf);
                        combined.push((to_not_nan(d), ex));
                    }
                    let d_new = self.dist_to_node(&nb_vec, node_id, &mut buf);
                    combined.push((to_not_nan(d_new), node_id));
                    combined.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                    let pruned = self.select_neighbors_heuristic(&combined, m_max);
                    self.write_neighbors_locked(nb, lc, &pruned);
                }
            }
        }

        if level > max_lvl {
            let mut ep_guard = self.entry_point.write().unwrap_or_else(|e| e.into_inner());
            let cur_max = ep_guard.map_or(0, |(_, ml)| ml);
            if level > cur_max || ep_guard.is_none() {
                *ep_guard = Some((node_id, level));
            }
        }

        Ok(())
    }

    fn greedy_search_layer(
        &self,
        query: &[f32],
        mut curr_obj: u32,
        curr_dist: &mut f32,
        layer: usize,
        buf: &mut Vec<f32>,
        nb_buf: &mut Vec<u32>,
    ) -> u32 {
        loop {
            let mut changed = false;
            self.read_neighbors(curr_obj, layer, nb_buf);
            for &nb in nb_buf.iter() {
                let d = self.dist_to_node(query, nb, buf);
                if d < *curr_dist {
                    *curr_dist = d;
                    curr_obj = nb;
                    changed = true;
                }
            }
            if !changed {
                return curr_obj;
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn search_layer_internal(
        &self,
        query: &[f32],
        entry_id: u32,
        ef: usize,
        layer: usize,
        num_nodes: usize,
        scratch: &mut SearchScratch,
        buf: &mut Vec<f32>,
        nb_buf: &mut Vec<u32>,
    ) -> Vec<(NotNan<f32>, u32)> {
        scratch.prepare(num_nodes.max(self.max_elements));
        scratch.mark_visited(entry_id);

        let d_ep = to_not_nan(self.dist_to_node(query, entry_id, buf));
        scratch.candidates.push(Reverse((d_ep, entry_id)));
        scratch.results.push((d_ep, entry_id));

        while let Some(Reverse((c_dist, c_id))) = scratch.candidates.pop() {
            let f_dist = scratch.results.peek().map_or(d_ep, |&(d, _)| d);
            if c_dist > f_dist && scratch.results.len() >= ef {
                break;
            }
            self.read_neighbors(c_id, layer, nb_buf);
            for &nb in nb_buf.iter() {
                if (nb as usize) < self.max_elements && scratch.mark_visited(nb) {
                    let d = to_not_nan(self.dist_to_node(query, nb, buf));
                    let worst = scratch.results.peek().map_or(d, |&(w, _)| w);
                    if d < worst || scratch.results.len() < ef {
                        scratch.candidates.push(Reverse((d, nb)));
                        scratch.results.push((d, nb));
                        if scratch.results.len() > ef {
                            scratch.results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<(NotNan<f32>, u32)> = scratch.results.drain().collect();
        out.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        out
    }

    fn select_neighbors_heuristic(
        &self,
        candidates: &[(NotNan<f32>, u32)],
        m_max: usize,
    ) -> Vec<u32> {
        let mut selected: Vec<(NotNan<f32>, u32)> = Vec::with_capacity(m_max);
        let mut discarded: Vec<u32> = Vec::new();
        let mut buf_a = Vec::with_capacity(self.dim);
        let mut buf_b = Vec::with_capacity(self.dim);

        for &(d_qe, e) in candidates {
            if selected.len() >= m_max {
                break;
            }
            let mut closer_to_query = true;
            for &(_, r) in &selected {
                let d_er = self.dist_between_nodes(e, r, &mut buf_a, &mut buf_b);
                if d_er <= d_qe.into_inner() {
                    closer_to_query = false;
                    break;
                }
            }
            if closer_to_query {
                selected.push((d_qe, e));
            } else {
                discarded.push(e);
            }
        }

        let mut out: Vec<u32> = selected.into_iter().map(|(_, id)| id).collect();
        for e in discarded {
            if out.len() >= m_max {
                break;
            }
            out.push(e);
        }
        out
    }

    /// Search for the `k` nearest neighbors using `config.ef_search`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search for the `k` nearest neighbors with an explicit `ef_search`.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(VectorId, f32)>, IndexError> {
        let ep_snapshot = *self.entry_point.read().unwrap_or_else(|e| e.into_inner());
        let Some((mut curr_obj, max_lvl)) = ep_snapshot else {
            return Err(IndexError::Empty);
        };
        if query.len() != self.dim {
            return Err(IndexError::Core(CoreError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            }));
        }
        self.distance
            .validate_vector(query)
            .map_err(IndexError::Core)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        CONCURRENT_TLS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let mut taken_buf = if self.metric == DistanceKind::Cosine {
                let mut b = std::mem::take(&mut scratch.query_buf);
                b.clear();
                b.extend_from_slice(query);
                normalize_in_place(&mut b);
                Some(b)
            } else {
                None
            };
            let q_slice: &[f32] = taken_buf.as_deref().unwrap_or(query);

            let mut buf = Vec::with_capacity(self.dim);
            let mut nb_buf = Vec::new();
            let mut curr_dist = self.dist_to_node(q_slice, curr_obj, &mut buf);
            for lc in (1..=max_lvl).rev() {
                curr_obj = self.greedy_search_layer(
                    q_slice,
                    curr_obj,
                    &mut curr_dist,
                    lc,
                    &mut buf,
                    &mut nb_buf,
                );
            }

            let ef = ef_search.max(k);
            let layer0_res = self.search_layer_internal(
                q_slice,
                curr_obj,
                ef,
                0,
                self.len(),
                &mut scratch,
                &mut buf,
                &mut nb_buf,
            );
            if let Some(b) = taken_buf.take() {
                scratch.query_buf = b;
            }

            let mut out: Vec<(VectorId, f32)> = layer0_res
                .into_iter()
                .take(k)
                .map(|(d, nid)| {
                    let ext_id = VectorId(self.ids[nid as usize].load(Ordering::Acquire));
                    (ext_id, d.into_inner())
                })
                .collect();
            out.sort_unstable_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            Ok(out)
        })
    }
}

impl SearchIndex for ConcurrentHnswIndex {
    fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
        ConcurrentHnswIndex::insert(self, id, vector)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        ConcurrentHnswIndex::search(self, query, k)
    }

    fn len(&self) -> usize {
        ConcurrentHnswIndex::len(self)
    }

    fn dim(&self) -> usize {
        ConcurrentHnswIndex::dim(self)
    }

    fn metric(&self) -> DistanceKind {
        ConcurrentHnswIndex::metric(self)
    }
}

#[inline]
fn to_not_nan(val: f32) -> NotNan<f32> {
    NotNan::new(val).unwrap_or_else(|_| NotNan::new(f32::MAX).unwrap_or_default())
}

#[inline]
fn normalize_in_place(v: &mut [f32]) {
    let norm_sq: f32 = v.iter().map(|&x| x * x).sum();
    let inv_norm = 1.0 / norm_sq.sqrt();
    for x in v.iter_mut() {
        *x *= inv_norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BruteForceIndex, HnswIndex};
    use std::collections::{HashSet, VecDeque};
    use std::sync::atomic::AtomicBool;
    use vl_core::L2Distance;

    fn make_random_vecs(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        state = state
                            .wrapping_mul(6_364_136_223_846_793_005)
                            .wrapping_add(1_442_695_040_888_963_407);
                        ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn concurrent_stress_test_20_loops() {
        let dim = 8;
        let n_items = 200usize;
        let cfg = HnswConfig::new(16, 64, 32, 42);

        for iter in 0..20 {
            let idx = Arc::new(ConcurrentHnswIndex::new(
                dim,
                Arc::new(L2Distance),
                cfg,
                n_items,
            ));
            let vecs = make_random_vecs(n_items, dim, 100 + iter as u64);
            let done = Arc::new(AtomicBool::new(false));

            std::thread::scope(|s| {
                // 4 writer threads
                let chunk = n_items / 4;
                for w in 0..4 {
                    let idx_w = Arc::clone(&idx);
                    let slice = &vecs[w * chunk..(w + 1) * chunk];
                    let base_id = w * chunk;
                    s.spawn(move || {
                        for (i, v) in slice.iter().enumerate() {
                            idx_w
                                .insert(VectorId((base_id + i) as u64), v.clone())
                                .unwrap();
                        }
                    });
                }

                // 4 reader threads searching continuously
                for v in vecs.iter().take(4) {
                    let idx_r = Arc::clone(&idx);
                    let done_r = Arc::clone(&done);
                    let q = v.clone();
                    s.spawn(move || {
                        while !done_r.load(Ordering::Relaxed) {
                            match idx_r.search(&q, 5) {
                                Ok(res) => {
                                    for w in res.windows(2) {
                                        assert!(w[0].1 <= w[1].1);
                                    }
                                    for (id, _) in res {
                                        assert!(id.0 < n_items as u64);
                                    }
                                }
                                Err(IndexError::Empty) => {}
                                Err(e) => panic!("unexpected search error: {e}"),
                            }
                        }
                    });
                }

                // Wait briefly for writers, then signal readers
                std::thread::sleep(std::time::Duration::from_millis(5));
                while idx.len() < n_items {
                    std::thread::yield_now();
                }
                done.store(true, Ordering::Release);
            });

            assert_eq!(idx.len(), n_items);

            // Post-join graph invariants: capacity, no self-loops, no duplicates, layer-0 reachability
            let mut nb_buf = Vec::new();
            for u in 0..n_items as u32 {
                let lvl = idx.node_levels[u as usize].load(Ordering::Relaxed) as usize;
                for l in 0..=lvl {
                    idx.read_neighbors(u, l, &mut nb_buf);
                    let cap = if l == 0 { cfg.m0 } else { cfg.m };
                    assert!(nb_buf.len() <= cap);
                    let mut seen = HashSet::new();
                    for &nb in &nb_buf {
                        assert_ne!(nb, u);
                        assert!(seen.insert(nb));
                    }
                }
            }

            let ep = idx.entry_point.read().unwrap().unwrap().0;
            let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n_items];
            for u in 0..n_items as u32 {
                idx.read_neighbors(u, 0, &mut nb_buf);
                for &nb in &nb_buf {
                    adj[u as usize].push(nb);
                    adj[nb as usize].push(u);
                }
            }
            let mut visited = vec![false; n_items];
            let mut queue = VecDeque::new();
            visited[ep as usize] = true;
            queue.push_back(ep);
            let mut reached = 0usize;
            while let Some(u) = queue.pop_front() {
                reached += 1;
                for &nb in &adj[u as usize] {
                    if !visited[nb as usize] {
                        visited[nb as usize] = true;
                        queue.push_back(nb);
                    }
                }
            }
            assert_eq!(reached, n_items, "iter {iter}: reached {reached}/{n_items}");
        }
    }

    #[test]
    fn concurrent_siftsmall_recall_within_tolerance() {
        let base_path = std::path::Path::new("../../data/siftsmall/siftsmall_base.fvecs");
        let query_path = std::path::Path::new("../../data/siftsmall/siftsmall_query.fvecs");
        if !base_path.exists() || !query_path.exists() {
            return;
        }

        fn read_fvecs_local(path: &std::path::Path) -> Vec<Vec<f32>> {
            use std::io::Read;
            let bytes = std::fs::read(path).unwrap();
            let mut cur = std::io::Cursor::new(bytes);
            let mut out = Vec::new();
            let mut hdr = [0u8; 4];
            while cur.read_exact(&mut hdr).is_ok() {
                let d = i32::from_le_bytes(hdr) as usize;
                let mut v = Vec::with_capacity(d);
                for _ in 0..d {
                    let mut fb = [0u8; 4];
                    cur.read_exact(&mut fb).unwrap();
                    v.push(f32::from_le_bytes(fb));
                }
                out.push(v);
            }
            out
        }

        let base = read_fvecs_local(base_path);
        let queries = read_fvecs_local(query_path);
        let dim = base[0].len();
        let cfg = HnswConfig::new(16, 200, 100, 42);

        let mut bf = BruteForceIndex::new(dim, Arc::new(L2Distance));
        let mut st = HnswIndex::new(dim, Arc::new(L2Distance), cfg);
        let conc = ConcurrentHnswIndex::new(dim, Arc::new(L2Distance), cfg, base.len());

        let items: Vec<(VectorId, Vec<f32>)> = base
            .iter()
            .enumerate()
            .map(|(i, v)| (VectorId(i as u64), v.clone()))
            .collect();

        for (id, v) in &items {
            bf.insert(*id, v.clone()).unwrap();
            st.insert(*id, v.clone()).unwrap();
        }
        conc.insert_batch_parallel(&items).unwrap();

        let k = 10;
        let mut st_recall = 0.0;
        let mut conc_recall = 0.0;
        for q in &queries {
            let gt: HashSet<VectorId> = bf
                .search(q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let st_got = st.search_with_ef(q, k, 100).unwrap();
            let conc_got = conc.search_with_ef(q, k, 100).unwrap();
            st_recall += st_got.iter().filter(|(id, _)| gt.contains(id)).count() as f64 / k as f64;
            conc_recall +=
                conc_got.iter().filter(|(id, _)| gt.contains(id)).count() as f64 / k as f64;
        }
        st_recall /= queries.len() as f64;
        conc_recall /= queries.len() as f64;
        println!("siftsmall ST recall@10={st_recall:.4}, Concurrent recall@10={conc_recall:.4}");
        assert!(
            (st_recall - conc_recall).abs() <= 0.01,
            "concurrent recall {conc_recall:.4} differs from ST {st_recall:.4} by > 0.01"
        );
    }
}
