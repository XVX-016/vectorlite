//! Single-threaded Hierarchical Navigable Small World (HNSW) index.
//!
//! Implements the Malkov & Yashunin (2018) algorithms:
//! - `INSERT` (Algorithm 1)
//! - `SEARCH-LAYER` (Algorithm 2)
//! - `SELECT-NEIGHBORS-HEURISTIC` (Algorithm 4)
//! - `K-NN-SEARCH` (Algorithm 5)
//!
//! Vectors are stored in one contiguous `Vec<f32>` arena, internal node IDs are
//! `u32`, and neighbor lists are stored in flat fixed-capacity arrays (no `Vec<Vec<>>`).

use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::Arc;

use ordered_float::NotNan;
use vl_core::kernels::{resolve_kernel, DistanceKernelFn};
use vl_core::{CoreError, Distance, DistanceKind, VectorId};

use crate::{IndexError, SearchIndex};

/// Configuration parameters for [`HnswIndex`].
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Maximum number of bi-directional links created per node on upper layers (`> 0`).
    pub m: usize,
    /// Maximum number of bi-directional links created per node on layer 0 (`2 * m`).
    pub m0: usize,
    /// Dynamic candidate list size during index construction.
    pub ef_construction: usize,
    /// Default dynamic candidate list size during search (`SearchIndex::search`).
    pub ef_search: usize,
    /// Deterministic RNG seed for layer assignment.
    pub seed: u64,
    /// Level normalization factor `1.0 / ln(m)`.
    pub ml: f64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self::new(16, 200, 50, 42)
    }
}

impl HnswConfig {
    /// Create a new [`HnswConfig`] with `m0 = 2 * m` and `ml = 1.0 / (m as f64).ln()`.
    ///
    /// # Panics
    ///
    /// Panics if `m < 2` (since `ln(1) == 0`). Use [`HnswConfig::try_new`] for fallible construction.
    pub fn new(m: usize, ef_construction: usize, ef_search: usize, seed: u64) -> Self {
        let m_clamped = m.max(2);
        Self {
            m: m_clamped,
            m0: m_clamped * 2,
            ef_construction: ef_construction.max(1),
            ef_search: ef_search.max(1),
            seed,
            ml: 1.0 / (m_clamped as f64).ln(),
        }
    }
}

/// Trace of nodes visited per layer during [`HnswIndex::search_with_trace`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchTrace {
    /// `visited_per_layer[layer]` lists the external [`VectorId`]s visited at `layer`
    /// (`0` = bottom layer, `visited_per_layer.len() - 1` = top layer).
    pub visited_per_layer: Vec<Vec<VectorId>>,
}

/// Reusable per-thread scratch space for HNSW search and construction.
///
/// Avoids per-query heap allocations and `HashSet` overhead by using a
/// generation-stamped `visited` array and reusable binary heaps.
#[derive(Debug)]
pub struct SearchScratch {
    pub(crate) visited: Vec<u32>,
    pub(crate) gen: u32,
    pub(crate) candidates: BinaryHeap<Reverse<(NotNan<f32>, u32)>>,
    pub(crate) results: BinaryHeap<(NotNan<f32>, u32)>,
    pub(crate) query_buf: Vec<f32>,
}

impl Default for SearchScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchScratch {
    /// Create an empty [`SearchScratch`].
    pub fn new() -> Self {
        Self {
            visited: Vec::new(),
            gen: 0,
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            query_buf: Vec::new(),
        }
    }

    pub(crate) fn prepare(&mut self, num_nodes: usize) {
        if self.visited.len() < num_nodes {
            self.visited.resize(num_nodes, 0);
        }
        if self.gen == u32::MAX {
            self.visited.fill(0);
            self.gen = 1;
        } else {
            self.gen += 1;
        }
        self.candidates.clear();
        self.results.clear();
    }

    #[inline]
    pub(crate) fn mark_visited(&mut self, node_id: u32) -> bool {
        let idx = node_id as usize;
        if self.visited[idx] == self.gen {
            false
        } else {
            self.visited[idx] = self.gen;
            true
        }
    }
}

thread_local! {
    static TLS_SCRATCH: RefCell<SearchScratch> = RefCell::new(SearchScratch::new());
}

/// Pure-Rust SplitMix64 PRNG for deterministic level generation.
#[derive(Debug, Clone)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Returns a uniform `f64` in `(0.0, 1.0]`.
    fn next_f64_open0(&mut self) -> f64 {
        // Use top 53 bits for uniform mantissa in [1, 2^53] / 2^53.
        let v = (self.next_u64() >> 11) + 1;
        (v as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Single-threaded HNSW approximate nearest-neighbour index.
pub struct HnswIndex {
    dim: usize,
    distance: Arc<dyn Distance>,
    metric: DistanceKind,
    kernel: DistanceKernelFn,
    config: HnswConfig,
    rng: SplitMix64,

    // Contiguous vector arena: `vectors[node_id * dim .. (node_id + 1) * dim]`
    vectors: Vec<f32>,
    ids: Vec<VectorId>,
    node_levels: Vec<u8>,

    // Layer 0 flat neighbor arena: stride `config.m0` per node
    l0_neighbors: Vec<u32>,
    l0_counts: Vec<u16>,

    // Upper layers (1..=level) flat neighbor arena: stride `config.m` per layer slot
    upper_neighbors: Vec<u32>,
    upper_counts: Vec<u16>,
    upper_offsets: Vec<u32>,
    upper_count_offsets: Vec<u32>,

    entry_point: Option<u32>,
    max_level: usize,

    // Scratch reused during single-threaded inserts
    insert_scratch: SearchScratch,
}

impl HnswIndex {
    /// Create a new empty [`HnswIndex`] with the given dimension, distance metric, and config.
    pub fn new(dim: usize, distance: Arc<dyn Distance>, config: HnswConfig) -> Self {
        let metric = distance.kind();
        let kernel_kind = match metric {
            DistanceKind::Cosine => DistanceKind::InnerProduct,
            other => other,
        };
        let kernel = resolve_kernel(kernel_kind);
        Self {
            dim,
            distance,
            metric,
            kernel,
            config,
            rng: SplitMix64::new(config.seed),
            vectors: Vec::new(),
            ids: Vec::new(),
            node_levels: Vec::new(),
            l0_neighbors: Vec::new(),
            l0_counts: Vec::new(),
            upper_neighbors: Vec::new(),
            upper_counts: Vec::new(),
            upper_offsets: Vec::new(),
            upper_count_offsets: Vec::new(),
            entry_point: None,
            max_level: 0,
            insert_scratch: SearchScratch::new(),
        }
    }

    /// Create a new [`HnswIndex`] with default [`HnswConfig`].
    pub fn with_defaults(dim: usize, distance: Arc<dyn Distance>) -> Self {
        Self::new(dim, distance, HnswConfig::default())
    }

    /// Return the configuration used by this index.
    pub fn config(&self) -> HnswConfig {
        self.config
    }

    /// The number of vectors currently stored.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// True if no vectors have been inserted yet.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// The expected dimensionality of vectors in this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The distance metric used by this index.
    pub fn metric(&self) -> DistanceKind {
        self.metric
    }

    /// Maximum layer currently present in the graph.
    pub fn max_level(&self) -> usize {
        self.max_level
    }

    /// Entry-point external [`VectorId`], if the index is non-empty.
    pub fn entry_point_id(&self) -> Option<VectorId> {
        self.entry_point.map(|id| self.ids[id as usize])
    }

    /// Approximate heap and struct memory owned by this index, in bytes.
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.vectors.capacity() * std::mem::size_of::<f32>()
            + self.ids.capacity() * std::mem::size_of::<VectorId>()
            + self.node_levels.capacity() * std::mem::size_of::<u8>()
            + self.l0_neighbors.capacity() * std::mem::size_of::<u32>()
            + self.l0_counts.capacity() * std::mem::size_of::<u16>()
            + self.upper_neighbors.capacity() * std::mem::size_of::<u32>()
            + self.upper_counts.capacity() * std::mem::size_of::<u16>()
            + self.upper_offsets.capacity() * std::mem::size_of::<u32>()
            + self.upper_count_offsets.capacity() * std::mem::size_of::<u32>()
            + self.insert_scratch.visited.capacity() * std::mem::size_of::<u32>()
            + self.insert_scratch.query_buf.capacity() * std::mem::size_of::<f32>()
    }

    /// Return the slice of neighbor external [`VectorId`]s for `node_idx` at `layer`.
    #[cfg(test)]
    pub(crate) fn node_neighbors_at_layer(&self, node_idx: u32, layer: usize) -> &[u32] {
        self.neighbors_slice(node_idx, layer)
    }

    #[inline]
    fn vec_slice(&self, node_id: u32) -> &[f32] {
        let start = node_id as usize * self.dim;
        &self.vectors[start..start + self.dim]
    }

    #[inline]
    fn dist_unchecked(&self, a: &[f32], b: &[f32]) -> f32 {
        let raw = (self.kernel)(a, b);
        if self.metric == DistanceKind::Cosine {
            (1.0 - raw).max(0.0)
        } else {
            raw
        }
    }

    #[inline]
    fn max_neighbors(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m0
        } else {
            self.config.m
        }
    }

    #[inline]
    fn neighbors_slice(&self, node_id: u32, layer: usize) -> &[u32] {
        let idx = node_id as usize;
        if layer == 0 {
            let count = self.l0_counts[idx] as usize;
            let start = idx * self.config.m0;
            &self.l0_neighbors[start..start + count]
        } else {
            let count_idx = self.upper_count_offsets[idx] as usize + (layer - 1);
            let count = self.upper_counts[count_idx] as usize;
            let start = self.upper_offsets[idx] as usize + (layer - 1) * self.config.m;
            &self.upper_neighbors[start..start + count]
        }
    }

    fn set_neighbors(&mut self, node_id: u32, layer: usize, new_neighbors: &[u32]) {
        let idx = node_id as usize;
        let cap = self.max_neighbors(layer);
        debug_assert!(new_neighbors.len() <= cap);
        let len = new_neighbors.len().min(cap);
        if layer == 0 {
            let start = idx * self.config.m0;
            self.l0_neighbors[start..start + len].copy_from_slice(&new_neighbors[..len]);
            self.l0_counts[idx] = len as u16;
        } else {
            let count_idx = self.upper_count_offsets[idx] as usize + (layer - 1);
            let start = self.upper_offsets[idx] as usize + (layer - 1) * self.config.m;
            self.upper_neighbors[start..start + len].copy_from_slice(&new_neighbors[..len]);
            self.upper_counts[count_idx] = len as u16;
        }
    }

    fn sample_level(&mut self) -> u8 {
        let u = self.rng.next_f64_open0();
        let lvl = (-u.ln() * self.config.ml).floor() as usize;
        lvl.min(u8::MAX as usize) as u8
    }

    /// Insert a vector into the HNSW index (Malkov & Yashunin Algorithm 1).
    ///
    /// # Errors
    ///
    /// * [`IndexError::Core(CoreError::DimensionMismatch)`] if `vector.len() != self.dim`.
    /// * [`IndexError::Core(CoreError::NonFiniteValue)`] if any component is NaN or infinite.
    /// * [`IndexError::Core(CoreError::ZeroVector)`] if cosine metric is used and vector is zero.
    pub fn insert(&mut self, id: VectorId, mut vector: Vec<f32>) -> Result<(), IndexError> {
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

        let node_id = u32::try_from(self.ids.len()).map_err(|_| IndexError::CapacityExceeded)?;
        let level = self.sample_level();

        // Append vector and allocate fixed-capacity neighbor slots
        self.vectors.extend_from_slice(&vector);
        self.ids.push(id);
        self.node_levels.push(level);

        self.l0_neighbors
            .resize(self.l0_neighbors.len() + self.config.m0, 0);
        self.l0_counts.push(0);

        if level > 0 {
            let num_upper = level as usize;
            let u_off = self.upper_neighbors.len() as u32;
            let uc_off = self.upper_counts.len() as u32;
            self.upper_offsets.push(u_off);
            self.upper_count_offsets.push(uc_off);
            self.upper_neighbors
                .resize(self.upper_neighbors.len() + num_upper * self.config.m, 0);
            self.upper_counts
                .resize(self.upper_counts.len() + num_upper, 0);
        } else {
            self.upper_offsets.push(0);
            self.upper_count_offsets.push(0);
        }

        let Some(mut curr_obj) = self.entry_point else {
            self.entry_point = Some(node_id);
            self.max_level = level as usize;
            return Ok(());
        };

        let q_start = node_id as usize * self.dim;
        let level_usize = level as usize;
        let max_lvl = self.max_level;

        // Phase 1: greedy descent from max_level down to level + 1 with ef = 1
        if max_lvl > level_usize {
            let mut curr_dist = self.dist_unchecked(
                &self.vectors[q_start..q_start + self.dim],
                self.vec_slice(curr_obj),
            );
            for lc in (level_usize + 1..=max_lvl).rev() {
                curr_obj = self.greedy_search_layer(
                    &self.vectors[q_start..q_start + self.dim],
                    curr_obj,
                    &mut curr_dist,
                    lc,
                );
            }
        }

        // Phase 2: search and connect from min(level, max_level) down to 0
        let top_connect = level_usize.min(max_lvl);
        let num_nodes = self.ids.len();
        let mut scratch = std::mem::take(&mut self.insert_scratch);

        for lc in (0..=top_connect).rev() {
            let candidates = {
                let q_vec = &self.vectors[q_start..q_start + self.dim];
                self.search_layer_internal(
                    q_vec,
                    curr_obj,
                    self.config.ef_construction,
                    lc,
                    num_nodes,
                    &mut scratch,
                )
            };

            if let Some(&(_, closest)) = candidates.first() {
                curr_obj = closest;
            }

            let m_max = self.max_neighbors(lc);
            let selected = {
                let q_vec = &self.vectors[q_start..q_start + self.dim];
                self.select_neighbors_heuristic(q_vec, &candidates, m_max)
            };

            self.set_neighbors(node_id, lc, &selected);

            // Add reverse edges and shrink if a neighbor exceeds capacity
            for &nb in &selected {
                let nb_idx = nb as usize;
                let (nb_count, nb_cap) = if lc == 0 {
                    (self.l0_counts[nb_idx] as usize, self.config.m0)
                } else {
                    let c_idx = self.upper_count_offsets[nb_idx] as usize + (lc - 1);
                    (self.upper_counts[c_idx] as usize, self.config.m)
                };

                if nb_count < nb_cap {
                    if lc == 0 {
                        let start = nb_idx * self.config.m0;
                        self.l0_neighbors[start + nb_count] = node_id;
                        self.l0_counts[nb_idx] = (nb_count + 1) as u16;
                    } else {
                        let start = self.upper_offsets[nb_idx] as usize + (lc - 1) * self.config.m;
                        let c_idx = self.upper_count_offsets[nb_idx] as usize + (lc - 1);
                        self.upper_neighbors[start + nb_count] = node_id;
                        self.upper_counts[c_idx] = (nb_count + 1) as u16;
                    }
                } else {
                    let nb_start = nb_idx * self.dim;
                    let mut combined: Vec<(NotNan<f32>, u32)> = Vec::with_capacity(nb_cap + 1);
                    for &existing in self.neighbors_slice(nb, lc) {
                        let d = self.dist_unchecked(
                            &self.vectors[nb_start..nb_start + self.dim],
                            self.vec_slice(existing),
                        );
                        combined.push((to_not_nan(d), existing));
                    }
                    let d_new = self.dist_unchecked(
                        &self.vectors[nb_start..nb_start + self.dim],
                        &self.vectors[q_start..q_start + self.dim],
                    );
                    combined.push((to_not_nan(d_new), node_id));
                    combined.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

                    let pruned = self.select_neighbors_heuristic(
                        &self.vectors[nb_start..nb_start + self.dim],
                        &combined,
                        nb_cap,
                    );
                    self.set_neighbors(nb, lc, &pruned);
                }
            }
        }

        self.insert_scratch = scratch;

        if level_usize > self.max_level {
            self.entry_point = Some(node_id);
            self.max_level = level_usize;
        }

        Ok(())
    }

    /// Greedy search on a single layer (`ef = 1`), updating `curr_dist` in place.
    fn greedy_search_layer(
        &self,
        query: &[f32],
        mut curr_obj: u32,
        curr_dist: &mut f32,
        layer: usize,
    ) -> u32 {
        loop {
            let mut changed = false;
            for &nb in self.neighbors_slice(curr_obj, layer) {
                let d = self.dist_unchecked(query, self.vec_slice(nb));
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

    /// `SEARCH-LAYER` (Malkov & Yashunin Algorithm 2), returning `(dist, node_id)`
    /// sorted by ascending distance.
    fn search_layer_internal(
        &self,
        query: &[f32],
        entry_id: u32,
        ef: usize,
        layer: usize,
        num_nodes: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<(NotNan<f32>, u32)> {
        scratch.prepare(num_nodes);
        scratch.mark_visited(entry_id);

        let d_ep = to_not_nan(self.dist_unchecked(query, self.vec_slice(entry_id)));
        scratch.candidates.push(Reverse((d_ep, entry_id)));
        scratch.results.push((d_ep, entry_id));

        while let Some(Reverse((c_dist, c_id))) = scratch.candidates.pop() {
            let f_dist = scratch.results.peek().map_or(d_ep, |&(d, _)| d);
            if c_dist > f_dist && scratch.results.len() >= ef {
                break;
            }

            for &nb in self.neighbors_slice(c_id, layer) {
                if scratch.mark_visited(nb) {
                    let d = to_not_nan(self.dist_unchecked(query, self.vec_slice(nb)));
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

    /// `SELECT-NEIGHBORS-HEURISTIC` (Malkov & Yashunin Algorithm 4) with
    /// `keepPrunedConnections = true`.
    ///
    /// Expects `candidates` sorted by ascending distance to `query`.
    fn select_neighbors_heuristic(
        &self,
        _query: &[f32],
        candidates: &[(NotNan<f32>, u32)],
        m_max: usize,
    ) -> Vec<u32> {
        let mut selected: Vec<(NotNan<f32>, u32)> = Vec::with_capacity(m_max);
        let mut discarded: Vec<u32> = Vec::new();

        for &(d_qe, e) in candidates {
            if selected.len() >= m_max {
                break;
            }
            let e_vec = self.vec_slice(e);
            let mut closer_to_query = true;
            for &(_, r) in &selected {
                let d_er = self.dist_unchecked(e_vec, self.vec_slice(r));
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

    /// Search for the `k` nearest neighbors using the index's configured `ef_search`.
    ///
    /// Uses a thread-local [`SearchScratch`] to avoid per-query allocations.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        self.search_with_ef(query, k, self.config.ef_search)
    }

    /// Search for the `k` nearest neighbors with an explicit `ef_search` beam width.
    pub fn search_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<Vec<(VectorId, f32)>, IndexError> {
        TLS_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            self.search_with_scratch(query, k, ef_search, &mut scratch)
        })
    }

    /// Search for the `k` nearest neighbors using a caller-provided [`SearchScratch`].
    pub fn search_with_scratch(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        scratch: &mut SearchScratch,
    ) -> Result<Vec<(VectorId, f32)>, IndexError> {
        let Some(mut curr_obj) = self.entry_point else {
            return Err(IndexError::Empty);
        };
        self.validate_query(query)?;
        if k == 0 {
            return Ok(Vec::new());
        }

        let mut taken_buf = if self.metric == DistanceKind::Cosine {
            let mut buf = std::mem::take(&mut scratch.query_buf);
            buf.clear();
            buf.extend_from_slice(query);
            normalize_in_place(&mut buf);
            Some(buf)
        } else {
            None
        };
        let q_slice: &[f32] = taken_buf.as_deref().unwrap_or(query);

        let mut curr_dist = self.dist_unchecked(q_slice, self.vec_slice(curr_obj));
        for lc in (1..=self.max_level).rev() {
            curr_obj = self.greedy_search_layer(q_slice, curr_obj, &mut curr_dist, lc);
        }

        let ef = ef_search.max(k);
        let layer0_res =
            self.search_layer_internal(q_slice, curr_obj, ef, 0, self.ids.len(), scratch);
        if let Some(buf) = taken_buf.take() {
            scratch.query_buf = buf;
        }

        let mut out: Vec<(VectorId, f32)> = layer0_res
            .into_iter()
            .take(k)
            .map(|(d, nid)| (self.ids[nid as usize], d.into_inner()))
            .collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        Ok(out)
    }

    /// Search for the `k` nearest neighbors and return the visited [`VectorId`]s per layer.
    pub fn search_with_trace(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
    ) -> Result<(Vec<(VectorId, f32)>, SearchTrace), IndexError> {
        let Some(mut curr_obj) = self.entry_point else {
            return Err(IndexError::Empty);
        };
        self.validate_query(query)?;

        let mut scratch = SearchScratch::new();
        let q_owned: Vec<f32>;
        let q_slice: &[f32] = if self.metric == DistanceKind::Cosine {
            let mut buf = query.to_vec();
            normalize_in_place(&mut buf);
            q_owned = buf;
            &q_owned
        } else {
            query
        };

        let mut visited_per_layer = vec![Vec::new(); self.max_level + 1];
        let mut curr_dist = self.dist_unchecked(q_slice, self.vec_slice(curr_obj));

        for lc in (1..=self.max_level).rev() {
            visited_per_layer[lc].push(self.ids[curr_obj as usize]);
            loop {
                let mut changed = false;
                for &nb in self.neighbors_slice(curr_obj, lc) {
                    visited_per_layer[lc].push(self.ids[nb as usize]);
                    let d = self.dist_unchecked(q_slice, self.vec_slice(nb));
                    if d < curr_dist {
                        curr_dist = d;
                        curr_obj = nb;
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
        }

        if k == 0 {
            return Ok((Vec::new(), SearchTrace { visited_per_layer }));
        }

        let ef = ef_search.max(k);
        scratch.prepare(self.ids.len());
        scratch.mark_visited(curr_obj);
        visited_per_layer[0].push(self.ids[curr_obj as usize]);

        let d_ep = to_not_nan(self.dist_unchecked(q_slice, self.vec_slice(curr_obj)));
        scratch.candidates.push(Reverse((d_ep, curr_obj)));
        scratch.results.push((d_ep, curr_obj));

        while let Some(Reverse((c_dist, c_id))) = scratch.candidates.pop() {
            let f_dist = scratch.results.peek().map_or(d_ep, |&(d, _)| d);
            if c_dist > f_dist && scratch.results.len() >= ef {
                break;
            }
            for &nb in self.neighbors_slice(c_id, 0) {
                if scratch.mark_visited(nb) {
                    visited_per_layer[0].push(self.ids[nb as usize]);
                    let d = to_not_nan(self.dist_unchecked(q_slice, self.vec_slice(nb)));
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

        let mut res: Vec<(NotNan<f32>, u32)> = scratch.results.drain().collect();
        res.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut out: Vec<(VectorId, f32)> = res
            .into_iter()
            .take(k)
            .map(|(d, nid)| (self.ids[nid as usize], d.into_inner()))
            .collect();
        out.sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });

        Ok((out, SearchTrace { visited_per_layer }))
    }

    fn validate_query(&self, query: &[f32]) -> Result<(), IndexError> {
        if query.len() != self.dim {
            return Err(IndexError::Core(CoreError::DimensionMismatch {
                expected: self.dim,
                got: query.len(),
            }));
        }
        self.distance
            .validate_vector(query)
            .map_err(IndexError::Core)?;
        Ok(())
    }
}

impl SearchIndex for HnswIndex {
    fn insert(&mut self, id: VectorId, vector: Vec<f32>) -> Result<(), IndexError> {
        HnswIndex::insert(self, id, vector)
    }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(VectorId, f32)>, IndexError> {
        HnswIndex::search(self, query, k)
    }

    fn len(&self) -> usize {
        HnswIndex::len(self)
    }

    fn dim(&self) -> usize {
        HnswIndex::dim(self)
    }

    fn metric(&self) -> DistanceKind {
        HnswIndex::metric(self)
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
    use crate::BruteForceIndex;
    use proptest::prelude::*;
    use std::collections::{HashSet, VecDeque};
    use vl_core::{CosineDistance, L2Distance};

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
    fn graph_invariants_10k_build() {
        let dim = 16;
        let n = 10_000;
        let cfg = HnswConfig::new(16, 100, 50, 12345);
        let mut idx = HnswIndex::new(dim, Arc::new(L2Distance), cfg);
        for (i, v) in make_random_vecs(n, dim, 99).into_iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        assert_eq!(idx.len(), n);

        // 1. Capacity bounds, no self-loops, no duplicate neighbors
        for u in 0..n as u32 {
            let lvl = idx.node_levels[u as usize] as usize;
            for l in 0..=lvl {
                let nbs = idx.node_neighbors_at_layer(u, l);
                let cap = if l == 0 { cfg.m0 } else { cfg.m };
                assert!(
                    nbs.len() <= cap,
                    "node {u} layer {l} exceeded cap {cap}: {}",
                    nbs.len()
                );
                let mut seen = HashSet::with_capacity(nbs.len());
                for &nb in nbs {
                    assert_ne!(nb, u, "self-loop on node {u} layer {l}");
                    assert!(
                        seen.insert(nb),
                        "duplicate neighbor {nb} on node {u} layer {l}"
                    );
                }
            }
        }

        // 2. Every node reachable from entry point at layer 0
        let ep = idx.entry_point.unwrap();
        let mut visited = vec![false; n];
        let mut queue = VecDeque::new();
        visited[ep as usize] = true;
        queue.push_back(ep);
        let mut reached = 0usize;
        while let Some(u) = queue.pop_front() {
            reached += 1;
            for &nb in idx.node_neighbors_at_layer(u, 0) {
                if !visited[nb as usize] {
                    visited[nb as usize] = true;
                    queue.push_back(nb);
                }
            }
        }
        assert_eq!(reached, n, "only {reached}/{n} nodes reachable on layer 0");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let vecs = make_random_vecs(500, 8, 7);
        let cfg = HnswConfig::new(12, 64, 32, 2026);
        let mut idx1 = HnswIndex::new(8, Arc::new(L2Distance), cfg);
        let mut idx2 = HnswIndex::new(8, Arc::new(L2Distance), cfg);
        for (i, v) in vecs.into_iter().enumerate() {
            idx1.insert(VectorId(i as u64), v.clone()).unwrap();
            idx2.insert(VectorId(i as u64), v).unwrap();
        }
        let q = vec![0.1f32; 8];
        let r1 = idx1.search(&q, 10).unwrap();
        let r2 = idx2.search(&q, 10).unwrap();
        assert_eq!(r1, r2);
        assert_eq!(idx1.l0_neighbors, idx2.l0_neighbors);
    }

    #[test]
    fn search_with_trace_records_visited_layers() {
        let mut idx = HnswIndex::with_defaults(4, Arc::new(L2Distance));
        for (i, v) in make_random_vecs(200, 4, 11).into_iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        let (res, trace) = idx.search_with_trace(&[0.0, 0.0, 0.0, 0.0], 5, 20).unwrap();
        assert_eq!(res.len(), 5);
        assert_eq!(trace.visited_per_layer.len(), idx.max_level() + 1);
        assert!(!trace.visited_per_layer[0].is_empty());
    }

    #[test]
    fn cosine_normalization_and_validation() {
        let mut idx = HnswIndex::with_defaults(3, Arc::new(CosineDistance));
        assert!(matches!(
            idx.insert(VectorId(0), vec![0.0, 0.0, 0.0]),
            Err(IndexError::Core(CoreError::ZeroVector))
        ));
        idx.insert(VectorId(1), vec![2.0, 0.0, 0.0]).unwrap();
        idx.insert(VectorId(2), vec![0.0, 5.0, 0.0]).unwrap();
        let res = idx.search(&[10.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(res[0].0, VectorId(1));
        assert!(res[0].1 < 1e-5);
        assert_eq!(res[1].0, VectorId(2));
        assert!((res[1].1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn recall_at_10_on_siftsmall_if_present() {
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

        let mut bf = BruteForceIndex::new(dim, Arc::new(L2Distance));
        let mut hnsw = HnswIndex::new(dim, Arc::new(L2Distance), HnswConfig::new(16, 200, 100, 42));
        for (i, v) in base.into_iter().enumerate() {
            let id = VectorId(i as u64);
            bf.insert(id, v.clone()).unwrap();
            hnsw.insert(id, v).unwrap();
        }

        let k = 10;
        let mut recall_sum = 0.0;
        for q in &queries {
            let gt: HashSet<VectorId> = bf
                .search(q, k)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got = hnsw.search_with_ef(q, k, 100).unwrap();
            let hits = got.iter().filter(|(id, _)| gt.contains(id)).count();
            recall_sum += hits as f64 / k as f64;
        }
        let recall = recall_sum / queries.len() as f64;
        println!("siftsmall HNSW recall@10 (ef_search=100): {recall:.4}");
        assert!(
            recall >= 0.95,
            "siftsmall recall@10 {recall:.4} < 0.95 at ef_search=100"
        );
    }

    #[test]
    fn scratch_reused_after_index_grows() {
        let mut idx = HnswIndex::new(4, Arc::new(L2Distance), HnswConfig::new(8, 32, 16, 1));
        for (i, v) in make_random_vecs(50, 4, 10).into_iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        let mut scratch = SearchScratch::new();
        let q = vec![0.0f32; 4];
        let r1 = idx.search_with_scratch(&q, 5, 16, &mut scratch).unwrap();
        assert_eq!(r1.len(), 5);
        assert_eq!(scratch.visited.len(), 50);

        for (i, v) in make_random_vecs(200, 4, 20).into_iter().enumerate() {
            idx.insert(VectorId((50 + i) as u64), v).unwrap();
        }
        let r2 = idx.search_with_scratch(&q, 5, 16, &mut scratch).unwrap();
        assert_eq!(r2.len(), 5);
        assert_eq!(scratch.visited.len(), 250);
    }

    #[test]
    fn scratch_generation_wraparound() {
        let mut idx = HnswIndex::new(4, Arc::new(L2Distance), HnswConfig::new(8, 32, 32, 2));
        for (i, v) in make_random_vecs(30, 4, 30).into_iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        let mut scratch = SearchScratch::new();
        scratch.prepare(30);
        scratch.gen = u32::MAX;
        scratch.visited.fill(u32::MAX);

        let q = vec![0.1f32; 4];
        let res = idx.search_with_scratch(&q, 5, 30, &mut scratch).unwrap();
        assert_eq!(res.len(), 5);
        assert_eq!(scratch.gen, 1);
    }

    #[test]
    fn two_hnsw_instances_alternated_on_same_thread() {
        let mut idx_a = HnswIndex::new(4, Arc::new(L2Distance), HnswConfig::new(8, 32, 16, 3));
        let mut idx_b = HnswIndex::new(4, Arc::new(L2Distance), HnswConfig::new(8, 32, 16, 4));
        for i in 0..40u64 {
            idx_a
                .insert(VectorId(i), vec![i as f32, 0.0, 0.0, 0.0])
                .unwrap();
            idx_b
                .insert(VectorId(1000 + i), vec![0.0, i as f32, 0.0, 0.0])
                .unwrap();
        }
        let ra1 = idx_a.search(&[0.0, 0.0, 0.0, 0.0], 3).unwrap();
        let rb1 = idx_b.search(&[0.0, 0.0, 0.0, 0.0], 3).unwrap();
        let ra2 = idx_a.search(&[39.0, 0.0, 0.0, 0.0], 3).unwrap();
        let rb2 = idx_b.search(&[0.0, 39.0, 0.0, 0.0], 3).unwrap();

        assert_eq!(ra1[0].0, VectorId(0));
        assert_eq!(rb1[0].0, VectorId(1000));
        assert_eq!(ra2[0].0, VectorId(39));
        assert_eq!(rb2[0].0, VectorId(1039));
    }

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 64,
            ..Default::default()
        })]

        #[test]
        fn ef_ge_n_returns_exact_results(
            n in 5usize..=40,
            dim in 2usize..=16,
            k in 1usize..=10,
            seed in 0u64..u64::MAX,
        ) {
            let k_clamped = k.min(n);
            let vecs = make_random_vecs(n, dim, seed);
            let mut bf = BruteForceIndex::new(dim, Arc::new(L2Distance));
            let mut hnsw = HnswIndex::new(
                dim,
                Arc::new(L2Distance),
                HnswConfig::new(8, n.max(16), n, seed),
            );
            for (i, v) in vecs.into_iter().enumerate() {
                let id = VectorId(i as u64);
                bf.insert(id, v.clone()).unwrap();
                hnsw.insert(id, v).unwrap();
            }
            let query = make_random_vecs(1, dim, seed.wrapping_add(1)).pop().unwrap();
            let expected: Vec<VectorId> = bf
                .search(&query, k_clamped)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got: Vec<VectorId> = hnsw
                .search_with_ef(&query, k_clamped, n)
                .unwrap()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            prop_assert_eq!(got, expected);
        }
    }
}
