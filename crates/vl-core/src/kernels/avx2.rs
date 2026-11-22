//! AVX2 and FMA distance kernels for x86_64.
//!
//! Each function is guarded by `#[cfg(target_arch = "x86_64")]` at the
//! module level and must only be called after verifying AVX2 and FMA support with
//! `is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")` at runtime.
//!
//! # Safety invariants shared by all kernels
//!
//! * The caller guarantees the CPU supports AVX2 and FMA (checked in `kernels::mod`).
//! * Input slices are valid, non-null, and have the same length.
//! * Loop unrolling uses 4 independent 256-bit accumulators (32 `f32`s per iteration)
//!   to break data dependency chains and maximize execution port throughput.
//! * All vector memory accesses use unaligned loads (`_mm256_loadu_ps`), so pointers
//!   need not be aligned to 32-byte boundaries.
//! * Chunks of 8 are processed next, and the tail (< 8 elements) is handled by a safe scalar loop.

#![cfg(target_arch = "x86_64")]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar;

/// AVX2 + FMA squared L2 distance: `Σ (a[i] - b[i])²`.
///
/// # Safety
///
/// Must only be called when AVX2 and FMA features are detected.
/// Slices `a` and `b` must have equal length.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();

    for i in 0..chunks32 {
        let base = i * 32;
        let va0 = _mm256_loadu_ps(ptr_a.add(base));
        let vb0 = _mm256_loadu_ps(ptr_b.add(base));
        let diff0 = _mm256_sub_ps(va0, vb0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        let va1 = _mm256_loadu_ps(ptr_a.add(base + 8));
        let vb1 = _mm256_loadu_ps(ptr_b.add(base + 8));
        let diff1 = _mm256_sub_ps(va1, vb1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        let va2 = _mm256_loadu_ps(ptr_a.add(base + 16));
        let vb2 = _mm256_loadu_ps(ptr_b.add(base + 16));
        let diff2 = _mm256_sub_ps(va2, vb2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        let va3 = _mm256_loadu_ps(ptr_a.add(base + 24));
        let vb3 = _mm256_loadu_ps(ptr_b.add(base + 24));
        let diff3 = _mm256_sub_ps(va3, vb3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
    }

    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    let processed32 = chunks32 * 32;
    let rem8 = (len - processed32) / 8;
    for j in 0..rem8 {
        let base = processed32 + j * 8;
        let va = _mm256_loadu_ps(ptr_a.add(base));
        let vb = _mm256_loadu_ps(ptr_b.add(base));
        let diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    let tail_start = processed32 + rem8 * 8;
    hsum256(acc) + scalar::l2_squared(&a[tail_start..], &b[tail_start..])
}

/// AVX2 + FMA inner (dot) product: `Σ a[i] * b[i]`.
///
/// # Safety
///
/// Must only be called when AVX2 and FMA features are detected.
/// Slices `a` and `b` must have equal length.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();

    let chunks32 = len / 32;
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();

    for i in 0..chunks32 {
        let base = i * 32;
        let va0 = _mm256_loadu_ps(ptr_a.add(base));
        let vb0 = _mm256_loadu_ps(ptr_b.add(base));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        let va1 = _mm256_loadu_ps(ptr_a.add(base + 8));
        let vb1 = _mm256_loadu_ps(ptr_b.add(base + 8));
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        let va2 = _mm256_loadu_ps(ptr_a.add(base + 16));
        let vb2 = _mm256_loadu_ps(ptr_b.add(base + 16));
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        let va3 = _mm256_loadu_ps(ptr_a.add(base + 24));
        let vb3 = _mm256_loadu_ps(ptr_b.add(base + 24));
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
    }

    let mut acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

    let processed32 = chunks32 * 32;
    let rem8 = (len - processed32) / 8;
    for j in 0..rem8 {
        let base = processed32 + j * 8;
        let va = _mm256_loadu_ps(ptr_a.add(base));
        let vb = _mm256_loadu_ps(ptr_b.add(base));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let tail_start = processed32 + rem8 * 8;
    hsum256(acc) + scalar::inner_product(&a[tail_start..], &b[tail_start..])
}

/// AVX2 + FMA cosine distance: `1 - dot(a,b) / (‖a‖ · ‖b‖)`.
///
/// Computes dot product and squared norms simultaneously using multiple registers.
///
/// # Safety
///
/// Must only be called when AVX2 and FMA features are detected.
/// Slices `a` and `b` must have equal length.
#[inline]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let mut dot0 = _mm256_setzero_ps();
    let mut dot1 = _mm256_setzero_ps();
    let mut na0 = _mm256_setzero_ps();
    let mut na1 = _mm256_setzero_ps();
    let mut nb0 = _mm256_setzero_ps();
    let mut nb1 = _mm256_setzero_ps();

    let chunks16 = len / 16;
    let ptr_a = a.as_ptr();
    let ptr_b = b.as_ptr();

    for i in 0..chunks16 {
        let base = i * 16;
        let va0 = _mm256_loadu_ps(ptr_a.add(base));
        let vb0 = _mm256_loadu_ps(ptr_b.add(base));
        dot0 = _mm256_fmadd_ps(va0, vb0, dot0);
        na0 = _mm256_fmadd_ps(va0, va0, na0);
        nb0 = _mm256_fmadd_ps(vb0, vb0, nb0);

        let va1 = _mm256_loadu_ps(ptr_a.add(base + 8));
        let vb1 = _mm256_loadu_ps(ptr_b.add(base + 8));
        dot1 = _mm256_fmadd_ps(va1, vb1, dot1);
        na1 = _mm256_fmadd_ps(va1, va1, na1);
        nb1 = _mm256_fmadd_ps(vb1, vb1, nb1);
    }

    let mut dot_acc = _mm256_add_ps(dot0, dot1);
    let mut na_acc = _mm256_add_ps(na0, na1);
    let mut nb_acc = _mm256_add_ps(nb0, nb1);

    let processed16 = chunks16 * 16;
    let rem8 = (len - processed16) / 8;
    for j in 0..rem8 {
        let base = processed16 + j * 8;
        let va = _mm256_loadu_ps(ptr_a.add(base));
        let vb = _mm256_loadu_ps(ptr_b.add(base));
        dot_acc = _mm256_fmadd_ps(va, vb, dot_acc);
        na_acc = _mm256_fmadd_ps(va, va, na_acc);
        nb_acc = _mm256_fmadd_ps(vb, vb, nb_acc);
    }

    let tail_start = processed16 + rem8 * 8;
    let tail_a = &a[tail_start..];
    let tail_b = &b[tail_start..];

    let dot = hsum256(dot_acc) + scalar::inner_product(tail_a, tail_b);
    let norm_a = (hsum256(na_acc) + tail_a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let norm_b = (hsum256(nb_acc) + tail_b.iter().map(|x| x * x).sum::<f32>()).sqrt();

    let denom = norm_a * norm_b;
    if denom == 0.0 {
        f32::NAN
    } else {
        1.0 - (dot / denom)
    }
}

/// Horizontal sum of a 256-bit AVX register containing 8 f32 lanes.
///
/// # Safety
///
/// Caller must ensure AVX2 support.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(shuf, sums);
    let result = _mm_add_ss(sums, hi64);
    _mm_cvtss_f32(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn avx2_fma_available() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }

    const EPS: f32 = 1e-4;

    #[test]
    fn avx2_l2_known_value() {
        if !avx2_fma_available() {
            return;
        }
        let a = vec![0.0f32; 64];
        let b: Vec<f32> = (1..=64).map(|x| x as f32).collect();
        let expected = super::super::scalar::l2_squared(&a, &b);
        // SAFETY: guarded by avx2_fma_available()
        let got = unsafe { l2_squared(&a, &b) };
        assert!(
            (got - expected).abs() < EPS,
            "AVX2 L2 {got} vs scalar {expected}"
        );
    }

    #[test]
    fn avx2_ip_known_value() {
        if !avx2_fma_available() {
            return;
        }
        let a: Vec<f32> = (1..=64).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=64).map(|x| x as f32).collect();
        let expected = super::super::scalar::inner_product(&a, &b);
        // SAFETY: guarded by avx2_fma_available()
        let got = unsafe { inner_product(&a, &b) };
        assert!(
            (got - expected).abs() < EPS,
            "AVX2 IP {got} vs scalar {expected}"
        );
    }

    #[test]
    fn avx2_cosine_known_value() {
        if !avx2_fma_available() {
            return;
        }
        let mut a = vec![0.0f32; 64];
        let mut b = vec![0.0f32; 64];
        a[0] = 1.0;
        b[1] = 1.0;
        // SAFETY: guarded by avx2_fma_available()
        let got = unsafe { cosine_distance(&a, &b) };
        assert!(
            (got - 1.0).abs() < EPS,
            "AVX2 cosine distance of orthogonal: {got}"
        );
    }

    #[test]
    fn avx2_handles_arbitrary_lengths() {
        if !avx2_fma_available() {
            return;
        }
        for len in [1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 67] {
            let a: Vec<f32> = (0..len).map(|x| (x + 1) as f32 * 0.5).collect();
            let b: Vec<f32> = (0..len).map(|x| (x + 2) as f32 * 0.3).collect();
            let scalar_l2 = super::super::scalar::l2_squared(&a, &b);
            // SAFETY: guarded by avx2_fma_available()
            let avx2_l2 = unsafe { l2_squared(&a, &b) };
            let diff = (avx2_l2 - scalar_l2).abs();
            let rel = diff / scalar_l2.abs().max(1e-9);
            assert!(
                rel < EPS || diff < 1e-3,
                "tail handling len {len}: avx2={avx2_l2} scalar={scalar_l2} diff={diff}"
            );
        }
    }

    #[test]
    fn avx2_unaligned_loads_lengths_0_to_40() {
        if !avx2_fma_available() {
            return;
        }
        // Allocate a large buffer and create sub-slices with deliberate misalignments (offset 1, 3, 5, 7, etc.)
        let mut buf_a = [0.0f32; 100];
        let mut buf_b = [0.0f32; 100];
        for i in 0..100 {
            buf_a[i] = (i + 1) as f32 * 0.17;
            buf_b[i] = (i + 5) as f32 * 0.23;
        }

        for len in 0..=40 {
            // Test multiple distinct odd/unaligned pointer offsets
            for offset_a in [0, 1, 3, 7] {
                for offset_b in [0, 1, 2, 5] {
                    let slice_a = &buf_a[offset_a..offset_a + len];
                    let slice_b = &buf_b[offset_b..offset_b + len];

                    let sc_l2 = super::super::scalar::l2_squared(slice_a, slice_b);
                    // SAFETY: guarded by avx2_fma_available()
                    let avx_l2 = unsafe { l2_squared(slice_a, slice_b) };
                    let diff_l2 = (avx_l2 - sc_l2).abs();
                    assert!(
                        diff_l2 < 1e-3 || diff_l2 / sc_l2.abs().max(1e-9) < 1e-4,
                        "L2 unaligned fail len {len} offset_a {offset_a} offset_b {offset_b}: sc={sc_l2} avx={avx_l2}"
                    );

                    let sc_ip = super::super::scalar::inner_product(slice_a, slice_b);
                    // SAFETY: guarded by avx2_fma_available()
                    let avx_ip = unsafe { inner_product(slice_a, slice_b) };
                    let diff_ip = (avx_ip - sc_ip).abs();
                    assert!(
                        diff_ip < 1e-3 || diff_ip / sc_ip.abs().max(1e-9) < 1e-4,
                        "IP unaligned fail len {len} offset_a {offset_a} offset_b {offset_b}: sc={sc_ip} avx={avx_ip}"
                    );

                    if len > 0 {
                        let sc_cos = super::super::scalar::cosine_distance(slice_a, slice_b);
                        // SAFETY: guarded by avx2_fma_available()
                        let avx_cos = unsafe { cosine_distance(slice_a, slice_b) };
                        let diff_cos = (avx_cos - sc_cos).abs();
                        assert!(
                            diff_cos < 1e-3,
                            "Cosine unaligned fail len {len}: sc={sc_cos} avx={avx_cos}"
                        );
                    }
                }
            }
        }
    }
}
