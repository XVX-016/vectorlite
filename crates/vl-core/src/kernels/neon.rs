//! NEON distance kernels for aarch64.
//!
//! This module is compiled **only** on `target_arch = "aarch64"` and exercised
//! in CI on the `macos-14` runner (Apple Silicon).
//!
//! # Safety invariants shared by all kernels
//!
//! * The caller guarantees NEON support via `std::arch::is_aarch64_feature_detected!("neon")`.
//! * Input slices are valid, non-null, and have the same length.
//! * Unaligned loads (`vld1q_f32`) allow slices with arbitrary alignment.
//! * The tail (< 4 elements) is handled by the scalar fallback.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*;

use super::scalar;

/// NEON squared L2 distance.
///
/// Processes 4 f32 lanes per iteration; scalar tail handles the remainder.
///
/// # Safety
///
/// Must only be called when NEON is available.
/// The slices `a` and `b` must have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let base = i * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        let diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    let tail_start = chunks * 4;
    vaddvq_f32(acc) + scalar::l2_squared(&a[tail_start..], &b[tail_start..])
}

/// NEON inner (dot) product.
///
/// # Safety
///
/// Must only be called when NEON is available.
/// The slices `a` and `b` must have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let base = i * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        acc = vfmaq_f32(acc, va, vb);
    }

    let tail_start = chunks * 4;
    vaddvq_f32(acc) + scalar::inner_product(&a[tail_start..], &b[tail_start..])
}

/// NEON cosine distance: `1 - dot(a,b) / (‖a‖ · ‖b‖)`.
///
/// Computes dot product and both squared norms in a single pass.
///
/// # Safety
///
/// Must only be called when NEON is available.
/// The slices `a` and `b` must have the same length.
#[target_feature(enable = "neon")]
pub unsafe fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc_dot = vdupq_n_f32(0.0);
    let mut acc_na = vdupq_n_f32(0.0);
    let mut acc_nb = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let base = i * 4;
        let va = vld1q_f32(a.as_ptr().add(base));
        let vb = vld1q_f32(b.as_ptr().add(base));
        acc_dot = vfmaq_f32(acc_dot, va, vb);
        acc_na = vfmaq_f32(acc_na, va, va);
        acc_nb = vfmaq_f32(acc_nb, vb, vb);
    }

    let tail_start = chunks * 4;
    let tail_a = &a[tail_start..];
    let tail_b = &b[tail_start..];

    let dot = vaddvq_f32(acc_dot) + scalar::inner_product(tail_a, tail_b);
    let norm_a = (vaddvq_f32(acc_na) + tail_a.iter().map(|x| x * x).sum::<f32>()).sqrt();
    let norm_b = (vaddvq_f32(acc_nb) + tail_b.iter().map(|x| x * x).sum::<f32>()).sqrt();

    let denom = norm_a * norm_b;
    if denom == 0.0 {
        f32::NAN
    } else {
        1.0 - (dot / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn neon_available() -> bool {
        std::arch::is_aarch64_feature_detected!("neon")
    }

    const EPS: f32 = 1e-4;

    #[test]
    fn neon_l2_known_value() {
        if !neon_available() {
            return;
        }
        let a = vec![0.0f32; 8];
        let b: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let expected = super::super::scalar::l2_squared(&a, &b);
        // SAFETY: guarded by neon_available()
        let got = unsafe { l2_squared(&a, &b) };
        assert!(
            (got - expected).abs() < EPS,
            "NEON L2 {got} vs scalar {expected}"
        );
    }

    #[test]
    fn neon_ip_known_value() {
        if !neon_available() {
            return;
        }
        let a: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let expected = super::super::scalar::inner_product(&a, &b);
        // SAFETY: guarded by neon_available()
        let got = unsafe { inner_product(&a, &b) };
        assert!(
            (got - expected).abs() < EPS,
            "NEON IP {got} vs scalar {expected}"
        );
    }

    #[test]
    fn neon_cosine_orthogonal() {
        if !neon_available() {
            return;
        }
        let a = vec![1.0f32, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0, 0.0];
        // SAFETY: guarded by neon_available()
        let got = unsafe { cosine_distance(&a, &b) };
        assert!((got - 1.0).abs() < EPS, "NEON cosine orthogonal: {got}");
    }

    #[test]
    fn neon_handles_non_multiple_of_4() {
        if !neon_available() {
            return;
        }
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let expected = super::super::scalar::l2_squared(&a, &b);
        // SAFETY: guarded by neon_available()
        let got = unsafe { l2_squared(&a, &b) };
        assert!(
            (got - expected).abs() < EPS,
            "NEON tail: got={got} expected={expected}"
        );
    }
}
