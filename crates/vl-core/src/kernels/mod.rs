//! Runtime dispatch layer for distance kernels.
//!
//! This module exposes [`compute_unchecked`] which executes the fastest
//! available kernel on the host CPU without repeating input validation:
//!
//! * **x86_64 + AVX2 + FMA** — SIMD path using 256-bit YMM registers with 4 accumulators (32 f32/iter).
//! * **aarch64 + NEON** — SIMD path using 128-bit NEON registers.
//! * **fallback** — Pure-Rust scalar path.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "aarch64")]
pub mod neon;

use crate::distance::DistanceKind;

/// Compute the distance between two equal-length f32 slices using the fastest
/// available kernel for the current CPU without checking dimensions or finiteness.
///
/// # Invariants (caller must ensure)
///
/// * `a.len() == b.len()`
/// * Elements in `a` and `b` are finite.
#[inline]
pub fn compute_unchecked(kind: DistanceKind, a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    // ---- x86_64 path --------------------------------------------------------
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: Verified AVX2 and FMA support on this CPU.
            // Slices are guaranteed valid &[f32] of equal length.
            return unsafe {
                match kind {
                    DistanceKind::L2 => avx2::l2_squared(a, b),
                    DistanceKind::InnerProduct => avx2::inner_product(a, b),
                    DistanceKind::Cosine => avx2::cosine_distance(a, b),
                }
            };
        }
    }

    // ---- aarch64 path -------------------------------------------------------
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // SAFETY: Verified NEON support on this CPU.
            return unsafe {
                match kind {
                    DistanceKind::L2 => neon::l2_squared(a, b),
                    DistanceKind::InnerProduct => neon::inner_product(a, b),
                    DistanceKind::Cosine => neon::cosine_distance(a, b),
                }
            };
        }
    }

    // ---- scalar fallback ----------------------------------------------------
    match kind {
        DistanceKind::L2 => scalar::l2_squared_unrolled4(a, b),
        DistanceKind::InnerProduct => scalar::inner_product_unrolled4(a, b),
        DistanceKind::Cosine => scalar::cosine_distance_unrolled4(a, b),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn rel_err(simd: f32, scalar_val: f32) -> f32 {
        (simd - scalar_val).abs() / scalar_val.abs().max(1e-9)
    }

    const TOL: f32 = 1e-4;

    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        #[test]
        fn dispatch_l2_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let scalar_val = scalar::l2_squared(&a, &b);
            let dispatched = compute_unchecked(DistanceKind::L2, &a, &b);
            prop_assert!(
                rel_err(dispatched, scalar_val) < TOL,
                "L2 rel_err={} dispatched={} scalar={}",
                rel_err(dispatched, scalar_val), dispatched, scalar_val
            );
        }

        #[test]
        fn dispatch_ip_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let scalar_val = scalar::inner_product(&a, &b);
            let dispatched = compute_unchecked(DistanceKind::InnerProduct, &a, &b);
            let err = (dispatched - scalar_val).abs();
            let rel = rel_err(dispatched, scalar_val);
            prop_assert!(
                rel < TOL || err < 1e-3,
                "IP rel_err={} abs_err={} dispatched={} scalar={}",
                rel, err, dispatched, scalar_val
            );
        }

        #[test]
        fn dispatch_cosine_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let scalar_val = scalar::cosine_distance(&a, &b);
            let dispatched = compute_unchecked(DistanceKind::Cosine, &a, &b);
            let err = (dispatched - scalar_val).abs();
            prop_assert!(
                err < 1e-3,
                "Cosine abs_err={} dispatched={} scalar={}",
                err, dispatched, scalar_val
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        #[test]
        fn avx2_l2_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::l2_squared(&a, &b);
            // SAFETY: guarded by is_x86_feature_detected!
            let si = unsafe { avx2::l2_squared(&a, &b) };
            prop_assert!(rel_err(si, sc) < TOL, "L2 rel_err={}", rel_err(si, sc));
        }

        #[test]
        fn avx2_ip_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::inner_product(&a, &b);
            // SAFETY: guarded by is_x86_feature_detected!
            let si = unsafe { avx2::inner_product(&a, &b) };
            let err = (si - sc).abs();
            prop_assert!(rel_err(si, sc) < TOL || err < 1e-3);
        }

        #[test]
        fn avx2_cosine_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::cosine_distance(&a, &b);
            // SAFETY: guarded by is_x86_feature_detected!
            let si = unsafe { avx2::cosine_distance(&a, &b) };
            prop_assert!((si - sc).abs() < 1e-3, "cosine abs_err={}", (si-sc).abs());
        }
    }

    #[cfg(target_arch = "aarch64")]
    proptest! {
        #![proptest_config(proptest::test_runner::Config {
            cases: 256,
            ..Default::default()
        })]

        #[test]
        fn neon_l2_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !std::arch::is_aarch64_feature_detected!("neon") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::l2_squared(&a, &b);
            // SAFETY: guarded by is_aarch64_feature_detected!
            let si = unsafe { neon::l2_squared(&a, &b) };
            prop_assert!(rel_err(si, sc) < TOL);
        }

        #[test]
        fn neon_ip_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !std::arch::is_aarch64_feature_detected!("neon") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::inner_product(&a, &b);
            // SAFETY: guarded by is_aarch64_feature_detected!
            let si = unsafe { neon::inner_product(&a, &b) };
            let err = (si - sc).abs();
            prop_assert!(rel_err(si, sc) < TOL || err < 1e-3);
        }

        #[test]
        fn neon_cosine_matches_scalar(
            dim in 1usize..=1024,
            seed_a in 0u64..u64::MAX,
            seed_b in 0u64..u64::MAX,
        ) {
            if !std::arch::is_aarch64_feature_detected!("neon") { return Ok(()); }
            let a = make_vec(dim, seed_a);
            let b = make_vec(dim, seed_b);
            let sc = scalar::cosine_distance(&a, &b);
            // SAFETY: guarded by is_aarch64_feature_detected!
            let si = unsafe { neon::cosine_distance(&a, &b) };
            prop_assert!((si - sc).abs() < 1e-3);
        }
    }

    fn make_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_add(1);
        (0..dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let bits = ((state >> 41) as u32) | 0x3f80_0000u32;
                let f = f32::from_bits(bits);
                if (state >> 63) == 0 {
                    f
                } else {
                    -f
                }
            })
            .collect()
    }
}
