//! Pure-Rust scalar distance kernels.
//!
//! These serve two roles:
//! 1. The fallback path on CPUs without supported SIMD extensions.
//! 2. The ground-truth reference in property tests that verify SIMD correctness.
//!
//! All functions operate on f32 slices and assume the caller has already
//! validated dimensions and finiteness. They use `debug_assert_eq!` for safety in debug builds.

/// Naive squared Euclidean (L2²) distance: `Σ (a[i] - b[i])²`.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Unrolled 4-accumulator squared Euclidean (L2²) distance.
///
/// Breaks dependency chains by maintaining 4 separate accumulators.
#[inline]
pub fn l2_squared_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }

    let tail_start = chunks * 4;
    let mut acc = (s0 + s1) + (s2 + s3);
    for i in tail_start..len {
        let d = a[i] - b[i];
        acc += d * d;
    }
    acc
}

/// Naive inner (dot) product: `Σ a[i] * b[i]`.
#[inline]
pub fn inner_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Unrolled 4-accumulator inner (dot) product.
#[inline]
pub fn inner_product_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut s0 = 0.0f32;
    let mut s1 = 0.0f32;
    let mut s2 = 0.0f32;
    let mut s3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        s0 += a[base] * b[base];
        s1 += a[base + 1] * b[base + 1];
        s2 += a[base + 2] * b[base + 2];
        s3 += a[base + 3] * b[base + 3];
    }

    let tail_start = chunks * 4;
    let mut acc = (s0 + s1) + (s2 + s3);
    for i in tail_start..len {
        acc += a[i] * b[i];
    }
    acc
}

/// Naive cosine distance: `1 - (a·b) / (‖a‖ · ‖b‖)`.
///
/// Returns `f32::NAN` if either vector is the zero vector.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = inner_product(a, b);
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    let denom = norm_a * norm_b;
    if denom == 0.0 {
        f32::NAN
    } else {
        1.0 - (dot / denom)
    }
}

/// Unrolled 4-accumulator cosine distance: `1 - (a·b) / (‖a‖ · ‖b‖)`.
#[inline]
pub fn cosine_distance_unrolled4(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 4;
    let mut dot0 = 0.0f32;
    let mut dot1 = 0.0f32;
    let mut dot2 = 0.0f32;
    let mut dot3 = 0.0f32;

    let mut na0 = 0.0f32;
    let mut na1 = 0.0f32;
    let mut na2 = 0.0f32;
    let mut na3 = 0.0f32;

    let mut nb0 = 0.0f32;
    let mut nb1 = 0.0f32;
    let mut nb2 = 0.0f32;
    let mut nb3 = 0.0f32;

    for i in 0..chunks {
        let base = i * 4;
        let a0 = a[base];
        let a1 = a[base + 1];
        let a2 = a[base + 2];
        let a3 = a[base + 3];

        let b0 = b[base];
        let b1 = b[base + 1];
        let b2 = b[base + 2];
        let b3 = b[base + 3];

        dot0 += a0 * b0;
        dot1 += a1 * b1;
        dot2 += a2 * b2;
        dot3 += a3 * b3;

        na0 += a0 * a0;
        na1 += a1 * a1;
        na2 += a2 * a2;
        na3 += a3 * a3;

        nb0 += b0 * b0;
        nb1 += b1 * b1;
        nb2 += b2 * b2;
        nb3 += b3 * b3;
    }

    let tail_start = chunks * 4;
    let mut dot = (dot0 + dot1) + (dot2 + dot3);
    let mut norm_sq_a = (na0 + na1) + (na2 + na3);
    let mut norm_sq_b = (nb0 + nb1) + (nb2 + nb3);

    for i in tail_start..len {
        let ai = a[i];
        let bi = b[i];
        dot += ai * bi;
        norm_sq_a += ai * ai;
        norm_sq_b += bi * bi;
    }

    let denom = norm_sq_a.sqrt() * norm_sq_b.sqrt();
    if denom == 0.0 {
        f32::NAN
    } else {
        1.0 - (dot / denom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn l2_squared_identical_vectors() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert!((l2_squared(&v, &v)).abs() < EPS);
        assert!((l2_squared_unrolled4(&v, &v)).abs() < EPS);
    }

    #[test]
    fn l2_squared_known_value() {
        let a = vec![0.0f32, 0.0, 0.0, 1.0, 2.0];
        let b = vec![1.0f32, 2.0, 2.0, 1.0, 2.0];
        let expected = 9.0;
        assert!((l2_squared(&a, &b) - expected).abs() < EPS);
        assert!((l2_squared_unrolled4(&a, &b) - expected).abs() < EPS);
    }

    #[test]
    fn inner_product_known_value() {
        let a = vec![1.0f32, 2.0, 3.0, 1.0];
        let b = vec![4.0f32, 5.0, 6.0, 2.0];
        let expected = 34.0;
        assert!((inner_product(&a, &b) - expected).abs() < EPS);
        assert!((inner_product_unrolled4(&a, &b) - expected).abs() < EPS);
    }

    #[test]
    fn cosine_distance_identical_unit_vectors() {
        let v = vec![1.0f32, 0.0, 0.0, 0.0, 0.0];
        assert!(cosine_distance(&v, &v).abs() < EPS);
        assert!(cosine_distance_unrolled4(&v, &v).abs() < EPS);
    }

    #[test]
    fn cosine_distance_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < EPS);
        assert!((cosine_distance_unrolled4(&a, &b) - 1.0).abs() < EPS);
    }

    #[test]
    fn cosine_distance_zero_vector_returns_nan() {
        let zero = vec![0.0f32, 0.0, 0.0];
        let other = vec![1.0f32, 2.0, 3.0];
        assert!(cosine_distance(&zero, &other).is_nan());
        assert!(cosine_distance_unrolled4(&zero, &other).is_nan());
    }

    #[test]
    fn inner_product_empty_slices() {
        assert_eq!(inner_product(&[], &[]), 0.0);
        assert_eq!(inner_product_unrolled4(&[], &[]), 0.0);
    }
}
