//! Criterion benchmarks comparing naive scalar, 4-accumulator scalar, and AVX2 distance kernels.
//!
//! Run with:
//!   cargo bench -p vl-bench --bench distance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use vl_core::kernels::scalar;

/// Deterministic LCG RNG.
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

// ---------------------------------------------------------------------------
// L2 benchmarks
// ---------------------------------------------------------------------------

fn bench_l2(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");

    for dim in [128usize, 768] {
        let a = make_vec(dim, 0xDEAD_BEEF);
        let b = make_vec(dim, 0xCAFE_BABE);

        group.bench_with_input(BenchmarkId::new("scalar_naive", dim), &dim, |bench, _| {
            bench.iter(|| scalar::l2_squared(black_box(a.as_slice()), black_box(b.as_slice())));
        });

        group.bench_with_input(BenchmarkId::new("scalar_unroll4", dim), &dim, |bench, _| {
            bench.iter(|| {
                scalar::l2_squared_unrolled4(black_box(a.as_slice()), black_box(b.as_slice()))
            });
        });

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                group.bench_with_input(BenchmarkId::new("avx2", dim), &dim, |bench, _| {
                    bench.iter(|| {
                        // SAFETY: guarded by is_x86_feature_detected!
                        unsafe {
                            vl_core::kernels::avx2::l2_squared(
                                black_box(a.as_slice()),
                                black_box(b.as_slice()),
                            )
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Inner product benchmarks
// ---------------------------------------------------------------------------

fn bench_inner_product(c: &mut Criterion) {
    let mut group = c.benchmark_group("inner_product");

    for dim in [128usize, 768] {
        let a = make_vec(dim, 0x1111_2222);
        let b = make_vec(dim, 0x3333_4444);

        group.bench_with_input(BenchmarkId::new("scalar_naive", dim), &dim, |bench, _| {
            bench.iter(|| scalar::inner_product(black_box(a.as_slice()), black_box(b.as_slice())));
        });

        group.bench_with_input(BenchmarkId::new("scalar_unroll4", dim), &dim, |bench, _| {
            bench.iter(|| {
                scalar::inner_product_unrolled4(black_box(a.as_slice()), black_box(b.as_slice()))
            });
        });

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                group.bench_with_input(BenchmarkId::new("avx2", dim), &dim, |bench, _| {
                    bench.iter(|| {
                        // SAFETY: guarded by is_x86_feature_detected!
                        unsafe {
                            vl_core::kernels::avx2::inner_product(
                                black_box(a.as_slice()),
                                black_box(b.as_slice()),
                            )
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Cosine benchmarks
// ---------------------------------------------------------------------------

fn bench_cosine(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dim in [128usize, 768] {
        let a = make_vec(dim, 0xAAAA_BBBB);
        let b = make_vec(dim, 0xCCCC_DDDD);

        group.bench_with_input(BenchmarkId::new("scalar_naive", dim), &dim, |bench, _| {
            bench
                .iter(|| scalar::cosine_distance(black_box(a.as_slice()), black_box(b.as_slice())));
        });

        group.bench_with_input(BenchmarkId::new("scalar_unroll4", dim), &dim, |bench, _| {
            bench.iter(|| {
                scalar::cosine_distance_unrolled4(black_box(a.as_slice()), black_box(b.as_slice()))
            });
        });

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                group.bench_with_input(BenchmarkId::new("avx2", dim), &dim, |bench, _| {
                    bench.iter(|| {
                        // SAFETY: guarded by is_x86_feature_detected!
                        unsafe {
                            vl_core::kernels::avx2::cosine_distance(
                                black_box(a.as_slice()),
                                black_box(b.as_slice()),
                            )
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

criterion_group!(benches, bench_l2, bench_inner_product, bench_cosine);
criterion_main!(benches);
