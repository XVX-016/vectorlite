//! Readers for the TEXMEX `fvecs` and `ivecs` binary vector formats.
//!
//! Both formats store a sequence of records:
//! `[dim: i32 little-endian][dim elements of f32 or i32 little-endian]`.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use vl_core::VectorId;

use crate::BenchError;

/// Maximum supported vector dimension to guard against corrupt huge headers.
const MAX_DIM: usize = 1_048_576;

/// Read all `f32` vectors from an `.fvecs` file.
///
/// # Errors
///
/// Returns [`BenchError::MalformedFile`] if the file is empty, truncated,
/// has a non-positive or inconsistent dimension, or contains non-finite floats.
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>, BenchError> {
    read_fvecs_limit(path, None)
}

/// Read up to `limit` `f32` vectors from an `.fvecs` file.
///
/// # Errors
///
/// Returns [`BenchError::MalformedFile`] if the file is empty, truncated,
/// has a non-positive or inconsistent dimension, or contains non-finite floats.
pub fn read_fvecs_limit(path: &Path, limit: Option<usize>) -> Result<Vec<Vec<f32>>, BenchError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_fvecs_from_reader(&mut reader, limit)
}

/// Read all `i32` vectors from an `.ivecs` file.
///
/// # Errors
///
/// Returns [`BenchError::MalformedFile`] if the file is empty, truncated,
/// or has a non-positive or inconsistent dimension.
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>, BenchError> {
    read_ivecs_limit(path, None)
}

/// Read up to `limit` `i32` vectors from an `.ivecs` file.
///
/// # Errors
///
/// Returns [`BenchError::MalformedFile`] if the file is empty, truncated,
/// or has a non-positive or inconsistent dimension.
pub fn read_ivecs_limit(path: &Path, limit: Option<usize>) -> Result<Vec<Vec<i32>>, BenchError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    read_ivecs_from_reader(&mut reader, limit)
}

/// Read ground-truth neighbor IDs from an `.ivecs` file and convert to [`VectorId`].
///
/// # Errors
///
/// Returns [`BenchError::MalformedFile`] if the file is malformed or contains
/// negative vector IDs.
pub fn read_groundtruth_ivecs(
    path: &Path,
    limit: Option<usize>,
) -> Result<Vec<Vec<VectorId>>, BenchError> {
    let raw = read_ivecs_limit(path, limit)?;
    raw.into_iter()
        .enumerate()
        .map(|(q_idx, row)| {
            row.into_iter()
                .map(|id| {
                    u64::try_from(id)
                        .map(VectorId)
                        .map_err(|_| BenchError::MalformedFile {
                            reason: format!(
                                "negative ground-truth ID {id} at query record {q_idx}"
                            ),
                        })
                })
                .collect()
        })
        .collect()
}

/// Write `i32` vectors to an `.ivecs` file.
///
/// # Errors
///
/// Returns [`BenchError::Io`] on write failure or [`BenchError::InvalidConfig`]
/// if a vector length exceeds `i32::MAX`.
pub fn write_ivecs(path: &Path, vecs: &[Vec<i32>]) -> Result<(), BenchError> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for v in vecs {
        let dim = i32::try_from(v.len()).map_err(|_| {
            BenchError::InvalidConfig(format!("vector length {} exceeds i32::MAX", v.len()))
        })?;
        writer.write_all(&dim.to_le_bytes())?;
        for &x in v {
            writer.write_all(&x.to_le_bytes())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write ground-truth [`VectorId`] lists to an `.ivecs` file.
///
/// # Errors
///
/// Returns [`BenchError::InvalidConfig`] if any ID exceeds `i32::MAX` or
/// [`BenchError::Io`] on write failure.
pub fn write_groundtruth_ivecs(path: &Path, gt: &[Vec<VectorId>]) -> Result<(), BenchError> {
    let raw: Vec<Vec<i32>> = gt
        .iter()
        .enumerate()
        .map(|(q_idx, row)| {
            row.iter()
                .map(|id| {
                    i32::try_from(id.0).map_err(|_| {
                        BenchError::InvalidConfig(format!(
                            "ground-truth ID {} at query {q_idx} exceeds i32::MAX",
                            id.0
                        ))
                    })
                })
                .collect::<Result<Vec<i32>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    write_ivecs(path, &raw)
}

/// Read `fvecs` records from any [`Read`] stream.
pub fn read_fvecs_from_reader<R: Read>(
    reader: &mut R,
    limit: Option<usize>,
) -> Result<Vec<Vec<f32>>, BenchError> {
    read_vecs_from_reader(reader, limit, |buf, rec_idx, elem_idx| {
        let val = f32::from_le_bytes(buf);
        if !val.is_finite() {
            return Err(BenchError::MalformedFile {
                reason: format!("non-finite f32 value at record {rec_idx}, element {elem_idx}"),
            });
        }
        Ok(val)
    })
}

/// Read `ivecs` records from any [`Read`] stream.
pub fn read_ivecs_from_reader<R: Read>(
    reader: &mut R,
    limit: Option<usize>,
) -> Result<Vec<Vec<i32>>, BenchError> {
    read_vecs_from_reader(reader, limit, |buf, _, _| Ok(i32::from_le_bytes(buf)))
}

fn read_vecs_from_reader<T, R, F>(
    reader: &mut R,
    limit: Option<usize>,
    mut parse_elem: F,
) -> Result<Vec<Vec<T>>, BenchError>
where
    R: Read,
    F: FnMut([u8; 4], usize, usize) -> Result<T, BenchError>,
{
    let mut records = Vec::new();
    let mut expected_dim: Option<usize> = None;

    while limit.map_or(true, |max| records.len() < max) {
        let Some(dim_raw) = read_header_dim(reader, records.len())? else {
            break;
        };

        if dim_raw <= 0 {
            return Err(BenchError::MalformedFile {
                reason: format!("invalid dimension {dim_raw} at record {}", records.len()),
            });
        }

        let dim = dim_raw as usize;
        if dim > MAX_DIM {
            return Err(BenchError::MalformedFile {
                reason: format!(
                    "dimension {dim} exceeds maximum supported {MAX_DIM} at record {}",
                    records.len()
                ),
            });
        }

        match expected_dim {
            Some(exp) if exp != dim => {
                return Err(BenchError::MalformedFile {
                    reason: format!(
                        "inconsistent dimension at record {}: expected {exp}, got {dim}",
                        records.len()
                    ),
                });
            }
            None => expected_dim = Some(dim),
            _ => {}
        }

        let mut vec = Vec::with_capacity(dim);
        let rec_idx = records.len();
        for elem_idx in 0..dim {
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf).map_err(|e| {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    BenchError::MalformedFile {
                        reason: format!(
                            "truncated vector payload at record {rec_idx}, element {elem_idx}/{dim}"
                        ),
                    }
                } else {
                    BenchError::Io(e)
                }
            })?;
            vec.push(parse_elem(buf, rec_idx, elem_idx)?);
        }
        records.push(vec);
    }

    if records.is_empty() && limit != Some(0) {
        return Err(BenchError::MalformedFile {
            reason: "file contains no vectors".into(),
        });
    }

    Ok(records)
}

/// Reads a 4-byte little-endian `i32` dimension header.
///
/// Returns `Ok(None)` only on a clean EOF (0 bytes remaining).
/// Returns `Err(MalformedFile)` if 1..=3 bytes were read before EOF.
fn read_header_dim<R: Read>(reader: &mut R, rec_idx: usize) -> Result<Option<i32>, BenchError> {
    let mut buf = [0u8; 4];
    let mut read = 0;
    while read < 4 {
        match reader.read(&mut buf[read..]) {
            Ok(0) => {
                if read == 0 {
                    return Ok(None);
                }
                return Err(BenchError::MalformedFile {
                    reason: format!(
                        "truncated dimension header at record {rec_idx} ({read}/4 bytes)"
                    ),
                });
            }
            Ok(n) => read += n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(BenchError::Io(e)),
        }
    }
    Ok(Some(i32::from_le_bytes(buf)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Cursor;

    fn encode_fvecs(vecs: &[Vec<f32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vecs {
            out.extend_from_slice(&(v.len() as i32).to_le_bytes());
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    fn encode_ivecs(vecs: &[Vec<i32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vecs {
            out.extend_from_slice(&(v.len() as i32).to_le_bytes());
            for &x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn fvecs_round_trip() {
        let input = vec![vec![1.0_f32, -2.5, 3.25], vec![0.0, 42.0, -0.125]];
        let bytes = encode_fvecs(&input);
        let parsed = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap();
        assert_eq!(parsed, input);
    }

    #[test]
    fn ivecs_round_trip() {
        let input = vec![vec![0_i32, 5, 10], vec![3, 7, 11]];
        let bytes = encode_ivecs(&input);
        let parsed = read_ivecs_from_reader(&mut Cursor::new(bytes), None).unwrap();
        assert_eq!(parsed, input);
    }

    #[test]
    fn rejects_empty_file() {
        let err = read_fvecs_from_reader(&mut Cursor::new(Vec::<u8>::new()), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_truncated_header() {
        let err = read_fvecs_from_reader(&mut Cursor::new(vec![1_u8, 2]), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_truncated_payload() {
        let mut bytes = encode_fvecs(&[vec![1.0, 2.0, 3.0]]);
        bytes.truncate(bytes.len() - 3);
        let err = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_non_positive_dimension() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0_i32.to_le_bytes());
        let err = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));

        let mut bytes_neg = Vec::new();
        bytes_neg.extend_from_slice(&(-4_i32).to_le_bytes());
        let err_neg = read_ivecs_from_reader(&mut Cursor::new(bytes_neg), None).unwrap_err();
        assert!(matches!(err_neg, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_inconsistent_dimension() {
        let mut bytes = encode_fvecs(&[vec![1.0, 2.0]]);
        bytes.extend_from_slice(&3_i32.to_le_bytes());
        bytes.extend_from_slice(&1.0_f32.to_le_bytes());
        bytes.extend_from_slice(&2.0_f32.to_le_bytes());
        bytes.extend_from_slice(&3.0_f32.to_le_bytes());
        let err = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_nan_in_fvecs() {
        let bytes = encode_fvecs(&[vec![1.0, f32::NAN]]);
        let err = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    #[test]
    fn rejects_negative_groundtruth_id() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad_gt.ivecs");
        std::fs::write(&path, encode_ivecs(&[vec![0, -1, 2]])).unwrap();
        let err = read_groundtruth_ivecs(&path, None).unwrap_err();
        assert!(matches!(err, BenchError::MalformedFile { .. }));
    }

    proptest! {
        #[test]
        fn prop_fvecs_round_trip(
            n in 1usize..=20,
            dim in 1usize..=32,
            vals in prop::collection::vec(-1000.0f32..1000.0f32, 1..=640)
        ) {
            let mut vecs = Vec::with_capacity(n);
            for i in 0..n {
                let mut v = Vec::with_capacity(dim);
                for j in 0..dim {
                    v.push(vals[(i * dim + j) % vals.len()]);
                }
                vecs.push(v);
            }
            let bytes = encode_fvecs(&vecs);
            let parsed = read_fvecs_from_reader(&mut Cursor::new(bytes), None).unwrap();
            prop_assert_eq!(parsed, vecs);
        }
    }
}
