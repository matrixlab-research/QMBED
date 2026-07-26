use std::time::{SystemTime, UNIX_EPOCH};

use approx::assert_abs_diff_eq;
use qmbed::Complex64;
use qmbed::archive::{
    ARCHIVE_VERSION, OperatorArchive, load_operator_npz, load_zip, save_operator_npz, save_zip,
};
use qmbed::operator::{LinearOperator, MatrixFormat, Operator};

#[test]
fn versioned_npz_round_trip_preserves_complex_values_and_requested_format() {
    assert_eq!(ARCHIVE_VERSION, 1);
    let operator = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.25, -0.5),
            Complex64::new(0.25, 0.5),
            Complex64::new(-2.0, 0.0),
        ],
    )
    .unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("qmbed-archive-{}-{nonce}.npz", std::process::id()));
    save_operator_npz(&operator, &path).unwrap();
    let loaded = load_operator_npz(&path, MatrixFormat::Csc).unwrap();
    assert_eq!(loaded.format(), MatrixFormat::Csc);
    for (actual, expected) in loaded.to_dense().iter().zip(operator.to_dense()) {
        assert_abs_diff_eq!(actual.re, expected.re, epsilon = 1.0e-15);
        assert_abs_diff_eq!(actual.im, expected.im, epsilon = 1.0e-15);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn named_archive_preserves_dense_sparse_names_formats_and_defaults() {
    let dense = Operator::from_dense(
        2,
        2,
        vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 2.0),
            Complex64::new(0.0, -2.0),
            Complex64::new(3.0, 0.0),
        ],
    )
    .unwrap();
    let sparse = Operator::from_triplets(
        2,
        2,
        [
            (0, 1, Complex64::new(0.25, 0.0)),
            (1, 0, Complex64::new(0.25, 0.0)),
        ],
        MatrixFormat::Csc,
    )
    .unwrap();
    let mut entries = OperatorArchive::new();
    entries
        .insert("dense", dense.clone(), Some(Complex64::new(1.0, 0.0)))
        .unwrap();
    entries.insert("sparse", sparse.clone(), None).unwrap();
    entries.insert_metadata("scalar_dtype", "float32").unwrap();
    let path = std::env::temp_dir().join(format!(
        "qmbed-rust-named-archive-{}.npz",
        std::process::id()
    ));
    save_zip(&path, &entries).unwrap();
    let restored = load_zip(&path).unwrap();
    assert_eq!(
        restored.get("dense").unwrap().operator.to_dense(),
        dense.to_dense()
    );
    assert_eq!(
        restored.get("dense").unwrap().default,
        Some(Complex64::new(1.0, 0.0))
    );
    assert_eq!(
        restored.get("sparse").unwrap().operator.format(),
        MatrixFormat::Csc
    );
    assert_eq!(
        restored.get("sparse").unwrap().operator.triplets(),
        sparse.triplets()
    );
    assert!(restored.get("sparse").unwrap().default.is_none());
    assert_eq!(restored.metadata_value("scalar_dtype"), Some("float32"));
    std::fs::remove_file(path).unwrap();
}
