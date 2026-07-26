use qmbed::QmbedError;

#[test]
fn qmbed_is_the_native_crate_identity() {
    assert_eq!(qmbed::VERSION, env!("CARGO_PKG_VERSION"));
    let error = QmbedError::InvalidOptions("example".into());
    assert_eq!(error.to_string(), "invalid options: example");
}
