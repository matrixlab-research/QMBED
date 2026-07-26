use qmbed::basis::SpinBasis1D;
use qmbed::operator::{
    Coupling, LocalOperator, MatrixFormat, OpProduct, OperatorBuilder, OperatorSpec,
};
use qmbed::runtime::{CpuRuntime, ExecutionProfile, Runtime, RuntimeLinearOperator};
use qmbed::{Complex64, QmbedError};

#[test]
fn cpu_runtime_applies_the_same_scientific_operator() {
    let basis = SpinBasis1D::builder(1).pauli(true).build().unwrap();
    let x = OpProduct::new([LocalOperator::X]).unwrap();
    let operator = OperatorBuilder::on(&basis)
        .term(OperatorSpec::from_product(x, [Coupling::new(1.0, vec![0])]).unwrap())
        .build(MatrixFormat::Csc)
        .unwrap();
    let runtime = CpuRuntime::new(2).unwrap();
    let input = runtime
        .upload(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
        .unwrap();
    let mut output = runtime.zeros(2).unwrap();

    operator.apply_on(&runtime, &input, &mut output).unwrap();

    assert_eq!(
        runtime.to_host(&output).unwrap(),
        vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]
    );
    assert_eq!(runtime.capabilities().threads_per_rank, 2);
}

#[test]
fn accelerator_and_distributed_requests_never_silently_fall_back() {
    let gpu = ExecutionProfile::local_gpu(0, 4);
    let mpi = ExecutionProfile::distributed_cpu(4, 8);
    assert!(matches!(
        CpuRuntime::from_profile(gpu),
        Err(QmbedError::UnsupportedBackend(_))
    ));
    assert!(matches!(
        CpuRuntime::from_profile(mpi),
        Err(QmbedError::UnsupportedBackend(_))
    ));
}

#[test]
fn cpu_vector_primitives_have_one_runtime_contract() {
    let runtime = CpuRuntime::new(1).unwrap();
    let left = runtime
        .upload(&[Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)])
        .unwrap();
    let mut right = runtime
        .upload(&[Complex64::new(0.0, 1.0), Complex64::new(1.0, 0.0)])
        .unwrap();
    runtime
        .axpy(Complex64::new(0.5, 0.0), &left, &mut right)
        .unwrap();
    assert_eq!(
        runtime.to_host(&right).unwrap(),
        vec![Complex64::new(0.5, 1.5), Complex64::new(2.0, 0.0)]
    );
    assert!((runtime.norm(&left).unwrap() - 6.0_f64.sqrt()).abs() < 1.0e-12);
}

#[test]
fn profiled_parallel_map_preserves_input_and_error_order() {
    let runtime = CpuRuntime::from_profile(ExecutionProfile::throughput(4)).unwrap();
    let values = [5_i32, 4, 3, 2, 1];
    let doubled = runtime.map_ordered(&values, |value| Ok(value * 2)).unwrap();
    assert_eq!(doubled, [10, 8, 6, 4, 2]);

    let error = runtime
        .map_ordered(&values, |value| {
            if *value == 4 || *value == 2 {
                Err(QmbedError::InvalidOptions(format!("failed at {value}")))
            } else {
                Ok(*value)
            }
        })
        .unwrap_err();
    assert_eq!(
        error.to_string(),
        QmbedError::InvalidOptions("failed at 4".into()).to_string()
    );
}
