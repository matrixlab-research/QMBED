use approx::assert_abs_diff_eq;
use qmbed::ad::{
    GradientStatus, ParameterDirection, ParameterDomain, ParameterSchema, ParameterValues,
    apply_jvp, apply_vjp, ground_state_energy_gradient,
};
use qmbed::operator::{MatrixFormat, Operator, QuantumComponent, QuantumOperator};
use qmbed::runtime::{CpuRuntime, Runtime};
use qmbed::solve::{EigshOptions, EigshWorkspace, eigsh};
use qmbed::{Complex64, QmbedError};

fn complex_matrix(entries: [Complex64; 4]) -> Operator {
    Operator::from_dense(2, 2, entries.to_vec()).unwrap()
}

fn operator_family() -> QuantumOperator {
    let first = complex_matrix([
        Complex64::new(1.0, 0.2),
        Complex64::new(0.3, -0.1),
        Complex64::new(-0.4, 0.6),
        Complex64::new(0.7, 0.0),
    ]);
    let second = complex_matrix([
        Complex64::new(0.2, -0.3),
        Complex64::new(-0.5, 0.2),
        Complex64::new(0.8, 0.1),
        Complex64::new(-0.6, 0.4),
    ]);
    QuantumOperator::new([
        QuantumComponent::required("first", first),
        QuantumComponent::required("second", second),
    ])
    .unwrap()
}

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert_abs_diff_eq!(actual.re, expected.re, epsilon = tolerance);
    assert_abs_diff_eq!(actual.im, expected.im, epsilon = tolerance);
}

fn assert_vector_close(actual: &[Complex64], expected: &[Complex64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (&actual, &expected) in actual.iter().zip(expected) {
        assert_complex_close(actual, expected, tolerance);
    }
}

fn real_pair(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| (left.conj() * right).re)
        .sum()
}

#[test]
fn native_jvp_matches_directional_finite_difference() {
    let operator = operator_family();
    let parameters = ParameterValues::complex(
        &operator,
        [Complex64::new(0.7, 0.2), Complex64::new(-0.4, 0.5)],
    )
    .unwrap();
    let parameter_direction = parameters
        .direction([Complex64::new(0.3, -0.2), Complex64::new(-0.1, 0.4)])
        .unwrap();
    let state = [Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.8)];
    let state_direction = [Complex64::new(-0.4, 0.1), Complex64::new(0.5, -0.2)];

    let runtime = CpuRuntime::new(1).unwrap();
    let runtime_state = runtime.upload(&state).unwrap();
    let runtime_state_direction = runtime.upload(&state_direction).unwrap();
    let result = apply_jvp(
        &runtime,
        &operator,
        &parameters,
        &parameter_direction,
        &runtime_state,
        &runtime_state_direction,
    )
    .unwrap();

    let mut expected_value = vec![Complex64::new(0.0, 0.0); 2];
    operator
        .apply_coefficients(parameters.values(), &state, &mut expected_value)
        .unwrap();
    assert_vector_close(
        &runtime.to_host(&result.value).unwrap(),
        &expected_value,
        1.0e-12,
    );

    let step = 1.0e-6;
    let positive_parameters: Vec<_> = parameters
        .values()
        .iter()
        .zip(parameter_direction.values())
        .map(|(value, direction)| *value + step * *direction)
        .collect();
    let negative_parameters: Vec<_> = parameters
        .values()
        .iter()
        .zip(parameter_direction.values())
        .map(|(value, direction)| *value - step * *direction)
        .collect();
    let positive_state: Vec<_> = state
        .iter()
        .zip(state_direction)
        .map(|(value, direction)| *value + step * direction)
        .collect();
    let negative_state: Vec<_> = state
        .iter()
        .zip(state_direction)
        .map(|(value, direction)| *value - step * direction)
        .collect();
    let mut positive = vec![Complex64::new(0.0, 0.0); 2];
    let mut negative = vec![Complex64::new(0.0, 0.0); 2];
    operator
        .apply_coefficients(&positive_parameters, &positive_state, &mut positive)
        .unwrap();
    operator
        .apply_coefficients(&negative_parameters, &negative_state, &mut negative)
        .unwrap();
    let numerical: Vec<_> = positive
        .iter()
        .zip(negative)
        .map(|(positive, negative)| (*positive - negative) / (2.0 * step))
        .collect();

    assert_vector_close(
        &runtime.to_host(&result.tangent).unwrap(),
        &numerical,
        2.0e-9,
    );
    assert_eq!(result.diagnostics.primal_applications, 4);
    assert_eq!(result.diagnostics.backward_applications, 0);
}

#[test]
fn native_vjp_satisfies_complex_adjoint_identity() {
    let operator = operator_family();
    let parameters = ParameterValues::complex(
        &operator,
        [Complex64::new(0.7, 0.2), Complex64::new(-0.4, 0.5)],
    )
    .unwrap();
    let parameter_direction = parameters
        .direction([Complex64::new(0.3, -0.2), Complex64::new(-0.1, 0.4)])
        .unwrap();
    let state = [Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.8)];
    let state_direction = [Complex64::new(-0.4, 0.1), Complex64::new(0.5, -0.2)];
    let output_cotangent = [Complex64::new(-0.7, 0.9), Complex64::new(0.25, -0.35)];

    let runtime = CpuRuntime::new(1).unwrap();
    let runtime_state = runtime.upload(&state).unwrap();
    let runtime_state_direction = runtime.upload(&state_direction).unwrap();
    let runtime_output_cotangent = runtime.upload(&output_cotangent).unwrap();

    let jvp = apply_jvp(
        &runtime,
        &operator,
        &parameters,
        &parameter_direction,
        &runtime_state,
        &runtime_state_direction,
    )
    .unwrap();
    let (_, pullback) = apply_vjp(&runtime, &operator, &parameters, &runtime_state).unwrap();
    let cotangents = pullback.backward(&runtime_output_cotangent).unwrap();

    let output_tangent = runtime.to_host(&jvp.tangent).unwrap();
    let state_cotangent = runtime.to_host(&cotangents.state).unwrap();
    let left = real_pair(&output_tangent, &output_cotangent);
    let parameter_pairing: f64 = parameter_direction
        .values()
        .iter()
        .zip(cotangents.parameters.values())
        .map(|(direction, gradient)| (direction.conj() * gradient).re)
        .sum();
    let right = real_pair(&state_direction, &state_cotangent) + parameter_pairing;

    assert_abs_diff_eq!(left, right, epsilon = 1.0e-11);
    assert_eq!(cotangents.diagnostics.primal_applications, 2);
    assert_eq!(cotangents.diagnostics.backward_applications, 4);
}

#[test]
fn real_parameters_project_cotangents_and_reject_complex_directions() {
    let operator = operator_family();
    let parameters = ParameterValues::real(&operator, [0.7, -0.4]).unwrap();
    assert!(matches!(
        parameters.direction([Complex64::new(0.2, 0.0), Complex64::new(0.0, 0.3)]),
        Err(QmbedError::InvalidOptions(_))
    ));
    let state = [Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.8)];
    let output_cotangent = [Complex64::new(-0.7, 0.9), Complex64::new(0.25, -0.35)];
    let runtime = CpuRuntime::new(1).unwrap();
    let runtime_state = runtime.upload(&state).unwrap();
    let runtime_output_cotangent = runtime.upload(&output_cotangent).unwrap();
    let (_, pullback) = apply_vjp(&runtime, &operator, &parameters, &runtime_state).unwrap();
    let cotangents = pullback.backward(&runtime_output_cotangent).unwrap();
    assert!(
        cotangents
            .parameters
            .values()
            .iter()
            .all(|gradient| gradient.im == 0.0)
    );
}

#[test]
fn schema_mismatch_is_rejected_before_operator_work() {
    let operator = operator_family();
    let schema = ParameterSchema::new(
        ["wrong".to_string(), "second".to_string()],
        [ParameterDomain::Complex, ParameterDomain::Complex],
    )
    .unwrap();
    let parameters = ParameterValues::new(
        schema.into(),
        [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
    )
    .unwrap();
    let runtime = CpuRuntime::new(1).unwrap();
    let state = runtime
        .upload(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)])
        .unwrap();
    assert!(matches!(
        apply_vjp(&runtime, &operator, &parameters, &state),
        Err(QmbedError::InvalidOptions(_))
    ));
}

#[test]
fn ground_state_energy_gradient_matches_central_finite_difference() {
    let diagonal = |values: [f64; 4]| {
        let mut matrix = vec![Complex64::new(0.0, 0.0); 16];
        for (index, value) in values.into_iter().enumerate() {
            matrix[4 * index + index] = Complex64::new(value, 0.0);
        }
        Operator::from_dense(4, 4, matrix).unwrap()
    };
    let operator = QuantumOperator::new([
        QuantumComponent::required("offset", diagonal([1.0, 0.0, 0.0, 0.0])),
        QuantumComponent::required("field", diagonal([-0.5, 0.3, 0.7, 1.1])),
        QuantumComponent::required("base", diagonal([0.0, 1.0, 2.0, 3.0])),
    ])
    .unwrap();
    let parameters = ParameterValues::real(&operator, [0.4, 0.8, 1.0]).unwrap();
    let options = EigshOptions::smallest_algebraic(2);
    let result = ground_state_energy_gradient(
        &operator,
        &parameters,
        options.clone(),
        &mut EigshWorkspace::new(),
    )
    .unwrap();
    assert_eq!(result.diagnostics.status, GradientStatus::Reliable);
    assert!(result.diagnostics.spectral_gap.unwrap() > 0.0);

    let step = 1.0e-6;
    for parameter in 0..parameters.values().len() {
        let mut positive = parameters.values().to_vec();
        let mut negative = parameters.values().to_vec();
        positive[parameter].re += step;
        negative[parameter].re -= step;
        let evaluate = |coefficients: &[Complex64]| {
            let values = operator
                .component_names()
                .zip(coefficients)
                .map(|(name, value)| (name.to_string(), *value))
                .collect();
            let matrix = operator.evaluate(&values, MatrixFormat::Csc).unwrap();
            eigsh(&matrix, options.clone()).unwrap().eigenvalues[0]
        };
        let numerical = (evaluate(&positive) - evaluate(&negative)) / (2.0 * step);
        assert_abs_diff_eq!(
            result.gradient.values()[parameter].re,
            numerical,
            epsilon = 1.0e-8
        );
    }
}

#[test]
fn ground_state_energy_gradient_marks_an_unresolved_degeneracy() {
    let diagonal = |values: [f64; 4]| {
        let mut matrix = vec![Complex64::new(0.0, 0.0); 16];
        for (index, value) in values.into_iter().enumerate() {
            matrix[4 * index + index] = Complex64::new(value, 0.0);
        }
        Operator::from_dense(4, 4, matrix).unwrap()
    };
    let operator = QuantumOperator::new([
        QuantumComponent::required("degenerate", diagonal([0.0, 0.0, 1.0, 2.0])),
        QuantumComponent::required("identity", diagonal([1.0, 1.0, 1.0, 1.0])),
    ])
    .unwrap();
    let parameters = ParameterValues::real(&operator, [1.0, 0.2]).unwrap();
    let result = ground_state_energy_gradient(
        &operator,
        &parameters,
        EigshOptions::smallest_algebraic(2),
        &mut EigshWorkspace::new(),
    )
    .unwrap();
    assert_eq!(result.diagnostics.status, GradientStatus::NonDifferentiable);
}

#[cfg(feature = "chainrules")]
#[test]
fn chainrules_adapter_observes_the_native_jvp_and_vjp() {
    use chainrules_core::{JvpRule, Pullback, VjpRule};
    use qmbed::ad::chainrules::{ApplyArgumentTangent, ApplyArguments, ApplyRule};

    let operator = operator_family();
    let parameters = ParameterValues::complex(
        &operator,
        [Complex64::new(0.7, 0.2), Complex64::new(-0.4, 0.5)],
    )
    .unwrap();
    let parameter_direction: ParameterDirection = parameters
        .direction([Complex64::new(0.3, -0.2), Complex64::new(-0.1, 0.4)])
        .unwrap();
    let runtime = CpuRuntime::new(1).unwrap();
    let state = runtime
        .upload(&[Complex64::new(0.6, -0.3), Complex64::new(-0.2, 0.8)])
        .unwrap();
    let state_direction = runtime
        .upload(&[Complex64::new(-0.4, 0.1), Complex64::new(0.5, -0.2)])
        .unwrap();
    let output_cotangent = runtime
        .upload(&[Complex64::new(-0.7, 0.9), Complex64::new(0.25, -0.35)])
        .unwrap();
    let arguments = ApplyArguments {
        parameters: &parameters,
        state: &state,
    };
    let tangent = ApplyArgumentTangent {
        parameters: &parameter_direction,
        state: &state_direction,
    };
    let rule = ApplyRule {
        runtime: &runtime,
        operator: &operator,
    };

    let (rule_value, rule_jvp) = rule.jvp(&arguments, &tangent).unwrap();
    let native = apply_jvp(
        &runtime,
        &operator,
        &parameters,
        &parameter_direction,
        &state,
        &state_direction,
    )
    .unwrap();
    assert_vector_close(
        &runtime.to_host(&rule_value.into_inner()).unwrap(),
        &runtime.to_host(&native.value).unwrap(),
        1.0e-12,
    );
    assert_vector_close(
        &runtime.to_host(&rule_jvp).unwrap(),
        &runtime.to_host(&native.tangent).unwrap(),
        1.0e-12,
    );

    let (_, pullback) = rule.vjp(&arguments).unwrap();
    let rule_cotangents = pullback.apply(output_cotangent).unwrap();
    let (_, native_pullback) = apply_vjp(&runtime, &operator, &parameters, &state).unwrap();
    let native_output_cotangent = runtime
        .upload(&[Complex64::new(-0.7, 0.9), Complex64::new(0.25, -0.35)])
        .unwrap();
    let native_cotangents = native_pullback.backward(&native_output_cotangent).unwrap();
    assert_vector_close(
        &runtime.to_host(&rule_cotangents.state).unwrap(),
        &runtime.to_host(&native_cotangents.state).unwrap(),
        1.0e-12,
    );
    assert_vector_close(
        rule_cotangents.parameters.values(),
        native_cotangents.parameters.values(),
        1.0e-12,
    );
}

#[cfg(feature = "chainrules")]
#[test]
fn chainrules_ground_energy_rule_reuses_the_native_gradient() {
    use chainrules_core::{JvpRule, Pullback, VjpRule};
    use qmbed::ad::chainrules::GroundStateEnergyRule;

    let diagonal = |values: [f64; 4]| {
        let mut matrix = vec![Complex64::new(0.0, 0.0); 16];
        for (index, value) in values.into_iter().enumerate() {
            matrix[4 * index + index] = Complex64::new(value, 0.0);
        }
        Operator::from_dense(4, 4, matrix).unwrap()
    };
    let operator = QuantumOperator::new([
        QuantumComponent::required("field", diagonal([-0.5, 0.3, 0.7, 1.1])),
        QuantumComponent::required("base", diagonal([0.0, 1.0, 2.0, 3.0])),
    ])
    .unwrap();
    let parameters = ParameterValues::real(&operator, [0.8, 1.0]).unwrap();
    let direction = parameters
        .direction([Complex64::new(0.4, 0.0), Complex64::new(-0.2, 0.0)])
        .unwrap();
    let rule = GroundStateEnergyRule {
        operator: &operator,
        options: EigshOptions::smallest_algebraic(2),
    };
    let native = ground_state_energy_gradient(
        &operator,
        &parameters,
        EigshOptions::smallest_algebraic(2),
        &mut EigshWorkspace::new(),
    )
    .unwrap();
    let (energy, directional) = rule.jvp(&parameters, &direction).unwrap();
    assert_abs_diff_eq!(energy, native.energy, epsilon = 1.0e-12);
    let expected_directional: f64 = native
        .gradient
        .values()
        .iter()
        .zip(direction.values())
        .map(|(gradient, direction)| (direction.conj() * gradient).re)
        .sum();
    assert_abs_diff_eq!(directional, expected_directional, epsilon = 1.0e-12);

    let (_, pullback) = rule.vjp(&parameters).unwrap();
    let scaled = pullback.apply(0.25).unwrap();
    for (actual, expected) in scaled.values().iter().zip(native.gradient.values()) {
        assert_complex_close(*actual, 0.25 * *expected, 1.0e-12);
    }
}
