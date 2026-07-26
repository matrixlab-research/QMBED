use std::sync::Arc;

use qmbed::basis::{Basis, BasisProjector, SpinBasis1D, U256, UserBasis};
use qmbed::dynamics::{DriveStep, Floquet, FloquetSpectrumOptions};
use qmbed::measure::{EntropyOrder, entanglement_entropy, entanglement_entropy_sector};
use qmbed::operator::{
    Coupling, LinearOperator, MatrixFormat, Operator, OperatorBuilder, OperatorSpec,
    QuantumComponent, QuantumOperator,
};
use qmbed::solve::{EigshOptions, EigshWorkspace, EvolutionOptions, eigsh_with_workspace, evolve};
use qmbed::{Complex64, Result};

fn spin_operator(basis: &SpinBasis1D, label: &str, coefficient: f64) -> Result<Operator> {
    let sites = basis.sites();
    let couplings: Vec<Coupling> = match label {
        "zz" => (0..sites)
            .map(|site| Coupling::new(coefficient, vec![site, (site + 1) % sites]))
            .collect(),
        "x" | "z" => (0..sites)
            .map(|site| Coupling::new(coefficient, vec![site]))
            .collect(),
        _ => unreachable!("the example uses only zz, x, and z"),
    };
    OperatorBuilder::on(basis)
        .term(OperatorSpec::new(label, couplings)?)
        .build(MatrixFormat::Csc)
}

fn parameter_scan() -> Result<()> {
    let basis = SpinBasis1D::builder(8).build()?;
    let family = QuantumOperator::new([
        QuantumComponent::with_default("interaction", spin_operator(&basis, "zz", 1.0)?, 1.0),
        QuantumComponent::required("field", spin_operator(&basis, "z", 1.0)?),
    ])?;
    let mut plan = family.plan(MatrixFormat::Csc)?;
    let mut workspace = EigshWorkspace::new();

    for field in [0.4, 0.8, 1.2] {
        let hamiltonian =
            plan.evaluate_coefficients(&[Complex64::new(1.0, 0.0), Complex64::new(field, 0.0)])?;
        let spectrum = eigsh_with_workspace(
            hamiltonian,
            EigshOptions::smallest_algebraic(2).with_tolerance(1.0e-9),
            &mut workspace,
        )?;
        println!("field={field:.1}, ground={:.8}", spectrum.eigenvalues[0]);
    }
    Ok(())
}

fn symmetry_projection() -> Result<()> {
    let parent = SpinBasis1D::builder(6).up(3).build()?;
    let reduced = SpinBasis1D::builder(6).up(3).momentum(0).build()?;
    let projector = BasisProjector::between(&reduced, &parent)?;
    let mut coordinate = vec![Complex64::new(0.0, 0.0); reduced.len()];
    coordinate[0] = Complex64::new(1.0, 0.0);
    let lifted = projector.lifted(&coordinate)?;
    assert_eq!(lifted.len(), parent.len());
    Ok(())
}

fn dynamics_and_measurement() -> Result<()> {
    let basis = SpinBasis1D::builder(1).pauli(true).build()?;
    let x = spin_operator(&basis, "x", 1.0)?;
    let z = spin_operator(&basis, "z", 1.0)?;
    let initial = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];

    let trajectory = evolve(
        &x,
        &initial,
        EvolutionOptions::new([0.0, 0.2, 0.4]).with_tolerance(1.0e-11),
    )?;
    assert_eq!(trajectory.states.len(), 3);

    let floquet = Floquet::new([
        DriveStep::new(Arc::new(z), 0.2)?,
        DriveStep::new(Arc::new(x), 0.3)?,
    ])?;
    let analysis = floquet.analyze(MatrixFormat::Dense)?;
    assert_eq!(analysis.eigensystem.quasienergies.len(), 2);
    let selected = floquet.selected_eigensystem(FloquetSpectrumOptions::new(1, 0.0))?;
    assert_eq!(selected.quasienergies.len(), 1);

    let scale = 1.0 / 2.0_f64.sqrt();
    let bell = [
        Complex64::new(scale, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(0.0, 0.0),
        Complex64::new(scale, 0.0),
    ];
    let entropy = entanglement_entropy(&bell, 2, 2, EntropyOrder::VonNeumann)?;
    assert!((entropy - 2.0_f64.ln()).abs() < 1.0e-12);
    Ok(())
}

fn custom_and_wide_states() -> Result<()> {
    let constrained = UserBasis::<u128>::builder(6)
        .state_filter(|state| state & (state << 1) == 0)?
        .operator('n', |state, site| {
            let occupied = ((state >> site) & 1) as f64;
            Ok(Some((state, Complex64::new(occupied, 0.0))))
        })
        .build()?;
    let number = OperatorBuilder::on(&constrained)
        .term(OperatorSpec::new(
            "n",
            (0..constrained.sites()).map(|site| Coupling::new(1.0, vec![site])),
        )?)
        .build(MatrixFormat::Csc)?;
    assert_eq!(number.shape().0, constrained.len());

    let state = U256::zero().with_bit(200, true)?;
    assert!(state.bit(200)?);
    let partner = U256::zero().with_bit(1, true)?.with_bit(199, true)?;
    let scale = 1.0 / 2.0_f64.sqrt();
    let entropy = entanglement_entropy_sector(
        &[Complex64::new(scale, 0.0), Complex64::new(scale, 0.0)],
        &[state, partner],
        201,
        &[200],
        &[],
        EntropyOrder::VonNeumann,
    )?;
    assert!((entropy - 2.0_f64.ln()).abs() < 1.0e-12);
    Ok(())
}

fn main() -> Result<()> {
    parameter_scan()?;
    symmetry_projection()?;
    dynamics_and_measurement()?;
    custom_and_wide_states()
}
