use qmbed::Complex64;
use qmbed::archive::{
    BasisArchive, load_basis_zip, load_operator_npz, save_basis_zip, save_operator_npz,
};
use qmbed::basis::ErasedState;
use qmbed::operator::{MatrixFormat, Operator};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments: Vec<_> = std::env::args().collect();
    match arguments.as_slice() {
        [_, mode, path] if mode == "write" => {
            let operator = Operator::from_dense(
                2,
                2,
                vec![
                    Complex64::new(1.0, 0.0),
                    Complex64::new(0.25, -0.5),
                    Complex64::new(0.25, 0.5),
                    Complex64::new(-2.0, 0.0),
                ],
            )?;
            save_operator_npz(&operator, path)?;
        }
        [_, mode, path] if mode == "read" => {
            let operator = load_operator_npz(path, MatrixFormat::Dense)?;
            for value in operator.to_dense() {
                println!("{:.17},{:.17}", value.re, value.im);
            }
        }
        [_, mode, path] if mode == "write-basis" => {
            let mut basis = BasisArchive::new(
                256,
                ["0", "3", "7"]
                    .into_iter()
                    .map(|state| ErasedState::from_decimal(256, state))
                    .collect::<qmbed::Result<Vec<_>>>()?,
            )?;
            basis.insert_metadata("kind", "spin")?;
            save_basis_zip(path, &basis)?;
        }
        [_, mode, path] if mode == "read-basis" => {
            let basis = load_basis_zip(path)?;
            println!("width_bits={}", basis.width_bits());
            for state in basis.states() {
                println!("{}", state.to_decimal());
            }
        }
        _ => {
            return Err("usage: archive_interop <write|read|write-basis|read-basis> <path>".into());
        }
    }
    Ok(())
}
