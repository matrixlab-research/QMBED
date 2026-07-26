# Verification

QMBED separates four questions that are often conflated:

1. does a public API exist;
2. does it return the right numerical result;
3. can it compose the full scientific workflow;
4. is the production route usable at representative scale?

## Repository CI

| Workflow | What it checks |
|---|---|
| `CI` | Rust formatting, Clippy, unit/integration tests, paper-scale visible contracts |
| `language-bindings` | C ABI, Python native and compatibility tests, Julia tests |
| `documentation` | strict MkDocs, rustdoc, Documenter, and assembled-site links |
| `release` | synchronized versions, packages, bindings, and tagged release gate |

The badges in the project README point directly to these workflows. A green
badge means the corresponding workflow passed on `main`; it is not a broader
claim than that workflow's checks.

## Independent benchmark

[QMBED Benchmark](https://github.com/matrixlab-research/QMBED-benchmark)
contains independent numerical oracles and twelve paper-shaped tasks:

- MBL shift-invert;
- XXZ Lanczos quench;
- Floquet full-unitary heating;
- spinful Hubbard and interacting SSH spectra;
- translation-sector XXZ;
- TFIM subspace fidelity;
- PXP and Bose–Hubbard quenches;
- Hubbard current;
- CoNb₂O₆ dynamical structure factor;
- cross-sector particle-addition spectrum.

Each timed case constructs its basis and Hamiltonian lazily, checks a physical
invariant, performs one warm-up, and records five samples on a common
single-thread runner. These workflows are literature anchored but are not
complete paper reproductions.

## Local gates

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
cargo test --release --test visible_contract -- --ignored --test-threads=1
```

For bindings:

```bash
cargo test --manifest-path bindings/capi/Cargo.toml
cargo build --release --manifest-path bindings/capi/Cargo.toml
python -m pip install -e "bindings/python[test]"
python -m unittest discover -s bindings/python/tests -v
julia --project=bindings/julia -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```
