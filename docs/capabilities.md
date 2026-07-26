# Capabilities

QMBED is organized around reusable mathematical operations.

## Bases and sectors

- spin, boson, spinless fermion, and spinful fermion bases;
- fixed particle or magnetization sectors;
- translation, parity, finite generated symmetry groups, and projectors;
- photon and tensor-product spaces;
- callback-defined local actions;
- `u128`, `U256`, `U1024`, `U4096`, and `U16384` state storage.

Fixed-sector enumeration scales with the requested combinatorial sector rather
than scanning the full parent Hilbert space. Symmetry reducers share cached
orbit traces across compatible constructions.

## Operators

- dense, CSC, CSR, DIA, and matrix-free forms;
- static, driven, and named parameterized Hamiltonians;
- rectangular source-to-target operators;
- reusable sparse matvec and parameter-scan plans;
- safe versioned dense and sparse archives.

All forms implement or adapt to one rectangular `LinearOperator` interface.

## Solvers and dynamics

- complete dense Hermitian eigendecomposition;
- extremal and shift-targeted Hermitian eigenpairs;
- restarted and fully reorthogonalized Lanczos;
- reusable eigensolver and shift-invert workspaces;
- exponential action, Krylov propagation, callable drives, FTLM, and LTLM;
- Floquet unitaries, quasienergies, and eigensystems.

## Measurements and workflows

- expectation values and fluctuations;
- pure and mixed partial traces;
- fermionic and general exchange phases for arbitrary subsystems;
- entanglement spectra and entropies;
- spectral functions and dynamical correlators;
- diagonal ensembles and level statistics;
- state/subspace tracking and Lindblad generators.

## Current execution boundary

The built-in runtime is single-rank CPU. Independent vector batches can use
bounded shared-memory throughput. GPU and multi-rank MPI profiles are explicit
extension points and currently return an error instead of silently falling
back to CPU.

Full Floquet spectra materialize a dense unitary and therefore retain
quadratic memory and cubic dense diagonalization scaling.
