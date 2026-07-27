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
- safe versioned dense/sparse operator archives and ordered basis manifests.

All forms implement or adapt to one rectangular `LinearOperator` interface.

## Solvers and dynamics

- complete dense Hermitian eigendecomposition;
- extremal and shift-targeted Hermitian eigenpairs;
- restarted and fully reorthogonalized Lanczos;
- reusable eigensolver and shift-invert workspaces;
- exponential action, Krylov propagation, callable drives, FTLM, and LTLM;
- Floquet unitaries, complete dense spectra, and matrix-free selected
  quasienergies.

## Measurements and workflows

- expectation values and fluctuations;
- pure and mixed partial traces;
- sector-native binary partial traces that do not enumerate the unrestricted
  parent Hilbert space;
- fermionic and general exchange phases for arbitrary subsystems;
- entanglement spectra and entropies;
- spectral functions and dynamical correlators;
- diagonal ensembles and level statistics;
- state/subspace tracking and Lindblad generators.

## Automatic differentiation

- exact runtime-generic JVP and one-shot VJP for `A(θ)x`;
- real and complex parameter domains with stable named schemas;
- gap-aware Hellmann--Feynman gradients for isolated ground-state energies;
- residual, spectral-gap, action-count, and recomputation diagnostics;
- optional `chainrules-core 0.2` JVP/VJP adapters that call the native rules.

The current AD surface does not yet cover eigenvector gauges, degenerate
eigenspaces, time evolution, thermal traces, Floquet maps, or response
functions. Those workflows require their own operation-level rules; passing
the implemented ground-state benchmarks is not evidence that these gaps are
closed.

## Current execution boundary

The built-in runtime is single-rank CPU. Independent vector batches can use
bounded shared-memory throughput. GPU and multi-rank MPI profiles are explicit
extension points and currently return an error instead of silently falling
back to CPU.

Full Floquet spectra materialize a dense unitary and therefore retain
quadratic memory and cubic dense diagonalization scaling. Workflows needing
only a few quasienergies can instead use `Floquet::selected_eigensystem`,
which applies the period map and its adjoint without constructing the dense
unitary.
