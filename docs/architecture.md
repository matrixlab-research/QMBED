# Architecture

QMBED has two narrow interfaces.

## Scientific narrow waist

`LinearOperator` is the boundary between a physical model and a numerical
algorithm. A Hamiltonian may be dense, sparse, parameterized, projected, block
structured, or matrix free; eigensolvers and dynamics consume the same action.

```text
basis + typed local terms
          │
          ▼
 universal operator assembly
          │
          ▼
     LinearOperator
       ├─ eigensolvers
       ├─ time evolution
       ├─ response functions
       └─ measurements
```

This is why optimization is capability based. For example, a finite-orbit
cache is selected because a basis supplies a finite symmetry action, and a
fixed sparse pattern is reused because a parameterized family shares its
nonzero structure. No Ising-, Hubbard-, or PXP-specific path is required.

## Execution narrow waist

`Runtime` owns vector storage and coarse operations such as apply, `axpy`,
dot products, and host transfer. The CPU implementation is complete.
Accelerator and MPI support can be added behind this boundary without exposing
backend buffers in basis or operator APIs.

## Language boundary

Rust calls the generic core directly. Python and Julia send the same typed
request schema through the C ABI:

```text
Python ─┐
        ├─ typed JSON + opaque handles ─ qmbed-capi ─ Rust core
Julia ──┘
```

Long-lived frontend objects use opaque handles for model, basis-plan, Krylov,
and exponential-action resources. Handles are safe to share at the protocol
boundary, never expose Rust addresses, and have explicit release operations.

Python's QuSpin compatibility surface translates old spellings into this
protocol. Julia intentionally exposes only the native model.

## Numerical backend

Dense eigendecomposition, dense products, and shifted sparse factorization are
isolated internally. Iterative algorithms call prepared factorization or
matrix kernels rather than redispatching within element or matvec loops. This
keeps future backend changes independent of scientific APIs.
