# QMBED

QMBED is a Rust-native exact-diagonalization toolkit for quantum many-body
workflows. Its three interfaces share one implementation:

<div class="grid cards" markdown>

-   :material-language-rust:{ .lg .middle } **Rust**

    ---

    The complete typed core for bases, operators, solvers, dynamics, and
    measurements.

    [:octicons-arrow-right-24: Rust guide](rust/)

-   :material-language-python:{ .lg .middle } **Python**

    ---

    A compact native API plus a versioned compatibility surface for migrating
    QuSpin programs.

    [:octicons-arrow-right-24: Python guide](python/)

-   :material-language-julia:{ .lg .middle } **Julia**

    ---

    A deliberately native interface over the same Rust core, without a second
    QuSpin-shaped API.

    [:octicons-arrow-right-24: Julia guide](julia/)

</div>

## One scientific core

All interfaces construct typed basis and operator requests, then enter the
same Rust implementation. Storage and execution choices are based on
mathematical capabilities rather than model names.

```text
language request → basis → operator → LinearOperator → solver/dynamics → observable
```

This keeps correctness and performance improvements reusable: a better sparse
assembler, symmetry cache, or Lanczos workspace benefits every language.

## Evidence boundary

Repository CI verifies public behavior and all language bindings. The separate
[QMBED Benchmark](https://github.com/matrixlab-research/QMBED-benchmark)
repository owns independent numerical oracles and twelve medium-size,
paper-shaped workflows. API presence, numerical parity, complete workflow
composition, and representative scale are reported separately.

Start with [Get started](getting-started.md), or read
[Architecture](architecture.md) to understand the shared-core design.
