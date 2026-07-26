# Python interface

The Python distribution contains two intentionally different surfaces:

| Surface | Import | Use it when |
|---|---|---|
| Native QMBED | `import qmbed` | Writing new code around typed bases and operators |
| QuSpin compatibility | `import quspin` | Migrating or validating an existing QuSpin workflow |

Both surfaces call the same Rust core through `qmbed-capi`. The compatibility
layer translates QuSpin objects and spellings; it is not a second numerical
backend.

## Native API

The native API follows the same sequence as Rust:

1. choose a basis;
2. construct typed local operator products and couplings;
3. choose solver options;
4. call a QMBED operation.

See [Get started](../getting-started.md#python) for an executable example and
the [Python API reference](api.md) for signatures.

## Compatibility API

The package includes an import-compatible `quspin` namespace. CI runs the
frozen QuSpin 1.0.1 test snapshot from upstream commit
`5bf9e5b266e6d8b70e5cf5973c7c7d59d62e412f` unchanged. Passing that suite is a
versioned migration contract, not a claim that every incidental warning,
dtype identity, SciPy keyword, or future QuSpin API is reproduced.

Read [QuSpin compatibility](quspin-compat.md) for the exact boundary.

## Native library lookup

The binding checks `QMBED_LIBRARY_PATH` first. In a source checkout it then
looks for the platform library under:

```text
bindings/capi/target/release/
bindings/capi/target/debug/
```

Build the shared boundary before the first numerical call:

```bash
cargo build --release --manifest-path bindings/capi/Cargo.toml
```
