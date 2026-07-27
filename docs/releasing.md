# Release process

QMBED has one version across four distribution surfaces:

- the `qmbed` Rust crate;
- the `qmbed` Python distribution;
- the `QMBED` Julia package; and
- the shared `qmbed-capi` native library.

`python ci/check_versions.py` rejects a release if those versions diverge.
The Python and Julia layers package the same C ABI; neither owns a second
solver implementation.

## One-time registry setup

Before creating the first public registry release:

1. Create a crates.io API token and save it as the
   `CARGO_REGISTRY_TOKEN` GitHub Actions secret.
2. Configure a PyPI trusted publisher for
   `matrixlab-research/QMBED`, workflow `release.yml`, environment `pypi`.
   The release job uses GitHub OIDC and does not store a PyPI password.
3. Install the JuliaRegistrator GitHub App for the repository.

The GitHub `pypi` environment should be limited to tagged releases and may
require a maintainer approval if desired.

## 1. Prepare native artifacts

Run **Prepare Julia native artifacts** on `main` with the intended version,
without a leading `v`.

The workflow builds the shared library on five architecture-native GitHub
runners. Linux linking occurs inside a `manylinux2014` container so the
artifact does not inherit Ubuntu 24.04's newer glibc requirement:

| Platform | Architecture |
|---|---|
| Linux glibc | x86_64, aarch64 |
| macOS | x86_64, aarch64 |
| Windows | x86_64 |

It packages deterministic archives, publishes the versioned
`artifacts-vX.Y.Z` prerelease, computes both SHA-256 and Julia artifact tree
hashes, and opens a PR containing `bindings/julia/Artifacts.toml`.

Merge that PR only after the preparatory workflow's artifact-backed Julia test
and the repository's normal branch checks pass. Do not create the final
version tag before this PR is merged.

## 2. Create the synchronized release

From the fully green `main` commit:

```bash
python ci/check_versions.py --tag vX.Y.Z
git tag -s vX.Y.Z -m "QMBED vX.Y.Z"
git push origin vX.Y.Z
```

The `release` workflow then:

1. repeats the Rust, C ABI, Python, and artifact-backed Julia gates;
2. builds and tests CPython 3.10–3.14 wheels on the supported platforms
   (Intel macOS stops at CPython 3.13 because Numba no longer ships newer
   Intel Mac binaries);
3. publishes the Rust crate to crates.io and the wheels to PyPI; and
4. creates the GitHub release with the same wheels attached.

Any registry failure prevents the GitHub release from presenting a partially
published version as complete.

## 3. Register the Julia subpackage

After the tagged workflow succeeds, comment on the tagged commit:

```text
@JuliaRegistrator register subdir=bindings/julia
```

Follow the JuliaRegistrator PR through General registry CI. The source tree
already contains the platform artifact hashes, so General installs do not
require a Rust compiler or a QMBED source checkout.

`QMBED` is an acronym-only package name. Current General auto-merge naming
rules expect at least one lowercase letter, so the initial registration is
expected to need a registry maintainer's manual review. Keep the public name
unless the project intentionally chooses a Julia-specific spelling; this is a
review gate, not a reason to fork the implementation.

## 4. Verify from clean environments

Verify the public registries rather than the repository checkout:

```bash
cargo add qmbed@X.Y.Z
python -m pip install "qmbed==X.Y.Z"
julia -e 'using Pkg; Pkg.add(name="QMBED", version="X.Y.Z")'
```

Run one eigensolve through each interface and confirm that Python resolves its
library inside `site-packages/qmbed` and Julia resolves it inside the artifact
store.
