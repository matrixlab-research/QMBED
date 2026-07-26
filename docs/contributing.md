# Contributing to the docs

The published site combines three builders:

- MkDocs builds the guide and Python reference;
- `cargo doc` builds the Rust API;
- Documenter.jl builds the Julia API.

## Guide and Python API

```bash
python -m pip install -r docs/requirements.txt
python -m pip install -e bindings/python
python -m mkdocs build --strict --site-dir site
```

## Rust API

```bash
cargo doc --no-deps
mkdir -p site/rust/api
cp -R target/doc/. site/rust/api/
cp docs/rustdoc-index.html site/rust/api/index.html
```

## Julia API

```bash
julia --project=bindings/julia/docs -e \
  'using Pkg; Pkg.develop(path="bindings/julia"); Pkg.instantiate()'
julia --project=bindings/julia/docs bindings/julia/docs/make.jl
mkdir -p site/julia/api
cp -R bindings/julia/docs/build/. site/julia/api/
```

Finally validate all internal HTML links:

```bash
python ci/check_docs.py site
```

The `documentation` GitHub workflow runs this complete sequence on pull
requests and deploys the assembled artifact from `main`.
