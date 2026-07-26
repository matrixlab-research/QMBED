#!/usr/bin/env julia

using Pkg.Artifacts
using Pkg.BinaryPlatforms
using SHA
using Tar

const TARGETS = [
    (
        "aarch64-apple-darwin",
        Platform("aarch64", "macos"),
    ),
    (
        "aarch64-unknown-linux-gnu",
        Platform("aarch64", "linux"; libc="glibc"),
    ),
    (
        "x86_64-apple-darwin",
        Platform("x86_64", "macos"),
    ),
    (
        "x86_64-pc-windows-msvc",
        Platform("x86_64", "windows"),
    ),
    (
        "x86_64-unknown-linux-gnu",
        Platform("x86_64", "linux"; libc="glibc"),
    ),
]

function sha256_file(path)
    open(path) do io
        bytes2hex(SHA.sha256(io))
    end
end

function generate_artifacts(assets_dir, version, output, base_url)
    occursin(r"^\d+\.\d+\.\d+$", version) ||
        error("version must have the form MAJOR.MINOR.PATCH")
    rm(output; force=true)
    for (target, platform) in TARGETS
        filename = "qmbed-capi-v$(version)-$(target).tar.gz"
        archive = joinpath(assets_dir, filename)
        isfile(archive) || error("missing native archive: $(archive)")
        tree_hash = Base.SHA1(Tar.tree_hash(`gzip -dc $archive`))
        url = rstrip(base_url, '/') * "/" * filename
        download_info = [(url, sha256_file(archive))]
        bind_artifact!(
            output,
            "qmbed_capi",
            tree_hash;
            platform,
            download_info,
            lazy=true,
            force=true,
        )
    end
end

length(ARGS) == 4 || error(
    "usage: generate_julia_artifacts.jl ASSETS_DIR VERSION OUTPUT BASE_URL",
)
generate_artifacts(ARGS...)
