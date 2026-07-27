"""Create a deterministic Julia artifact archive from a QMBED C library."""

from __future__ import annotations

import argparse
import gzip
from io import BytesIO
from pathlib import Path
import tarfile


TARGET_EXTENSIONS = {
    "aarch64-apple-darwin": ".dylib",
    "aarch64-unknown-linux-gnu": ".so",
    "x86_64-apple-darwin": ".dylib",
    "x86_64-pc-windows-msvc": ".dll",
    "x86_64-unknown-linux-gnu": ".so",
}


def package_artifact(
    target: str,
    version: str,
    target_dir: Path,
    output_dir: Path,
) -> Path:
    try:
        extension = TARGET_EXTENSIONS[target]
    except KeyError as error:
        supported = ", ".join(sorted(TARGET_EXTENSIONS))
        raise ValueError(f"unsupported target {target!r}; expected one of {supported}") from error

    cargo_name = "qmbed_capi.dll" if extension == ".dll" else f"libqmbed_capi{extension}"
    library = target_dir / target / "release" / cargo_name
    if not library.is_file():
        raise FileNotFoundError(f"native library not found: {library}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / f"qmbed-capi-v{version}-{target}.tar.gz"
    data = library.read_bytes()
    member = tarfile.TarInfo(f"lib/libqmbed_capi{extension}")
    member.size = len(data)
    member.mode = 0o755
    member.mtime = 0
    member.uid = 0
    member.gid = 0
    member.uname = ""
    member.gname = ""
    with archive.open("wb") as raw_archive:
        with gzip.GzipFile(fileobj=raw_archive, mode="wb", mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.USTAR_FORMAT,
            ) as tar:
                tar.addfile(member, BytesIO(data))
    return archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, choices=sorted(TARGET_EXTENSIONS))
    parser.add_argument("--version", required=True)
    parser.add_argument("--target-dir", type=Path, default=Path("bindings/capi/target"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archive = package_artifact(
        args.target,
        args.version,
        args.target_dir,
        args.output_dir,
    )
    print(archive)


if __name__ == "__main__":
    main()
