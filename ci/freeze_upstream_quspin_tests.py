#!/usr/bin/env python3
"""Freeze and validate the official QuSpin Python compatibility tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


UPSTREAM_REPOSITORY = "https://github.com/QuSpin/QuSpin.git"
UPSTREAM_COMMIT = "5bf9e5b266e6d8b70e5cf5973c7c7d59d62e412f"
UPSTREAM_VERSION = "1.0.1"
SNAPSHOT = Path("bindings/python/compat_tests/quspin-1.0.1")
INITIAL_PASSING = {"test/test_version.py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_files(snapshot: Path) -> list[Path]:
    return [snapshot / "LICENSE.rst", *sorted((snapshot / "test").glob("*.py"))]


def write_snapshot(root: Path, source: Path) -> None:
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream checkout is {commit}, expected {UPSTREAM_COMMIT}"
        )

    source_tests = sorted((source / "test").glob("test_*.py"))
    if len(source_tests) != 73:
        raise ValueError(f"expected 73 upstream tests, found {len(source_tests)}")

    snapshot = root / SNAPSHOT
    if snapshot.exists():
        raise FileExistsError(
            f"{snapshot} already exists; remove it only when intentionally refreezing"
        )
    (snapshot / "test").mkdir(parents=True)
    shutil.copy2(source / "LICENSE.rst", snapshot / "LICENSE.rst")
    shutil.copy2(source / "test" / "__init__.py", snapshot / "test" / "__init__.py")
    for source_test in source_tests:
        shutil.copy2(source_test, snapshot / "test" / source_test.name)

    manifest = {
        "schema_version": 1,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_version": UPSTREAM_VERSION,
        "license": "BSD-3-Clause",
        "test_file_count": len(source_tests),
        "files": {
            str(path.relative_to(snapshot)): sha256(path)
            for path in snapshot_files(snapshot)
        },
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    all_tests = {f"test/{path.name}" for path in source_tests}
    status = {
        "schema_version": 1,
        "target_version": UPSTREAM_VERSION,
        "passing": sorted(INITIAL_PASSING),
        "unsupported": sorted(all_tests - INITIAL_PASSING),
        "unsupported_reason": (
            "The copied test is part of the compatibility contract, but its "
            "required QuSpin object protocol is not implemented yet."
        ),
    }
    (snapshot / "compat_status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n"
    )


def validate_snapshot(root: Path) -> tuple[int, int]:
    snapshot = root / SNAPSHOT
    manifest = json.loads((snapshot / "manifest.json").read_text())
    if manifest["upstream_repository"] != UPSTREAM_REPOSITORY:
        raise ValueError("upstream repository does not match the frozen source")
    if manifest["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("upstream commit does not match the frozen source")
    if manifest["upstream_version"] != UPSTREAM_VERSION:
        raise ValueError("upstream version does not match the frozen source")

    actual_files = {
        str(path.relative_to(snapshot)): sha256(path)
        for path in snapshot_files(snapshot)
    }
    if actual_files != manifest["files"]:
        expected = set(manifest["files"])
        actual = set(actual_files)
        modified = sorted(
            path
            for path in expected & actual
            if manifest["files"][path] != actual_files[path]
        )
        raise ValueError(
            "snapshot contents differ from the manifest: "
            f"missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}, "
            f"modified={modified}"
        )

    tests = {
        path
        for path in actual_files
        if path.startswith("test/test_") and path.endswith(".py")
    }
    if len(tests) != manifest["test_file_count"]:
        raise ValueError(
            f"manifest says {manifest['test_file_count']} tests, found {len(tests)}"
        )

    status = json.loads((snapshot / "compat_status.json").read_text())
    passing = set(status["passing"])
    unsupported = set(status["unsupported"])
    if passing & unsupported:
        raise ValueError("passing and unsupported compatibility tests overlap")
    if passing | unsupported != tests:
        raise ValueError(
            "compatibility status must classify every copied test exactly once"
        )
    return len(passing), len(unsupported)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.source is not None:
        write_snapshot(args.root, args.source.resolve())
    passing, unsupported = validate_snapshot(args.root)
    print(
        "official QuSpin 1.0.1 compatibility tests: "
        f"{passing} passing, {unsupported} explicitly unsupported"
    )


if __name__ == "__main__":
    main()
