from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ci.check_versions import check_versions


class VersionTests(unittest.TestCase):
    def write_surface(self, root: Path, path: str, text: str) -> None:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)

    def test_repository_versions_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(check_versions(root), "0.2.0")

    def test_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write_surface(root, "Cargo.toml", '[package]\nversion="1.0.0"\n')
            self.write_surface(
                root,
                "bindings/capi/Cargo.toml",
                '[package]\nversion="1.0.0"\n',
            )
            self.write_surface(
                root,
                "bindings/python/pyproject.toml",
                '[project]\nversion="1.0.1"\n',
            )
            self.write_surface(
                root,
                "bindings/julia/Project.toml",
                'version="1.0.0"\n',
            )
            with self.assertRaisesRegex(ValueError, "python=1.0.1"):
                check_versions(root)


if __name__ == "__main__":
    unittest.main()
