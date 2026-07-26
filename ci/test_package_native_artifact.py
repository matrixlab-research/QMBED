from pathlib import Path
import tarfile
import tempfile
import unittest

from ci.package_native_artifact import package_artifact


class NativeArtifactPackagingTests(unittest.TestCase):
    def test_archive_is_deterministic_and_has_julia_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target_dir = root / "target"
            library = (
                target_dir
                / "x86_64-unknown-linux-gnu"
                / "release"
                / "libqmbed_capi.so"
            )
            library.parent.mkdir(parents=True)
            library.write_bytes(b"qmbed-test-library")

            archive = package_artifact(
                "x86_64-unknown-linux-gnu",
                "0.2.0",
                target_dir,
                root / "artifacts",
            )
            first = archive.read_bytes()
            package_artifact(
                "x86_64-unknown-linux-gnu",
                "0.2.0",
                target_dir,
                root / "artifacts",
            )
            self.assertEqual(archive.read_bytes(), first)

            with tarfile.open(archive, "r:gz") as packaged:
                [member] = packaged.getmembers()
                self.assertEqual(member.name, "lib/libqmbed_capi.so")
                self.assertEqual(member.mode, 0o755)
                extracted = packaged.extractfile(member)
                self.assertIsNotNone(extracted)
                self.assertEqual(extracted.read(), b"qmbed-test-library")


if __name__ == "__main__":
    unittest.main()
