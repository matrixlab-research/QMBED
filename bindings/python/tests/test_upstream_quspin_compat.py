from __future__ import annotations

import ast
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import quspin


SNAPSHOT = (
    Path(__file__).resolve().parents[1] / "compat_tests" / "quspin-1.0.1"
)


class UpstreamQuSpinCompatibilityTests(unittest.TestCase):
    def test_snapshot_is_complete_and_unchanged(self):
        repository = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(repository))
        try:
            from ci.freeze_upstream_quspin_tests import validate_snapshot

            passing, unsupported = validate_snapshot(repository)
        finally:
            sys.path.remove(str(repository))
        self.assertGreaterEqual(passing, 1)
        self.assertEqual(passing + unsupported, 73)

    def test_every_copied_test_is_valid_python(self):
        for path in sorted((SNAPSHOT / "test").glob("test_*.py")):
            with self.subTest(test=path.name):
                ast.parse(path.read_bytes(), filename=str(path))

    def test_currently_supported_tests_run_without_modification(self):
        status = json.loads((SNAPSHOT / "compat_status.json").read_text())
        self.assertEqual(quspin.__version__, "1.0.1")
        for relative_path in status["passing"]:
            path = SNAPSHOT / relative_path
            module_name = f"_qmbed_upstream_{path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            tests = [
                value
                for name, value in vars(module).items()
                if name.startswith("test") and callable(value)
            ]
            # Some upstream files are executable test scripts: importing the
            # unchanged module runs their assertions directly. Function-based
            # files are imported first and then invoked explicitly below.
            for test in tests:
                with self.subTest(test=f"{relative_path}::{test.__name__}"):
                    test()


if __name__ == "__main__":
    unittest.main()
