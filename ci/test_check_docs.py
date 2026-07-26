"""Tests for the offline assembled-documentation validator."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from check_docs import LinkCollector, local_target


class LinkCollectorTests(unittest.TestCase):
    def test_collects_only_anchor_hrefs(self):
        parser = LinkCollector()
        parser.feed('<a href="../api/">API</a><img src="ignored.png">')
        self.assertEqual(parser.targets, ["../api/"])


class LocalTargetTests(unittest.TestCase):
    def test_resolves_directory_links_to_index(self):
        with TemporaryDirectory() as directory:
            site = Path(directory).resolve()
            source = site / "python" / "index.html"
            self.assertEqual(
                local_target(site, source, "../rust/"),
                site / "rust" / "index.html",
            )

    def test_ignores_external_and_fragment_links(self):
        with TemporaryDirectory() as directory:
            site = Path(directory).resolve()
            source = site / "index.html"
            self.assertIsNone(local_target(site, source, "https://example.com"))
            self.assertIsNone(local_target(site, source, "#section"))


if __name__ == "__main__":
    unittest.main()
