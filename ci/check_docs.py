#!/usr/bin/env python3
"""Validate the assembled QMBED documentation site without network access."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


REQUIRED_PAGES = (
    "index.html",
    "getting-started/index.html",
    "rust/index.html",
    "rust/api/index.html",
    "rust/api/qmbed/index.html",
    "python/index.html",
    "python/api/index.html",
    "julia/index.html",
    "julia/api/index.html",
)


class LinkCollector(HTMLParser):
    """Collect hyperlink targets from one generated HTML document."""

    def __init__(self) -> None:
        super().__init__()
        self.targets: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attributes: list[tuple[str, str | None]],
    ) -> None:
        if tag != "a":
            return
        for name, value in attributes:
            if name == "href" and value:
                self.targets.append(value)


def local_target(site: Path, source: Path, target: str) -> Path | None:
    """Resolve one local generated-site link to the file it should reach."""

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc or target.startswith(("#", "mailto:", "javascript:")):
        return None
    path = unquote(parsed.path)
    if path == "/QMBED":
        path = "/"
    elif path.startswith("/QMBED/"):
        path = path.removeprefix("/QMBED")
    if not path:
        return None
    candidate = site / path.lstrip("/") if path.startswith("/") else source.parent / path
    if candidate.is_dir() or not candidate.suffix:
        candidate /= "index.html"
    return candidate.resolve()


def validate(site: Path) -> list[str]:
    """Return human-readable errors for missing pages and broken local links."""

    errors = [
        f"missing required page: {relative}"
        for relative in REQUIRED_PAGES
        if not (site / relative).is_file()
    ]
    site_root = site.resolve()
    for source in sorted(site.rglob("*.html")):
        parser = LinkCollector()
        parser.feed(source.read_text(encoding="utf-8"))
        for target in parser.targets:
            candidate = local_target(site_root, source.resolve(), target)
            if candidate is None:
                continue
            try:
                candidate.relative_to(site_root)
            except ValueError:
                errors.append(f"{source.relative_to(site)} escapes site root: {target}")
                continue
            if not candidate.is_file():
                errors.append(f"{source.relative_to(site)} -> {target}")
    return errors


def main() -> int:
    """Parse the site directory, validate it, and return a shell exit status."""

    parser = argparse.ArgumentParser()
    parser.add_argument("site", type=Path)
    arguments = parser.parse_args()
    if not arguments.site.is_dir():
        parser.error(f"site directory does not exist: {arguments.site}")
    errors = validate(arguments.site)
    if errors:
        print("documentation validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    pages = sum(1 for _ in arguments.site.rglob("*.html"))
    print(f"documentation validation passed: {pages} HTML pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
