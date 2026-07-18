#!/usr/bin/env python3
"""Check local href/src targets across the public static-site tree."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

SKIP_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    "_public_pages_export",
    "build",
    "private_library",
    "tmp_downloads",
}
EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "javascript:", "data:", "//", "#")


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.values: list[str] = []

    def handle_starttag(self, _tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.values.append(value)


def local_target(root: Path, page: Path, value: str) -> Path | None:
    if not value or value.startswith(EXTERNAL_PREFIXES):
        return None
    local = unquote(value.split("#", 1)[0].split("?", 1)[0])
    if not local:
        return None
    target = root / local.lstrip("/") if local.startswith("/") else page.parent / local
    target = target.resolve()
    if target.is_dir():
        target = target / "index.html"
    return target


def embedded_document_target(root: Path, page: Path, value: str) -> Path | None:
    """Resolve the local file carried by a document_reader ``doc`` query."""
    parsed = urlsplit(value)
    if Path(parsed.path).name != "document_reader.html":
        return None
    documents = parse_qs(parsed.query).get("doc")
    if not documents:
        return None
    reader = local_target(root, page, parsed.path)
    if reader is None:
        return None
    document = unquote(documents[0])
    target = (
        root / document.lstrip("/")
        if document.startswith("/")
        else reader.parent / document
    )
    return target.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    pages = [
        page
        for page in root.rglob("*.html")
        if not any(part in SKIP_PARTS for part in page.relative_to(root).parts)
    ]
    checked = 0
    missing = []
    for page in pages:
        link_parser = LinkParser()
        link_parser.feed(page.read_text(encoding="utf-8", errors="replace"))
        for value in link_parser.values:
            targets = [local_target(root, page, value)]
            document_target = embedded_document_target(root, page, value)
            if document_target is not None:
                targets.append(document_target)
            for target in targets:
                if target is None:
                    continue
                checked += 1
                if not target.exists():
                    missing.append((str(page.relative_to(root)), value))

    print(f"public_html={len(pages)} checked={checked} missing={len(missing)}")
    for page, value in missing:
        print(f"{page}: {value}")
    return int(bool(missing))


if __name__ == "__main__":
    raise SystemExit(main())
