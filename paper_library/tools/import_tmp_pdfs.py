#!/usr/bin/env python3
"""Sync downloaded PDFs from tmp folders into the public paper library index.

Usage:
    python3 paper_library/tools/import_tmp_pdfs.py

It supports two source locations:
  - paper_library/inbox (for manually put new PDFs)
  - repo tmp_downloads (historical downloads)

Matching policy:
  1) exact filename match against existing local_pdf values;
  2) fuzzy match against entry id/title with strict score threshold;
  3) year consistency check to reduce false positives.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_FILE = ROOT / "papers.json"
PDF_DIR = ROOT / "pdfs"
INBOX = ROOT / "inbox"
TMP_ROOT = Path(__file__).resolve().parents[2] / "tmp_downloads"
REPORT = ROOT / "import_from_tmp_report.md"

SCORE_THRESHOLD = 0.87
SCORE_GAP = 0.05
MANUAL_MAP_FILE = ROOT / "manual_pdf_map.json"


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def year_from_filename(name: str) -> str:
    match = re.search(r"(20\d{2}|19\d{2})", name)
    return match.group(1) if match else ""


def choose_match(entry: dict, file_stem: str) -> float:
    score = max(
        SequenceMatcher(None, file_stem, norm(entry["title"])).ratio(),
        SequenceMatcher(None, file_stem, norm(entry["id"].replace("-", " "))).ratio(),
        SequenceMatcher(None, file_stem, norm(entry["id"])).ratio(),
    )
    return score


def source_pdfs():
    for pdf in TMP_ROOT.rglob("*.pdf"):
        yield pdf
    if INBOX.exists():
        for pdf in INBOX.rglob("*.pdf"):
            yield pdf


def already_bound(papers: list[dict], filename: str) -> bool:
    return any(
        entry.get("local_pdf", "").endswith(f"/{filename}")
        or entry.get("local_pdf", "") == filename
        for entry in papers
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="do not copy files, only print plan")
    parser.add_argument(
        "--manual-map",
        type=Path,
        default=MANUAL_MAP_FILE,
        help="JSON map: {filename: paper_id}",
    )
    args = parser.parse_args()

    papers = json.loads(JSON_FILE.read_text(encoding="utf-8"))
    paper_by_id = {entry["id"]: entry for entry in papers}
    manual_map = {}
    if args.manual_map.exists():
        try:
            manual_map = {
                str(k): str(v)
                for k, v in json.loads(args.manual_map.read_text(encoding="utf-8")).items()
            }
        except Exception:
            manual_map = {}
            print(f"警告：手工映射文件读取失败 {args.manual_map}，已忽略。")
    candidates = [entry for entry in papers if not entry.get("local_pdf")]
    matched: list[tuple[str, str, float]] = []
    unmatched: list[tuple[str, list[str]]] = []

    for src in sorted(set(source_pdfs()), key=lambda p: p.name):
        if src.suffix.lower() != ".pdf":
            continue
        if already_bound(papers, src.name):
            continue

        file_stem = norm(src.stem)
        file_year = year_from_filename(src.stem)
        if src.name in manual_map:
            entry_id = manual_map[src.name]
            manual_entry = paper_by_id.get(entry_id)
            if manual_entry and not manual_entry.get("local_pdf"):
                matched.append((src.name, entry_id, 1.0))
                dst = PDF_DIR / src.name
                if not args.dry_run:
                    dst.parent.mkdir(exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    manual_entry["local_pdf"] = f"pdfs/{src.name}"
                    if "本地PDF已缓存" not in manual_entry.get("access", ""):
                        manual_entry["access"] = f"本地PDF已缓存（{str(src.parent)}）。" + (
                            (" " + manual_entry["access"]) if manual_entry.get("access") else ""
                        )
                    candidates = [c for c in candidates if c is not manual_entry]
                    for ext in [".png", ".jpg"]:
                        preview_src = src.with_suffix(ext)
                        if preview_src.exists():
                            preview_dst = ROOT / "previews" / preview_src.name
                            shutil.copy2(preview_src, preview_dst)
                            manual_entry["preview"] = f"previews/{preview_src.name}"
                continue
            unmatched.append((src.name, [f"{entry_id} (手工映射不存在或已入库)"]))
            continue

        scored: list[tuple[float, int, str, dict]] = []

        for entry in candidates:
            entry_year = str(entry.get("year", "")).strip()
            score = choose_match(entry, file_stem)
            if score < SCORE_THRESHOLD:
                continue

            # year mismatch is allowed only if score is extremely high
            if file_year and entry_year and file_year != entry_year and score < 0.97:
                continue

            overlap = len(set(file_stem.split()) & set(norm(entry["title"]).split()))
            scored.append((score, overlap, entry["id"], entry))

        scored.sort(reverse=True, key=lambda row: (row[0], row[1]))
        if not scored:
            unmatched.append((src.name, []))
            continue

        top = scored[0]
        if len(scored) > 1 and top[0] - scored[1][0] < SCORE_GAP:
            unmatched.append((src.name, [scored[0][2], scored[1][2], scored[2][2] if len(scored) > 2 else ""]))
            continue

        score, _, entry_id, entry = top
        matched.append((src.name, entry_id, score))
        dst = PDF_DIR / src.name
        if not args.dry_run and not dst.exists():
            dst.parent.mkdir(exist_ok=True)
            shutil.copy2(src, dst)
        entry["local_pdf"] = f"pdfs/{src.name}"

        # keep status text explicit for local availability
        if "本地PDF已缓存" not in entry.get("access", ""):
            entry["access"] = f"本地PDF已缓存（{str(src.parent)}）。" + (
                (" " + entry["access"]) if entry.get("access") else ""
            )
        candidates = [c for c in candidates if c is not entry]

        # copy preview if source has matching image file
        for ext in [".png", ".jpg"]:
            preview_src = src.with_suffix(ext)
            if preview_src.exists():
                preview_dst = ROOT / "previews" / preview_src.name
                if not args.dry_run:
                    preview_dst.parent.mkdir(exist_ok=True)
                    shutil.copy2(preview_src, preview_dst)
                    entry["preview"] = f"previews/{preview_src.name}"

    # report
    lines = [
        "# tmp 下载自动入库报告",
        "",
        f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 已匹配入库：{len(matched)}",
        f"- 未匹配：{len(unmatched)}",
        "",
        "## 已匹配（入库）",
    ]
    if matched:
        for name, pid, score in matched:
            lines.append(f"- `{name}` -> `{pid}` (score={score:.3f})")
    else:
        lines.append("- 本次无高置信度可入库文件。")

    lines.extend(["", "## 未匹配（需人工命名/确认）"])
    if unmatched:
        for name, options in unmatched:
            if options:
                options_txt = ", ".join([f"`{x}`" for x in options if x])
                lines.append(f"- `{name}`（候选：{options_txt}）")
            else:
                lines.append(f"- `{name}`")
    else:
        lines.append("- 当前无未匹配文件。")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    JSON_FILE.write_text(json.dumps(papers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Matched={len(matched)}; Unmatched={len(unmatched)}; Report: {REPORT}")


if __name__ == "__main__":
    main()
