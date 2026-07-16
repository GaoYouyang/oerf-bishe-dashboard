#!/usr/bin/env python3
"""Build the local-only unmatched PDF review page.

The page is safe to publish because it lists filenames and candidate metadata only.
The actual PDFs stay under private_library/, which is ignored by git.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "paper_library"
PRIVATE_QUEUE = ROOT / "private_library" / "unmatched_downloads"
REPORT = LIB / "import_from_tmp_report.md"
CANDIDATES = LIB / "unmatched_pdf_candidates.md"
OUT = LIB / "unmatched_pdf_queue.html"


def esc(value: str) -> str:
    return html.escape(value or "", quote=True)


def human_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.2f} MB"


def unmatched_names() -> list[str]:
    if not REPORT.exists():
        return []
    names: list[str] = []
    collect = False
    for line in REPORT.read_text(encoding="utf-8").splitlines():
        if line.startswith("## 未匹配"):
            collect = True
            continue
        if collect and line.startswith("## "):
            break
        if collect and line.startswith("- `"):
            match = re.search(r"`([^`]+)`", line)
            if match:
                names.append(match.group(1))
    return names


def candidate_map() -> dict[str, list[str]]:
    if not CANDIDATES.exists():
        return {}
    current = ""
    mapping: dict[str, list[str]] = {}
    for line in CANDIDATES.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            mapping[current] = []
            continue
        if current and line.startswith("- "):
            mapping[current].append(line[2:])
    return mapping


def main() -> None:
    candidates = candidate_map()
    rows: list[str] = []
    for name in unmatched_names():
        private_path = PRIVATE_QUEUE / name
        size = human_size(private_path) if private_path.exists() else "未找到本机文件"
        suggestions = candidates.get(name) or ["无明显候选"]
        suggestion_html = "<br>".join(esc(item) for item in suggestions[:8])
        private_note = f"private_library/unmatched_downloads/{name}"
        rows.append(
            f"""
        <tr>
          <td><strong>{esc(name)}</strong><br /><span class="muted"><code>{esc(private_note)}</code></span></td>
          <td>{esc(size)}</td>
          <td>{suggestion_html}</td>
        </tr>
            """.strip()
        )

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>未映射 PDF 待确认队列</title>
  <style>
    :root {{
      --bg: #f7faf9;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #607080;
      --line: #d9e2e4;
      --green: #147b68;
      --blue: #2866b1;
      --amber: #9a6a00;
      --soft: #f8f0dc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 24px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}
    .panel {{
      max-width: 1200px;
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    h1 {{ margin: 8px 0 10px; }}
    a {{ color: var(--blue); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid var(--line); padding: 9px 10px; vertical-align: top; font-size: 14px; }}
    th {{ background: #eef5f2; text-align: left; }}
    code {{ word-break: break-all; }}
    .muted {{ color: var(--muted); font-size: 13px; }}
    .notice {{
      border: 1px solid #ead7b5;
      border-left: 5px solid var(--amber);
      border-radius: 8px;
      background: var(--soft);
      padding: 12px;
      color: #6e5010;
      margin: 12px 0;
    }}
    .button {{
      display: inline-flex;
      align-items: center;
      min-height: 38px;
      margin-right: 8px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font-weight: 700;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <a class="button" href="index.html">返回论文库</a>
    <h1>未映射 PDF 待确认队列</h1>
    <p class="muted">这些文件已经下载到本机，但尚未与 <code>paper_library/papers.json</code> 中条目稳定映射。公开网页只保留文件名和候选信息；全文文件保留在不会上传的 <code>private_library/</code>。</p>
    <div class="notice">处理原则：确认开放许可或机构库公开版本后，再复制进 <code>paper_library/pdfs/</code> 并写入 <code>local_pdf</code>；学校 VPN/订阅得到的版本只留本机私有队列。</div>
    <p class="muted">更新时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}；待确认文件：{len(rows)} 个。</p>
    <table>
      <thead><tr><th>文件</th><th>本机大小</th><th>候选条目</th></tr></thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    OUT.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
