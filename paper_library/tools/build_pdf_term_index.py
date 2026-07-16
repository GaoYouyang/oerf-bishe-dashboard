#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.parse import quote

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT.parent
PAPERS_PATH = ROOT / "papers.json"
TERMS_PATH = SITE_ROOT / "glossary_terms.js"
OUT_JSON = ROOT / "term_index.json"
OUT_HTML = ROOT / "term_index.html"
OUT_READERS = ROOT / "readers"


def clean_trailing_whitespace(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()) + "\n"


def read_js_array(path: Path, marker: str) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"^\s*window\.[A-Z0-9_]+\s*=\s*", "", text.strip())
    text = re.sub(r";\s*$", "", text)
    # The data file is deliberately JSON-like JS. Convert unquoted object keys.
    text = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', text)
    return json.loads(text)


def alias_pattern(alias: str) -> re.Pattern[str]:
    normalized = alias.strip()
    escaped = re.escape(normalized)
    if re.fullmatch(r"[A-Za-z0-9+./_-]+", normalized):
        flags = (
            0
            if re.fullmatch(r"[A-Z][A-Z0-9+.-]{1,5}", normalized)
            else re.IGNORECASE
        )
        return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", flags)
    return re.compile(escaped, re.IGNORECASE)


def term_patterns(term: dict) -> list[re.Pattern[str]]:
    """Compile only names and curated aliases for PDF text matching.

    ``keywords`` are intentionally excluded: they support broad website search,
    but generic entries such as ``noise`` or ``least squares`` are not evidence
    that a paper contains the specific glossary concept they describe.
    """
    aliases = [term.get("zh", ""), term.get("en", ""), *(term.get("aliases") or [])]
    parts: list[str] = []
    for alias in aliases:
        parts.extend(re.split(r"[;,|]", alias))
    unique_parts = dict.fromkeys(
        part.strip().casefold() for part in parts if len(part.strip()) >= 2
    )
    original_by_key = {
        part.strip().casefold(): part.strip()
        for part in parts
        if len(part.strip()) >= 2
    }
    return [alias_pattern(original_by_key[key]) for key in unique_parts]


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    return pages


def build_index() -> list[dict]:
    papers = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    terms = read_js_array(TERMS_PATH, "OERF_GLOSSARY_TERMS")
    compiled: list[tuple[dict, list[re.Pattern[str]]]] = []
    for term in terms:
        compiled.append((term, term_patterns(term)))

    index: list[dict] = []
    for paper in papers:
        local_pdf = paper.get("local_pdf") or ""
        if not local_pdf:
            continue
        pdf_path = ROOT / local_pdf
        if not pdf_path.exists():
            continue
        try:
            pages = extract_pages(pdf_path)
        except Exception as exc:
            index.append({
                "id": paper.get("id"),
                "title": paper.get("title"),
                "local_pdf": local_pdf,
                "error": str(exc),
                "terms": []
            })
            continue

        term_hits = []
        for term, patterns in compiled:
            page_numbers: list[int] = []
            page_hits: list[dict] = []
            total = 0
            for i, text in enumerate(pages, start=1):
                count = sum(len(pattern.findall(text)) for pattern in patterns)
                if count:
                    page_numbers.append(i)
                    page_hits.append({"page": i, "count": count})
                    total += count
            if total:
                term_hits.append({
                    "zh": term.get("zh"),
                    "en": term.get("en"),
                    "category": term.get("category"),
                    "explain": term.get("explain"),
                    "use": term.get("use"),
                    "pitfall": term.get("pitfall"),
                    "keywords": term.get("keywords"),
                    "count": total,
                    "pages": page_numbers,
                    "page_hits": page_hits,
                    "page_count": len(page_numbers)
                })
        term_hits.sort(key=lambda x: (-x["count"], x["zh"]))
        page_summary = build_page_summary(term_hits)
        index.append({
            "id": paper.get("id"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "group": paper.get("group"),
            "priority": paper.get("priority"),
            "local_pdf": local_pdf,
            "landing_url": paper.get("landing_url"),
            "online_pdf_url": paper.get("online_pdf_url"),
            "why": paper.get("why"),
            "extract": paper.get("extract"),
            "access": paper.get("access"),
            "reader_url": f"readers/{paper.get('id')}.html",
            "term_count": len(term_hits),
            "page_summary": page_summary,
            "terms": term_hits
        })
    return index


def build_page_summary(term_hits: list[dict], max_terms_per_page: int = 12) -> list[dict]:
    page_map: dict[int, list[dict]] = {}
    for term in term_hits:
        for hit in term.get("page_hits") or []:
            page = int(hit.get("page") or 0)
            if page <= 0:
                continue
            page_map.setdefault(page, []).append({
                "zh": term.get("zh"),
                "en": term.get("en"),
                "category": term.get("category"),
                "count": int(hit.get("count") or 0)
            })

    summary: list[dict] = []
    for page, terms in sorted(page_map.items()):
        ordered = sorted(terms, key=lambda t: (-t["count"], t.get("zh") or ""))
        summary.append({
            "page": page,
            "total": sum(t["count"] for t in ordered),
            "terms": ordered[:max_terms_per_page],
            "hidden_terms": max(0, len(ordered) - max_terms_per_page)
        })
    return summary


def asset_viewer_href(local_pdf: str, prefix_to_root: str = "../") -> str:
    return f"{prefix_to_root}asset_viewer.html?asset={quote('paper_library/' + local_pdf, safe='')}"


def page_links(local_pdf: str, pages: list[int], prefix_to_root: str = "../") -> str:
    if not pages:
        return ""
    visible = pages[:28]
    links = "".join(
        f'<a href="{html.escape(asset_viewer_href(local_pdf, prefix_to_root))}#page={page}" target="_blank" rel="noreferrer">p.{page}</a>'
        for page in visible
    )
    if len(pages) > len(visible):
        links += f'<span class="more">+{len(pages) - len(visible)} pages</span>'
    return links


def term_library_link(term_zh: str, prefix: str = "../") -> str:
    return f"{prefix}term_index.html?q={quote(term_zh)}"


def glossary_link(term_zh: str) -> str:
    return f"../../index.html?term={quote(term_zh)}#glossary"


def item_text(item: dict) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in ["id", "title", "authors", "venue", "group", "why", "extract"]
    ).lower()


def reader_role(item: dict) -> str:
    text = item_text(item)
    if "he mainline" in text or "yuanzhe he" in text:
        return "何远哲同步主线：建议优先精读，把它直接拆成问题定义、数据需求、baseline 和可问师兄的问题。"
    if "piv" in text:
        return "PIV-BOST / 速度测量接口：重点看折射误差如何进入粒子图像或速度向量，以及补偿发生在哪一层。"
    if "optical flow" in text or "displacement" in text:
        return "BOS 位移估计与前处理：重点看图像位移、置信度、噪声和下游重构误差传播。"
    if "4d" in text or "tensor" in text or "time-resolved" in text:
        return "四维/时序重构支撑：重点看低秩、时序一致性、内存和速度之间的取舍。"
    if "calibration" in text or "geometry" in text or "view" in text:
        return "实验几何与误差控制：重点看相机、背景、视角、mask 和坐标标定如何影响结果。"
    if "spectrometer" in text or "spectroscopy" in text:
        return "OERF 计算成像/光谱旁支：用于理解课题组能力边界，除非师兄明确需要，否则不作为 BOST 主线。"
    return "BOS/BOST 或计算流动可视化支撑文献：先读它解决的测量问题，再判断能否转化为本科可做的小模块。"


def reader_steps(item: dict) -> list[tuple[str, str]]:
    return [
        ("定位角色", reader_role(item)),
        ("抓方法图", "优先找 optical setup、forward model、reconstruction pipeline、evaluation table，不先陷入所有推导。"),
        ("抽可复现变量", "记录输入数据、几何参数、噪声/视角设置、baseline、loss/regularization 和评价指标。"),
        ("查术语页码", "用本页的术语卡片跳到 PDF 对应页，建立自己的概念索引。"),
        ("转成毕设问题", "把这篇文献写成一个可执行问题：我能复现哪一段、比较什么、问师兄要什么数据。"),
    ]


def suggested_topics(item: dict) -> list[tuple[str, str, str]]:
    text = item_text(item)
    rules = [
        ("T1", "NeRIF/BOST 鲁棒性", ["nerif", "refractive index", "bost", "tomography", "neural field"]),
        ("T2", "少视角与不确定度", ["uncertainty", "sparse-view", "few-view", "few view", "limited-view"]),
        ("T3", "PIV-BOST 补偿", ["piv", "particle image velocimetry", "velocity"]),
        ("T4", "4D BOST 低秩时序", ["4d", "tensor", "time-resolved", "temporal", "dynamic"]),
        ("T5", "BOS 位移 benchmark", ["optical flow", "displacement", "registration", "cross-correlation"]),
        ("T6", "几何标定误差", ["calibration", "geometry", "view", "camera", "depth-of-field"]),
        ("T8", "计算流动可视化综述线", ["computational flow visualization", "visualization", "review"]),
        ("T12", "小型光谱/计算成像旁支", ["spectrometer", "spectroscopy", "spectral"]),
    ]
    matches: list[tuple[str, str, str]] = []
    for topic_id, title, keywords in rules:
        if any(keyword in text for keyword in keywords):
            matches.append((topic_id, title, "由标题、分组或关键词匹配；用于判断这篇文献可服务的选题模块。"))
    if not matches:
        matches = [
            ("T1", "NeRIF/BOST 鲁棒性", "默认主线入口：用来判断是否能转成 BOST/反问题模块。"),
            ("T5", "BOS 位移 benchmark", "默认工具入口：多数 BOS/Schlieren 文献都可反查位移或图像质量。"),
            ("T6", "几何标定误差", "默认实验入口：多数光学诊断文献都绕不开系统几何和误差。"),
        ]
    return matches[:4]


def write_reader_pages(index: list[dict]) -> None:
    OUT_READERS.mkdir(exist_ok=True)
    for old in OUT_READERS.glob("*.html"):
        old.unlink()

    for item in index:
        if item.get("error"):
            continue
        terms = item.get("terms", [])
        top_terms = terms[:8]
        top_cards = "".join(
            f"""
            <article class="focus-term">
              <a class="term-title" href="{glossary_link(t["zh"])}" target="_blank" rel="noreferrer">{html.escape(t["zh"])} <span>{html.escape(t.get("en") or "")}</span></a>
              <p>{html.escape(t.get("explain") or "")}</p>
              <div class="page-links">{page_links(item.get("local_pdf") or "", t.get("pages") or [], "../../")}</div>
              <div class="term-actions">
                <a href="{term_library_link(t["zh"])}">跨论文反查</a>
                <a href="{glossary_link(t["zh"])}" target="_blank" rel="noreferrer">术语解释</a>
              </div>
            </article>
            """
            for t in top_terms
        ) or '<p class="muted">未匹配到当前术语词典。</p>'

        rows = "".join(
            f"""
            <article class="term-row" data-text="{html.escape(json.dumps(t, ensure_ascii=False))}">
              <div>
                <a class="term-name" href="{glossary_link(t["zh"])}" target="_blank" rel="noreferrer">{html.escape(t["zh"])}</a>
                <p class="en">{html.escape(t.get("en") or "")}</p>
                <p>{html.escape(t.get("explain") or "")}</p>
                <p><b>读这篇时看什么：</b>{html.escape(t.get("use") or "")}</p>
                <p><b>易混点：</b>{html.escape(t.get("pitfall") or "")}</p>
              </div>
              <div class="term-side">
                <span>{html.escape(t.get("category") or "")}</span>
                <b>{t.get("count", 0)} hits</b>
                <small>{t.get("page_count", 0)} pages</small>
                <div class="page-links">{page_links(item.get("local_pdf") or "", t.get("pages") or [], "../../")}</div>
                <div class="term-actions">
                  <a href="{term_library_link(t["zh"])}">跨论文反查</a>
                  <a href="{glossary_link(t["zh"])}" target="_blank" rel="noreferrer">术语解释</a>
                </div>
              </div>
            </article>
            """
            for t in terms
        ) or '<p class="muted">未匹配到当前术语词典。</p>'

        page_cards = []
        for page_item in item.get("page_summary") or []:
            page = page_item.get("page")
            mini_terms = "".join(
                f'<a class="mini-term" href="{glossary_link(term["zh"])}" target="_blank" rel="noreferrer">'
                f'{html.escape(term.get("zh") or "")}<span>{term.get("count", 0)}x</span></a>'
                for term in page_item.get("terms") or []
            )
            hidden = (
                f'<small class="more">+{page_item.get("hidden_terms")} more terms</small>'
                if page_item.get("hidden_terms")
                else ""
            )
            page_cards.append(f"""
              <article class="page-card">
                <a class="page-title" href="{html.escape(asset_viewer_href(item.get("local_pdf") or "", "../../"))}#page={page}" target="_blank" rel="noreferrer">p.{page}</a>
                <small>{page_item.get("total", 0)} term hits</small>
                <div class="mini-terms">{mini_terms}{hidden}</div>
              </article>
            """)
        page_map_html = "".join(page_cards) or '<p class="muted">未生成页码术语地图。</p>'

        source_link = (
            f'<a class="button" href="{html.escape(item.get("landing_url") or "")}" target="_blank" rel="noreferrer">Publisher / source</a>'
            if item.get("landing_url")
            else ""
        )
        online_link = (
            f'<a class="button" href="{html.escape(item.get("online_pdf_url") or "")}" target="_blank" rel="noreferrer">Online PDF</a>'
            if item.get("online_pdf_url")
            else ""
        )
        category_summary: dict[str, int] = {}
        for term in terms:
            category_summary[term.get("category") or "未分类"] = category_summary.get(term.get("category") or "未分类", 0) + 1
        category_html = "".join(
            f"<span>{html.escape(category)} · {count}</span>"
            for category, count in sorted(category_summary.items(), key=lambda x: (-x[1], x[0]))
        )
        detail_link = f"../paper_detail.html?id={quote(str(item.get('id') or ''), safe='')}"
        guide_cards = "".join(
            f"""
            <article class="guide-card">
              <b>{html.escape(title)}</b>
              <p>{html.escape(text)}</p>
            </article>
            """
            for title, text in reader_steps(item)
        )
        topic_cards = "".join(
            f"""
            <article class="topic-card">
              <span>{html.escape(topic_id)}</span>
              <h3>{html.escape(title)}</h3>
              <p>{html.escape(why)}</p>
              <a href="../../topic_deep_dive.html?id={quote(topic_id)}">打开选题详情</a>
            </article>
            """
            for topic_id, title, why in suggested_topics(item)
        )

        html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(item.get("title") or "")} · 术语导读</title>
  <style>
    :root {{ --bg:#f7f8f6; --panel:#fff; --ink:#1d2328; --muted:#65717a; --line:#d9e0df; --green:#16756f; --blue:#315f93; --amber:#9a6a00; --soft:#eef6f3; --shadow:0 16px 36px rgba(30,45,55,.08); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; color:var(--ink); background:linear-gradient(90deg,rgba(247,248,246,.94),rgba(247,248,246,.76)),url("../../assets/backgrounds/oerf-refractive-field.png") right -150px top 18px / min(920px,92vw) auto no-repeat fixed,var(--bg); line-height:1.62; letter-spacing:0; }}
    header, main {{ padding:28px clamp(18px,4vw,58px); }}
    header {{ border-bottom:1px solid var(--line); background:linear-gradient(180deg,#fff,#edf6f3); }}
    h1 {{ margin:14px 0 10px; max-width:1100px; font-size:clamp(28px,4vw,48px); line-height:1.12; letter-spacing:0; }}
    h2 {{ margin:26px 0 12px; font-size:24px; }}
    p {{ margin:6px 0; }}
    .muted, .en {{ color:var(--muted); }}
    .buttons, .chips, .focus-grid, .page-links, .term-actions, .mini-terms {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .button, .page-links a, .term-actions a {{ display:inline-flex; align-items:center; min-height:34px; padding:7px 10px; border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); text-decoration:none; font-weight:700; }}
    .button.primary {{ background:var(--green); border-color:var(--green); color:#fff; }}
    .summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }}
    .summary div, .focus-term, .term-row, .note {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); box-shadow:var(--shadow); }}
    .summary div {{ padding:14px 16px; }}
    .summary b {{ display:block; font-size:28px; line-height:1; color:var(--green); }}
    .note {{ padding:14px 16px; color:var(--muted); box-shadow:none; }}
    .chips span {{ border-radius:999px; padding:5px 9px; background:#eaf0f7; color:var(--blue); font-size:12px; font-weight:700; }}
    .guide-grid {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:12px 0 20px; }}
    .guide-card, .topic-card {{ border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.94); padding:13px; min-width:0; }}
    .guide-card {{ border-left:4px solid var(--green); }}
    .guide-card b {{ display:block; color:var(--green); font-size:13px; margin-bottom:5px; }}
    .guide-card p, .topic-card p {{ margin:0; color:#43545c; font-size:13px; line-height:1.55; }}
    .topic-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:12px 0 20px; }}
    .topic-card span {{ display:inline-flex; border-radius:999px; padding:4px 8px; background:var(--soft); color:var(--green); font-size:12px; font-weight:800; }}
    .topic-card h3 {{ margin:8px 0 5px; font-size:16px; line-height:1.3; }}
    .topic-card a {{ display:inline-flex; align-items:center; min-height:30px; margin-top:8px; padding:5px 8px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--blue); text-decoration:none; font-size:12px; font-weight:800; }}
    .focus-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .focus-term {{ padding:15px; }}
    .term-title, .term-name {{ color:var(--green); font-weight:800; text-decoration:none; }}
    .term-title span {{ display:block; color:var(--muted); font-weight:600; font-size:13px; }}
    .term-actions {{ margin-top:8px; }}
    .term-actions a {{ min-height:28px; padding:4px 7px; color:var(--blue); font-size:12px; }}
    .page-map {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:10px; }}
    .page-card {{ border:1px solid var(--line); border-radius:8px; background:#fff; padding:12px; }}
    .page-title {{ display:inline-flex; margin-bottom:6px; color:var(--green); font-size:18px; font-weight:900; text-decoration:none; }}
    .page-card small {{ display:block; color:var(--muted); }}
    .mini-term {{ display:inline-flex; align-items:center; gap:5px; border-radius:999px; padding:4px 7px; background:#f6efe2; border:1px solid #ead7b5; color:var(--amber); font-size:12px; font-weight:700; text-decoration:none; }}
    .mini-term span {{ color:#6f5b2b; font-weight:700; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px,1fr) 190px; gap:10px; margin:16px 0; }}
    input, select {{ min-height:42px; border:1px solid var(--line); border-radius:8px; padding:10px 12px; background:#fff; font:inherit; }}
    .term-list {{ display:grid; gap:12px; }}
    .term-row {{ display:grid; grid-template-columns:minmax(0,1fr) 240px; gap:14px; padding:15px; }}
    .term-row p {{ margin:4px 0; }}
    .term-side {{ display:flex; flex-direction:column; align-items:flex-start; gap:7px; }}
    .term-side span {{ border-radius:999px; padding:4px 8px; background:var(--soft); color:var(--green); font-size:12px; font-weight:700; }}
    .term-side b {{ font-size:24px; color:var(--amber); }}
    .page-links a {{ min-height:28px; padding:4px 7px; font-size:12px; color:var(--blue); }}
    .more {{ color:var(--muted); font-size:12px; padding:4px 0; }}
    footer {{ margin-top:28px; color:var(--muted); font-size:12px; }}
    @media (max-width:1080px) {{ .guide-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .topic-grid {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:900px) {{ .summary, .focus-grid, .term-row, .toolbar, .page-map, .guide-grid, .topic-grid {{ grid-template-columns:1fr; }} body {{ background-attachment:scroll; }} }}
  </style>
</head>
<body>
  <header>
    <div class="buttons">
      <a class="button" href="../index.html">Back to paper library</a>
      <a class="button" href="../term_index.html">PDF term index</a>
      <a class="button" href="../../index.html#glossary">Glossary</a>
      <a class="button" href="{html.escape(detail_link)}">Paper detail</a>
    </div>
    <h1>{html.escape(item.get("title") or "")}</h1>
    <p class="muted">{html.escape(item.get("authors") or "")}</p>
    <p class="muted">{html.escape(item.get("venue") or "")} · {html.escape(str(item.get("year") or ""))}</p>
    <div class="buttons">
      <a class="button primary" href="{html.escape(asset_viewer_href(item.get("local_pdf") or "", "../../"))}" target="_blank" rel="noreferrer">Open PDF</a>
      {online_link}
      {source_link}
    </div>
    <div class="summary">
      <div><b>{len(terms)}</b><span>matched terms</span></div>
      <div><b>{sum(t.get("count", 0) for t in terms)}</b><span>term hits</span></div>
      <div><b>{len(category_summary)}</b><span>categories</span></div>
      <div><b>{html.escape(item.get("priority") or "")}</b><span>{html.escape(item.get("group") or "")}</span></div>
    </div>
    <div class="chips">{category_html}</div>
  </header>
  <main>
    <section class="note">
      <p><b>阅读定位：</b>{html.escape(item.get("why") or "")}</p>
      <p><b>精读时提取：</b>{html.escape(item.get("extract") or "")}</p>
      <p><b>边界说明：</b>本页只保存术语命中、解释和页码跳转，不复制论文正文。点击页码会回到原 PDF 的对应页。</p>
    </section>
    <h2>读这篇的路线</h2>
    <section class="guide-grid">{guide_cards}</section>
    <h2>关联毕设选题</h2>
    <section class="topic-grid">{topic_cards}</section>
    <h2>先看这几个术语</h2>
    <section class="focus-grid">{top_cards}</section>
    <h2>按页术语地图</h2>
    <section class="note">
      <p><b>使用方式：</b>先点 p.页码打开原 PDF，再看这一页命中的概念。这个区域相当于“读论文导航”，不是论文正文重排。</p>
    </section>
    <section class="page-map">{page_map_html}</section>
    <h2>全术语检索</h2>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索术语、英文名、类别、解释，例如 BOST / density / PINN" />
      <select id="category">
        <option value="all">All categories</option>
        {''.join(f'<option value="{html.escape(category)}">{html.escape(category)}</option>' for category in sorted(category_summary))}
      </select>
    </div>
    <section id="terms" class="term-list">{rows}</section>
    <footer>Generated from open cached PDFs with pypdf. Private/WebVPN PDFs are excluded from public reader pages.</footer>
  </main>
  <script>
    const search = document.getElementById('search');
    const category = document.getElementById('category');
    const rows = Array.from(document.querySelectorAll('.term-row'));
    function apply() {{
      const q = search.value.trim().toLowerCase();
      const parts = q.split(/\\s+/).filter(Boolean);
      rows.forEach((row) => {{
        const data = JSON.parse(row.dataset.text);
        const text = row.innerText.toLowerCase();
        const okQ = !parts.length || parts.every((part) => text.includes(part));
        const okC = category.value === 'all' || data.category === category.value;
        row.style.display = okQ && okC ? '' : 'none';
      }});
    }}
    const params = new URLSearchParams(location.search);
    const initial = params.get('q') || params.get('term') || '';
    if (initial) search.value = initial;
    search.addEventListener('input', apply);
    category.addEventListener('change', apply);
    apply();
  </script>
  <script src="../../site_link_router.js" data-site-root="../../"></script>
</body>
</html>
"""
        (OUT_READERS / f"{item.get('id')}.html").write_text(clean_trailing_whitespace(html_text), encoding="utf-8")


def write_html(index: list[dict]) -> None:
    total_hits = sum(sum(t["count"] for t in item.get("terms", [])) for item in index)
    cards = []
    for item in index:
        if item.get("error"):
            term_html = f'<p class="warn">PDF text extraction failed: {html.escape(item["error"])}</p>'
        else:
            term_html = "".join(
                f'<a class="term" href="?q={quote(t["zh"])}" title="筛选所有包含该术语的论文">'
                f'{html.escape(t["zh"])} <span>{t["count"]}x / p.{", ".join(map(str, t["pages"][:8]))}</span></a>'
                for t in item.get("terms", [])[:24]
            ) or '<p class="muted">未匹配到当前术语词典。</p>'
        reader_link = (
            f'<a href="{html.escape(item.get("reader_url") or "#")}">Term reader</a>'
            if item.get("reader_url") and not item.get("error")
            else ""
        )
        cards.append(f"""
          <article class="card" data-text="{html.escape(json.dumps(item, ensure_ascii=False))}">
            <div class="meta">
              <span>{html.escape(item.get("priority") or "")}</span>
              <span>{html.escape(item.get("group") or "")}</span>
              <span>{html.escape(str(item.get("year") or ""))}</span>
              <span>{html.escape(str(item.get("term_count") or 0))} terms</span>
            </div>
            <h3>{html.escape(item.get("title") or "")}</h3>
            <p>{html.escape(item.get("authors") or "")}</p>
            <p class="muted">{html.escape(item.get("venue") or "")}</p>
            <div class="terms">{term_html}</div>
            <div class="actions">
              {reader_link}
              <a href="{html.escape(asset_viewer_href(item.get("local_pdf") or "", "../"))}" target="_blank" rel="noreferrer">Open PDF</a>
              <a href="{html.escape(item.get("landing_url") or "#")}" target="_blank" rel="noreferrer">Source</a>
            </div>
          </article>
        """)

    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OERF / BOST PDF 术语页码索引</title>
  <style>
    :root {{ --bg:#f7f8f6; --panel:#fff; --ink:#1d2328; --muted:#65717a; --line:#d9e0df; --teal:#16756f; --blue:#315f93; --gold:#a46a14; --shadow:0 16px 38px rgba(30,45,55,.08); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",Arial,sans-serif; color:var(--ink); background:linear-gradient(90deg,rgba(247,248,246,.94),rgba(247,248,246,.78)),url("../assets/backgrounds/oerf-refractive-field.png") right -150px top 18px / min(920px,92vw) auto no-repeat fixed,var(--bg); line-height:1.62; letter-spacing:0; }}
    header, main {{ padding:28px clamp(18px,4vw,56px); }}
    header {{ border-bottom:1px solid var(--line); background:linear-gradient(180deg,#fff,#edf6f3); }}
    h1 {{ margin:12px 0 10px; font-size:clamp(30px,4vw,50px); line-height:1.08; letter-spacing:0; }}
    h2 {{ margin:0 0 12px; font-size:22px; line-height:1.25; }}
    p {{ margin-top:0; }}
    .muted {{ color:var(--muted); }}
    .button, .actions a {{ display:inline-flex; align-items:center; min-height:36px; padding:7px 11px; border:1px solid var(--line); border-radius:8px; background:#fff; color:var(--ink); text-decoration:none; font-weight:700; }}
    .toolbar {{ display:grid; grid-template-columns:minmax(240px,1fr) 180px; gap:10px; margin:16px 0; }}
    input, select {{ min-height:42px; border:1px solid var(--line); border-radius:8px; padding:10px 12px; font:inherit; background:#fff; }}
    .summary {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:18px; }}
    .summary div, .card {{ border:1px solid var(--line); border-radius:8px; background:var(--panel); box-shadow:var(--shadow); }}
    .summary div {{ padding:14px 16px; }}
    .summary b {{ display:block; color:var(--teal); font-size:30px; line-height:1; }}
    .guide-row {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:0 0 18px; }}
    .guide-card {{ border:1px solid var(--line); border-radius:8px; background:rgba(255,255,255,.92); box-shadow:var(--shadow); padding:13px; border-left:4px solid var(--teal); }}
    .guide-card b {{ display:block; color:var(--teal); font-size:13px; margin-bottom:5px; }}
    .guide-card p {{ margin:0; color:#43545c; font-size:13px; line-height:1.55; }}
    .grid {{ display:grid; gap:14px; }}
    .card {{ padding:16px; }}
    .card h3 {{ margin:8px 0 6px; font-size:19px; line-height:1.35; }}
    .meta, .terms, .actions {{ display:flex; flex-wrap:wrap; gap:7px; }}
    .meta span {{ border-radius:999px; padding:4px 8px; background:#eaf0f7; color:var(--blue); font-size:12px; font-weight:700; }}
    .term {{ border-radius:999px; padding:5px 8px; background:#f6efe2; color:var(--gold); border:1px solid #ead7b5; font-size:12px; font-weight:700; text-decoration:none; }}
    .term span {{ color:#6f5b2b; font-weight:600; }}
    .warn {{ color:#aa3f31; }}
    footer {{ color:var(--muted); font-size:12px; margin-top:24px; }}
    @media (max-width:980px) {{ .guide-row {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:820px) {{ .summary, .toolbar, .guide-row {{ grid-template-columns:1fr; }} body {{ background-attachment:scroll; }} }}
  </style>
</head>
<body>
  <header>
    <a class="button" href="./index.html">Back to paper library</a>
    <a class="button" href="../index.html#glossary">Back to glossary</a>
    <h1>PDF 术语页码索引</h1>
    <p class="muted">只扫描公开缓存 PDF，生成术语出现次数、页码、跨论文反查和每篇论文的术语导读/页码地图页；不复制论文正文。点击术语会筛出所有含该术语的论文，进入单篇导读页可继续按页打开原 PDF。</p>
    <div class="summary">
      <div><b>{len(index)}</b><span>open cached PDFs scanned</span></div>
      <div><b>{sum(1 for item in index if item.get("term_count"))}</b><span>PDFs with term hits</span></div>
      <div><b>{total_hits}</b><span>term occurrences</span></div>
    </div>
  </header>
  <main>
    <h2>术语索引使用路线</h2>
    <section class="guide-row" aria-label="术语索引使用路线">
      <article class="guide-card"><b>先定位论文</b><p>用搜索框找 NeRIF、PIV-BOST、4D、optical flow 或作者名，进入单篇导读页。</p></article>
      <article class="guide-card"><b>再点页码</b><p>术语命中只做导航，点击页码回到原 PDF 核对图、公式和上下文。</p></article>
      <article class="guide-card"><b>回到选题</b><p>单篇导读页会把论文挂到候选毕设方向，方便你判断是否适合作为模块。</p></article>
      <article class="guide-card"><b>保持边界</b><p>这里只扫描公开缓存 PDF；VPN-only 或订阅全文不进入公开术语索引。</p></article>
    </section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索论文、作者、术语，例如 NeRIF / PINN / Gladstone-Dale / PIV" />
      <select id="sort">
        <option value="default">默认顺序</option>
        <option value="terms">术语数优先</option>
        <option value="year">年份优先</option>
      </select>
    </div>
    <section id="grid" class="grid">
      {''.join(cards)}
    </section>
    <footer>Generated from open cached PDFs with pypdf. Private/WebVPN PDFs are excluded.</footer>
  </main>
  <script>
    const search = document.getElementById('search');
    const sort = document.getElementById('sort');
    const grid = document.getElementById('grid');
    const cards = Array.from(document.querySelectorAll('.card'));
    function apply() {{
      const q = search.value.trim().toLowerCase();
      const terms = q.split(/\\s+/).filter(Boolean);
      const ordered = cards.slice().sort((a,b) => {{
        const da = JSON.parse(a.dataset.text);
        const db = JSON.parse(b.dataset.text);
        if (sort.value === 'terms') return (db.term_count || 0) - (da.term_count || 0);
        if (sort.value === 'year') return String(db.year || '').localeCompare(String(da.year || ''));
        return 0;
      }});
      ordered.forEach(card => {{
        const text = card.innerText.toLowerCase();
        const ok = !terms.length || terms.every(t => text.includes(t));
        card.style.display = ok ? '' : 'none';
        grid.appendChild(card);
      }});
    }}
    const params = new URLSearchParams(location.search);
    const initial = params.get('q') || params.get('term') || '';
    if (initial) search.value = initial;
    search.addEventListener('input', apply);
    sort.addEventListener('change', apply);
    apply();
  </script>
  <script src="../site_link_router.js" data-site-root="../"></script>
</body>
</html>
"""
    OUT_HTML.write_text(clean_trailing_whitespace(html_text), encoding="utf-8")


def main() -> int:
    index = build_index()
    OUT_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    write_reader_pages(index)
    write_html(index)
    print(f"scanned {len(index)} cached PDFs; wrote {OUT_JSON.relative_to(SITE_ROOT)}, {OUT_HTML.relative_to(SITE_ROOT)} and {len(list(OUT_READERS.glob('*.html')))} reader pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
