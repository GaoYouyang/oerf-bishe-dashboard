from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent


def tag(text: str) -> str:
    text = text.lower()
    if text == "must":
        return "must"
    if text in {"support", "supporting"}:
        return "support"
    return "optional"


def esc(value: str) -> str:
    return html.escape(value or "", quote=True)


def asset_viewer_href(local_path: str) -> str:
    return "../asset_viewer.html?asset=" + quote(f"paper_library/{local_path}", safe="")


def card(paper: dict) -> str:
    local = paper.get("local_pdf") or ""
    preview = paper.get("preview") or ""
    pdf_link = asset_viewer_href(local) if local else paper.get("online_pdf_url") or paper.get("landing_url") or "#"
    detail_link = f"paper_detail.html?id={quote(paper['id'], safe='')}"
    preview_html = (
        f'<a href="{esc(detail_link)}">'
        f'<img src="{esc(preview)}" alt="{esc(paper["title"])} preview" />'
        "</a>"
        if preview
        else '<div class="no-preview">PDF / DOI</div>'
    )
    local_badge = "cached PDF" if local else "online link"
    doi = paper.get("doi") or "no DOI"
    local_button = (
        f'<a class="button" href="{esc(asset_viewer_href(local))}" target="_blank" rel="noreferrer">Open local PDF</a>'
        if local
        else ""
    )
    reader_button = (
        f'<a class="button" href="readers/{esc(paper["id"])}.html">术语导读/页码地图</a>'
        if local
        else ""
    )
    detail_button = f'<a class="button primary" href="{esc(detail_link)}">论文详情页</a>'
    online_button = (
        f'<a class="button" href="{esc(paper.get("online_pdf_url", ""))}" target="_blank" rel="noreferrer">Open online PDF</a>'
        if paper.get("online_pdf_url")
        else ""
    )
    landing_button = (
        f'<a class="button" href="{esc(paper.get("landing_url", ""))}" target="_blank" rel="noreferrer">Publisher / source</a>'
        if paper.get("landing_url")
        else ""
    )
    return f"""
      <article class="paper-card" data-group="{esc(paper['group'])}" data-priority="{esc(paper['priority'])}">
        <div class="preview">{preview_html}</div>
        <div class="paper-body">
          <div class="meta-row">
            <span class="pill {tag(paper['priority'])}">{esc(paper['priority'])}</span>
            <span class="pill">{esc(paper['group'])}</span>
            <span class="pill">{esc(local_badge)}</span>
          </div>
          <h3><a href="{esc(detail_link)}">{esc(paper['title'])}</a></h3>
          <p class="authors">{esc(paper['authors'])}</p>
          <p class="venue">{esc(paper['venue'])} · {esc(paper['year'])} · {esc(doi)}</p>
          <p><strong>Why:</strong> {esc(paper['why'])}</p>
          <p><strong>Extract:</strong> {esc(paper['extract'])}</p>
          <p class="access">{esc(paper['access'])}</p>
          <div class="buttons">{detail_button}{reader_button}{local_button}{online_button}{landing_button}</div>
        </div>
      </article>
    """.strip()


def main() -> None:
    papers = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    groups = sorted({p["group"] for p in papers})
    priority_order = {"must": 0, "support": 1, "supporting": 2, "optional": 3, "advanced": 4}
    priorities = sorted(
        {p["priority"] for p in papers},
        key=lambda value: (priority_order.get(value, 99), value),
    )
    group_options = "\n".join(f'<option value="{esc(g)}">{esc(g)}</option>' for g in groups)
    priority_options = "\n".join(
        f'<option value="{esc(priority)}">{esc(priority)}</option>'
        for priority in priorities
    )
    cards = "\n".join(card(p) for p in papers)
    cached_count = sum(1 for p in papers if p.get("local_pdf"))
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OERF / BOST Paper Library</title>
  <link rel="icon" href="data:," />
  <style>
    :root {{
      --text: #1f2933;
      --muted: #607080;
      --line: #d9e2e4;
      --panel: #ffffff;
      --soft: #eef7f4;
      --green: #147b68;
      --blue: #2866b1;
      --amber: #9a6a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #f7faf9;
      line-height: 1.6;
    }}
    header {{
      padding: 34px clamp(18px, 4vw, 60px);
      border-bottom: 1px solid var(--line);
      background: linear-gradient(180deg, #ffffff, #f2f8f6);
    }}
    h1 {{ margin: 0; font-size: clamp(30px, 4vw, 54px); letter-spacing: 0; }}
    header p {{ max-width: 980px; color: var(--muted); font-size: 17px; }}
    main {{ padding: 22px clamp(18px, 4vw, 60px) 60px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr 220px 180px;
      gap: 10px;
      margin: 18px 0 22px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      font-size: 15px;
      background: #fff;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 20px;
    }}
    .summary div, .notice {{
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .summary b {{ display: block; font-size: 30px; color: var(--green); }}
    .notice {{ margin: 16px 0; color: var(--muted); }}
    .paper-grid {{ display: grid; gap: 14px; }}
    .paper-card {{
      display: grid;
      grid-template-columns: 190px 1fr;
      gap: 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }}
    .preview {{
      min-height: 230px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fbfa;
      display: flex;
      align-items: center;
      justify-content: center;
      overflow: hidden;
    }}
    .preview img {{ width: 100%; height: 100%; object-fit: cover; object-position: top center; display: block; }}
    .no-preview {{ color: var(--muted); font-weight: 700; }}
    .paper-body h3 {{ margin: 8px 0 6px; font-size: 21px; line-height: 1.25; }}
    .authors, .venue, .access {{ color: var(--muted); margin: 4px 0; }}
    .meta-row, .buttons {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .term-strip {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      margin: 10px 0 0;
      min-height: 26px;
    }}
    .term-chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 8px;
      background: #f7f0df;
      color: #8b5d0b;
      border: 1px solid #ead7aa;
      font-size: 12px;
      font-weight: 700;
      text-decoration: none;
    }}
    .term-chip:hover {{ text-decoration: none; filter: brightness(0.98); }}
    .pill {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      background: var(--soft);
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
    }}
    .pill.must {{ background: #e7f3ee; color: var(--green); }}
    .pill.support {{ background: #eef4ff; color: var(--blue); }}
    .button {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 38px;
      padding: 8px 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      color: var(--text);
      text-decoration: none;
      font-weight: 700;
      background: #fff;
    }}
    .button.primary {{ background: var(--green); color: #fff; border-color: var(--green); }}
    footer {{ color: var(--muted); padding-top: 22px; }}
    @media (max-width: 820px) {{
      .toolbar, .summary, .paper-card {{ grid-template-columns: 1fr; }}
      .preview {{ min-height: 180px; }}
    }}
  </style>
</head>
<body>
  <header>
    <a class="button" href="../index.html">Back to dashboard</a>
    <a class="button" href="../pdf_cache_audit.html">PDF cache audit</a>
    <a class="button" href="term_index.html">PDF term index</a>
    <a class="button" href="../document_reader.html?doc=paper_library%2Fpapers.json">Metadata reader</a>
    <h1>OERF / BOST 论文 PDF 跳转库</h1>
    <p>当前收录 {len(papers)} 篇核心论文与开放阅读入口，其中 {cached_count} 篇已缓存开放 PDF。缓存原则：只缓存 arXiv、NASA、开放许可或明确公开 PDF；Springer/ACM/Elsevier 等版权不明确的论文只提供出版社或在线 PDF 链接。</p>
    <div class="summary">
      <div><b>{len(papers)}</b><span>papers / links</span></div>
      <div><b>{cached_count}</b><span>cached open PDFs</span></div>
      <div><b>{len(groups)}</b><span>topic groups</span></div>
    </div>
  </header>
  <main>
    <div class="notice">建议阅读顺序：先读 NeRIF、PIV-BOST、4D BOST 三篇何远哲主线，再读 NeDF/NRIP/open BOS dataset 作为方法邻居，最后读 Computational Flow Visualization 和 BOS 综述补总框架。</div>
    <div class="notice">术语辅助阅读：公开缓存 PDF 会生成“术语导读/页码地图”，按单篇论文列出核心术语、解释、易混点、页码跳转、按页术语地图和跨论文反查，尽量接近维基百科式的读论文导航，但不复制论文正文。</div>
    <p><a class="button" href="../fulltext_access_queue.html">受限全文访问队列</a><a class="button" href="../xmu_vpn_private_library_protocol.html">私有库边界说明</a></p>
    <div class="toolbar">
      <input id="search" type="search" placeholder="搜索 title / author / DOI / why，例如 NeRIF, PIV-BOST, 4D, open data" />
      <select id="group">
        <option value="all">All groups</option>
        {group_options}
      </select>
      <select id="priority">
        <option value="all">All priority</option>
        {priority_options}
      </select>
    </div>
    <section id="papers" class="paper-grid">
      {cards}
    </section>
    <footer>Generated from <code>paper_library/papers.json</code>. Last updated: 2026-07-15.</footer>
  </main>
  <script src="../glossary_terms.js"></script>
  <script>
    const glossaryTerms = window.OERF_GLOSSARY_TERMS || [];
    function normalizeText(value) {{
      return (value || '').toLowerCase().replace(/\\s+/g, ' ');
    }}
    function termMatches(text, term) {{
      const aliases = [term.zh, term.en, term.keywords, ...(term.aliases || [])];
      return aliases
        .filter(Boolean)
        .some(alias => normalizeText(text).includes(normalizeText(alias)));
    }}
    function addTermChips() {{
      cards.forEach((card) => {{
        const text = card.innerText;
        const matched = glossaryTerms.filter(term => termMatches(text, term)).slice(0, 8);
        if (!matched.length) return;
        const strip = document.createElement('div');
        strip.className = 'term-strip';
        matched.forEach(term => {{
          const a = document.createElement('a');
          a.className = 'term-chip';
          a.href = '../index.html?term=' + encodeURIComponent(term.zh) + '#glossary';
          a.target = '_blank';
          a.rel = 'noreferrer';
          a.title = term.explain;
          a.textContent = term.zh;
          strip.appendChild(a);
        }});
        card.querySelector('.paper-body')?.appendChild(strip);
      }});
    }}
    const search = document.getElementById('search');
    const group = document.getElementById('group');
    const priority = document.getElementById('priority');
    const cards = Array.from(document.querySelectorAll('.paper-card'));
    function applyFilters() {{
      const q = search.value.trim().toLowerCase();
      const terms = q.split(/\\s+/).filter(Boolean);
      const g = group.value;
      const p = priority.value;
      cards.forEach((card) => {{
        const text = card.innerText.toLowerCase();
        const okQ = !terms.length || terms.every((term) => text.includes(term));
        const okG = g === 'all' || card.dataset.group === g;
        const okP = p === 'all' || card.dataset.priority === p;
        card.style.display = okQ && okG && okP ? '' : 'none';
      }});
    }}
    search.addEventListener('input', applyFilters);
    group.addEventListener('change', applyFilters);
    priority.addEventListener('change', applyFilters);
    addTermChips();
    const params = new URLSearchParams(window.location.search);
    const initialQuery = params.get('q');
    const initialGroup = params.get('group');
    const initialPriority = params.get('priority');
    if (initialQuery) search.value = initialQuery;
    if (initialGroup && Array.from(group.options).some(option => option.value === initialGroup)) group.value = initialGroup;
    if (initialPriority && Array.from(priority.options).some(option => option.value === initialPriority)) priority.value = initialPriority;
    applyFilters();
  </script>
  <script src="../site_link_router.js" data-site-root="../"></script>
</body>
</html>
"""
    (ROOT / "index.html").write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
