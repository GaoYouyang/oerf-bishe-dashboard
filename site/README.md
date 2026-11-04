# Main site pages

Five pages share `site/style.css` and `site/site.js`: `index.html`, `method.html`, `results.html`,
`log.html`, `archive.html`. Everything else in the repository is the pre-2026-11 site, kept
unchanged and reachable from `archive.html`. The old homepage is `legacy-home.html`.

The detailed daily log `operator-learning/daily-progress.html` (linked from `log.html`) has the
main-site header and look via `site/daily-log.css`, loaded after `site/style.css`; its entries,
filters and calendar script are unchanged. It keeps the older pages' `data-i18n-zh/en` translator
(`assets/site-language.js`). Both language keys (`oerf-lang`, `oerf-site-language`) are written on
every switch, so the choice carries across old and new pages.

- Bilingual text: write `<span lang="zh">…</span><span lang="en">…</span>` side by side (or put
  `lang` on a block element). The button in the header switches language and remembers it.
- New log entry: copy one `<article class="entry">` block in `log.html`, place it first, edit the
  date, title and bullets.
- New result: copy one `<section class="result">` block in `results.html`; tag classes are
  `pos`, `mixed`, `neg`, `plan`.
- No build step and no external libraries; GitHub Pages serves the files as they are.
