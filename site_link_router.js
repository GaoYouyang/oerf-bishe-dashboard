(function () {
  const currentScript = document.currentScript;
  const siteRoot = currentScript?.dataset.siteRoot || "./";
  const rootUrl = new URL(siteRoot, location.href);
  const textExtensions = new Set([
    "md", "markdown", "json", "csv", "tsv", "bib", "txt",
    "py", "js", "ts", "jsx", "tsx", "css", "xml", "yaml", "yml", "toml", "sha256"
  ]);
  const assetExtensions = new Set(["pdf", "png", "jpg", "jpeg", "gif", "svg", "webp"]);
  const pageExtensions = new Set(["html", "htm"]);
  const blockedPrefixes = ["private_library/", "tmp_downloads/", ".git/", "_public_pages_export/"];

  function injectStyles() {
    if (document.getElementById("site-link-router-styles")) return;
    const style = document.createElement("style");
    style.id = "site-link-router-styles";
    style.textContent = `
      a.site-routed-link {
        text-decoration-thickness: 1px;
        text-underline-offset: 3px;
      }
      a.site-routed-link:not(.button):not(.tag):not(.nav-link):not(.mini-button):not(.page-title):not(.term-chip):not(.mini-term)::after {
        content: attr(data-route-label);
        display: inline-flex;
        align-items: center;
        max-width: 100%;
        min-height: 18px;
        margin-left: 6px;
        padding: 1px 5px;
        border: 1px solid rgba(49, 95, 147, .18);
        border-radius: 999px;
        background: rgba(238, 244, 251, .86);
        color: #315f93;
        font-size: 10px;
        font-weight: 850;
        line-height: 1.2;
        vertical-align: 1px;
      }
      a.site-routed-link[data-routed="asset"]:not(.button):not(.tag):not(.nav-link):not(.mini-button)::after {
        border-color: rgba(22, 117, 111, .18);
        background: rgba(234, 244, 241, .9);
        color: #16756f;
      }
      a.site-routed-link[data-routed="page"]:not(.button):not(.tag):not(.nav-link):not(.mini-button)::after {
        border-color: rgba(154, 104, 24, .2);
        background: rgba(248, 240, 220, .88);
        color: #8a5c14;
      }
      a.site-routed-link[data-routed="external"]:not(.button):not(.tag):not(.nav-link):not(.mini-button)::after {
        border-color: rgba(85, 96, 104, .18);
        background: rgba(244, 247, 246, .92);
        color: #53636b;
      }
      .buttons a.site-routed-link::after,
      .button-row a.site-routed-link::after,
      .context-actions a.site-routed-link::after,
      .context-nav a.site-routed-link::after,
      .link-cloud a.site-routed-link::after,
      .map-card a.site-routed-link::after,
      .page-links a.site-routed-link::after,
      .pill-row a.site-routed-link::after,
      .resource-actions a.site-routed-link::after,
      .resource-list a.site-routed-link::after,
      .term-actions a.site-routed-link::after,
      .topic-card a.site-routed-link::after {
        display: none !important;
      }
      @media (max-width: 640px) {
        a.site-routed-link:not(.button):not(.tag):not(.nav-link):not(.mini-button)::after {
          display: none !important;
        }
      }
    `;
    document.head.appendChild(style);
  }

  function extensionOf(pathname) {
    const clean = pathname.split("?")[0].split("#")[0];
    const index = clean.lastIndexOf(".");
    return index >= 0 ? clean.slice(index + 1).toLowerCase() : "";
  }

  function relativeToRoot(url) {
    const root = rootUrl.href.endsWith("/") ? rootUrl.href : `${rootUrl.href}/`;
    if (!url.href.startsWith(root)) return "";
    return decodeURIComponent(url.href.slice(root.length).split("#")[0].split("?")[0]);
  }

  function isBlockedPath(path) {
    const clean = String(path || "").replace(/^\.?\//, "");
    return blockedPrefixes.some((prefix) => clean.startsWith(prefix));
  }

  function isAlreadyWrapped(href) {
    return /(^|\/)(document_reader|asset_viewer)\.html\?/.test(href);
  }

  function externalLabel(url) {
    const host = String(url.hostname || "").replace(/^www\./, "").toLowerCase();
    const path = String(url.pathname || "").toLowerCase();
    if (host === "doi.org" || /^\/10\./.test(path)) return "DOI / 出版源";
    if (host.includes("arxiv.org")) return path.includes("/pdf/") ? "arXiv PDF" : "arXiv";
    if (host.includes("github.com")) return "代码 / GitHub";
    if (host.includes("youtube.com") || host.includes("youtu.be") || host.includes("bilibili.com")) return "视频";
    if (host.includes("springer.com") || host.includes("sciencedirect.com") || host.includes("pubs.aip.org") || host.includes("opg.optica.org") || host.includes("iopscience.iop.org") || host.includes("dl.acm.org")) return "出版社页";
    if (host.includes("figshare.com") || host.includes("zenodo.org") || host.includes("osf.io")) return "数据 / supplement";
    if (host.includes("readthedocs.io") || host.includes("docs.") || host.includes("opencv.org")) return "文档";
    return "外部来源";
  }

  function labelFor(type, ext, rel, url) {
    if (type === "document") return "文档";
    if (type === "asset" && ext === "pdf") return "站内 PDF/资源查看器";
    if (type === "asset") return "站内图像/资源查看器";
    if (type === "external" && url) return externalLabel(url);
    if (type === "page") {
      if (/^asset_viewer\.html/.test(rel)) {
        try {
          const asset = new URL(url.href).searchParams.get("asset") || "";
          if (/\.pdf(?:$|[#?])/.test(asset)) return "站内 PDF/资源查看器";
          if (/\.(png|jpe?g|gif|svg|webp)(?:$|[#?])/.test(asset)) return "站内图像/资源查看器";
        } catch {}
        return "资源页";
      }
      if (/^document_reader\.html/.test(rel)) return "文档页";
      if (/^paper_library\/readers\//.test(rel)) return "术语导读";
      if (/^paper_library\/paper_detail\.html/.test(rel)) return "论文详情";
      if (/^paper_library\/term_index\.html/.test(rel)) return "术语索引";
      if (/^paper_library\/index\.html/.test(rel)) return "论文库";
      if (/^topic_deep_dive\.html/.test(rel)) return "选题详情";
      return "专题页";
    }
    return "站内页";
  }

  function decorate(anchor, type, ext, rel, url) {
    anchor.dataset.routed = type;
    anchor.dataset.routeLabel = labelFor(type, ext, rel, url);
    anchor.classList.add("site-routed-link");
    if (!anchor.getAttribute("title")) {
      anchor.setAttribute("title", `打开${labelFor(type, ext, rel, url)}：${anchor.textContent.trim() || anchor.getAttribute("href") || ""}`);
    }
  }

  function shouldSkip(anchor) {
    const href = anchor.getAttribute("href") || "";
    if (!href || href.startsWith("#")) return true;
    if (anchor.id === "rawLink" || anchor.dataset.raw === "true" || anchor.hasAttribute("download")) return true;
    if (/^(mailto:|tel:|javascript:)/i.test(href)) return true;
    return false;
  }

  function routeHref(anchor) {
    if (shouldSkip(anchor)) return;
    const href = anchor.getAttribute("href") || "";
    let url;
    try {
      url = new URL(href, location.href);
    } catch {
      return;
    }

    if (url.origin !== location.origin || url.protocol !== location.protocol) {
      if (/^https?:$/i.test(url.protocol)) decorate(anchor, "external", "", url.hostname, url);
      return;
    }
    const rel = relativeToRoot(url);
    if (!rel || isBlockedPath(rel)) return;

    const ext = extensionOf(url.pathname);
    if (isAlreadyWrapped(href)) {
      decorate(anchor, "page", ext || "html", rel, url);
      return;
    }

    if (textExtensions.has(ext)) {
      const routed = new URL("document_reader.html", rootUrl);
      routed.searchParams.set("doc", rel);
      if (url.hash) routed.searchParams.set("anchor", decodeURIComponent(url.hash.slice(1)));
      anchor.href = routed.href;
      decorate(anchor, "document", ext, rel, url);
      return;
    }

    if (assetExtensions.has(ext)) {
      const routed = new URL("asset_viewer.html", rootUrl);
      routed.searchParams.set("asset", rel);
      if (url.hash) routed.hash = url.hash;
      anchor.href = routed.href;
      decorate(anchor, "asset", ext, rel, url);
      return;
    }

    if (pageExtensions.has(ext)) {
      decorate(anchor, "page", ext, rel, url);
    }
  }

  function routeLinks() {
    injectStyles();
    document.querySelectorAll("a[href]").forEach(routeHref);
  }

  let pending = false;
  function scheduleRouteLinks() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => {
      pending = false;
      routeLinks();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", routeLinks, { once: true });
  } else {
    routeLinks();
  }

  if ("MutationObserver" in window) {
    const observeRoot = document.documentElement || document.body;
    if (!observeRoot) return;
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        if (mutation.type === "childList" && mutation.addedNodes.length) {
          scheduleRouteLinks();
          return;
        }
        if (mutation.type === "attributes" && mutation.attributeName === "href") {
          scheduleRouteLinks();
          return;
        }
      }
    });
    observer.observe(observeRoot, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["href"]
    });
  }
})();
