(() => {
  const files = Array.isArray(window.TDBOST_FILES) ? window.TDBOST_FILES : [];
  const fileGroups = document.getElementById("fileGroups");
  const fileSearch = document.getElementById("fileSearch");
  const codeSearch = document.getElementById("codeSearch");
  const codeLines = document.getElementById("codeLines");
  const matchStatus = document.getElementById("matchStatus");
  const prevMatch = document.getElementById("prevMatch");
  const nextMatch = document.getElementById("nextMatch");
  const filePath = document.getElementById("filePath");
  const fileTitle = document.getElementById("fileTitle");
  const fileDescription = document.getElementById("fileDescription");
  const fileLevel = document.getElementById("fileLevel");
  const fileLines = document.getElementById("fileLines");
  const fileFocus = document.getElementById("fileFocus");
  const fileInput = document.getElementById("fileInput");
  const fileOutput = document.getElementById("fileOutput");
  const upstreamLink = document.getElementById("upstreamLink");
  const copyLink = document.getElementById("copyLink");
  const copyStatus = document.getElementById("copyStatus");

  let activeFile = null;
  let activeSource = "";
  let matchRows = [];
  let matchIndex = -1;

  function sourceUrl(path) {
    return "./source/" + path.split("/").map(encodeURIComponent).join("/");
  }

  function upstreamUrl(path) {
    return "https://github.com/Hyz617/TDBOST/blob/3393ca700fd0447685caf4314d87e0f99fc3ef12/" +
      path.split("/").map(encodeURIComponent).join("/");
  }

  function groupedFiles(query = "") {
    const normalized = query.trim().toLowerCase();
    const filtered = files.filter((file) => {
      if (!normalized) return true;
      return [
        file.path,
        file.title,
        file.group,
        file.level,
        file.description,
        file.focus
      ].join(" ").toLowerCase().includes(normalized);
    });

    return filtered.reduce((groups, file) => {
      if (!groups.has(file.group)) groups.set(file.group, []);
      groups.get(file.group).push(file);
      return groups;
    }, new Map());
  }

  function renderFileGroups(query = "") {
    const groups = groupedFiles(query);
    fileGroups.replaceChildren();

    if (!groups.size) {
      const empty = document.createElement("div");
      empty.className = "file-empty";
      empty.textContent = "没有匹配文件。可以搜索 main、光线、训练、MMmodel 或 loss。";
      fileGroups.appendChild(empty);
      return;
    }

    groups.forEach((groupFiles, groupName) => {
      const section = document.createElement("section");
      const heading = document.createElement("h3");
      const count = document.createElement("span");
      const list = document.createElement("div");

      heading.className = "file-group-title";
      heading.textContent = groupName;
      count.textContent = String(groupFiles.length);
      heading.appendChild(count);
      list.className = "file-list";

      groupFiles.forEach((file) => {
        const button = document.createElement("button");
        const meta = document.createElement("span");
        button.type = "button";
        button.className = "file-button";
        button.dataset.path = file.path;
        button.setAttribute("aria-current", String(activeFile && activeFile.path === file.path));
        button.append(document.createTextNode(file.path));
        meta.textContent = file.level + " · " + file.lines + " 行";
        button.appendChild(meta);
        button.addEventListener("click", () => loadFile(file.path));
        list.appendChild(button);
      });

      section.append(heading, list);
      fileGroups.appendChild(section);
    });
  }

  function setFileMetadata(file) {
    filePath.textContent = file.path;
    fileTitle.textContent = file.title;
    fileDescription.textContent = file.description;
    fileLevel.textContent = file.level;
    fileLines.textContent = file.lines + " 行";
    fileFocus.textContent = "第一遍只找：" + file.focus;
    fileInput.textContent = file.input;
    fileOutput.textContent = file.output;
    upstreamLink.href = upstreamUrl(file.path);
    document.title = file.path + " 中文导读 | TDBOST";
  }

  function appendHighlightedText(container, line, query) {
    if (!query) {
      container.textContent = line || " ";
      return false;
    }

    const lowerLine = line.toLowerCase();
    const lowerQuery = query.toLowerCase();
    let cursor = 0;
    let found = false;

    while (cursor < line.length) {
      const foundAt = lowerLine.indexOf(lowerQuery, cursor);
      if (foundAt < 0) {
        container.appendChild(document.createTextNode(line.slice(cursor)));
        break;
      }
      found = true;
      if (foundAt > cursor) {
        container.appendChild(document.createTextNode(line.slice(cursor, foundAt)));
      }
      const mark = document.createElement("mark");
      mark.textContent = line.slice(foundAt, foundAt + query.length);
      container.appendChild(mark);
      cursor = foundAt + query.length;
    }

    if (!line.length) container.textContent = " ";
    return found;
  }

  function renderCode(query = "") {
    const normalized = query.trim();
    const fragment = document.createDocumentFragment();
    matchRows = [];
    matchIndex = -1;

    activeSource.split("\n").forEach((line, index) => {
      const row = document.createElement("div");
      const number = document.createElement("span");
      const text = document.createElement("span");
      const lineNumber = index + 1;
      const trimmed = line.trimStart();

      row.className = "code-row";
      row.id = "L" + lineNumber;
      if (trimmed.startsWith("#")) row.classList.add("is-comment");
      if (trimmed.startsWith('"""') || trimmed.startsWith("'''")) row.classList.add("is-doc");

      number.className = "line-number";
      number.textContent = String(lineNumber);
      number.setAttribute("aria-hidden", "true");
      text.className = "line-text";

      const matched = appendHighlightedText(text, line, normalized);
      if (matched) matchRows.push(row);
      row.append(number, text);
      fragment.appendChild(row);
    });

    codeLines.replaceChildren(fragment);
    updateMatchStatus();
  }

  function updateMatchStatus() {
    const hasMatches = matchRows.length > 0;
    if (!codeSearch.value.trim()) {
      matchStatus.textContent = "输入关键词";
    } else if (!hasMatches) {
      matchStatus.textContent = "0 处";
    } else if (matchIndex < 0) {
      matchStatus.textContent = matchRows.length + " 处";
    } else {
      matchStatus.textContent = (matchIndex + 1) + " / " + matchRows.length;
    }
    prevMatch.disabled = !hasMatches;
    nextMatch.disabled = !hasMatches;
  }

  function goToMatch(delta) {
    if (!matchRows.length) return;
    if (matchIndex >= 0) matchRows[matchIndex].classList.remove("is-match");
    matchIndex = (matchIndex + delta + matchRows.length) % matchRows.length;
    const row = matchRows[matchIndex];
    row.classList.add("is-match");
    row.scrollIntoView({ behavior: "smooth", block: "center" });
    updateMatchStatus();
  }

  async function loadFile(path, options = {}) {
    const file = files.find((item) => item.path === path) || files[0];
    if (!file) return;

    activeFile = file;
    setFileMetadata(file);
    renderFileGroups(fileSearch.value);
    codeLines.innerHTML = '<div class="loading">正在载入中文注释源码…</div>';

    try {
      const response = await fetch(sourceUrl(file.path), { cache: "no-store" });
      if (!response.ok) throw new Error("HTTP " + response.status);
      activeSource = await response.text();
      codeSearch.value = "";
      renderCode();

      const url = new URL(window.location.href);
      url.searchParams.set("file", file.path);
      if (!options.line) url.searchParams.delete("line");
      history.replaceState({}, "", url);

      const requestedLine = Number(options.line || 0);
      if (requestedLine > 0) {
        requestAnimationFrame(() => {
          const target = document.getElementById("L" + requestedLine);
          if (target) target.scrollIntoView({ block: "center" });
        });
      }
    } catch (error) {
      activeSource = "";
      codeLines.innerHTML = '<div class="load-error">源码载入失败。请从 GitHub Pages 地址打开本页，或检查该文件是否已随页面发布。</div>';
      matchStatus.textContent = "载入失败";
      console.error(error);
    }
  }

  fileSearch.addEventListener("input", () => renderFileGroups(fileSearch.value));
  codeSearch.addEventListener("input", () => renderCode(codeSearch.value));
  prevMatch.addEventListener("click", () => goToMatch(-1));
  nextMatch.addEventListener("click", () => goToMatch(1));

  document.querySelectorAll("[data-file]").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      loadFile(link.dataset.file);
      const reader = document.getElementById("reader");
      if (reader) reader.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  copyLink.addEventListener("click", async () => {
    const url = new URL(window.location.href);
    url.searchParams.set("file", activeFile ? activeFile.path : "run.py");
    try {
      await navigator.clipboard.writeText(url.toString());
      copyStatus.textContent = "当前文件链接已复制。";
    } catch (_error) {
      copyStatus.textContent = "浏览器未允许复制，请直接复制地址栏链接。";
    }
  });

  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      codeSearch.focus();
    }
    if (event.key === "Enter" && document.activeElement === codeSearch && matchRows.length) {
      event.preventDefault();
      goToMatch(1);
    }
  });

  const params = new URLSearchParams(window.location.search);
  const requestedFile = params.get("file");
  const initialFile = files.some((file) => file.path === requestedFile) ? requestedFile : "run.py";
  renderFileGroups();
  loadFile(initialFile, { line: params.get("line") });
})();
