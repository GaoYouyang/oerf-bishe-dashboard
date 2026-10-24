/* Calendar grouping uses explicit event dates, never the browser's current day. */
(() => {
  const monthNames = ["January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"];

  function groupEntries(root, entries) {
    const groups = new Map();
    for (const entry of entries) {
      const date = entry.dataset.date;
      if (!/^\d{4}-\d{2}-\d{2}$/.test(date || "")) {
        throw new Error("Research log entry requires an explicit ISO event date");
      }
      const month = date.slice(0, 7);
      if (!groups.has(month)) groups.set(month, []);
      groups.get(month).push(entry);
    }
    // Keep old phase anchors for incoming historical links, but do not let their
    // stale ranges label events that were appended months later.
    const historicalPhases = [...root.querySelectorAll(".phase")];
    historicalPhases.forEach(phase => {
      const dates = [...phase.querySelectorAll(".day-entry")].map(entry => entry.dataset.date).sort();
      if (dates.length) phase.dataset.calendarTarget = `log-month-${dates.at(-1).slice(0, 7)}`;
    });
    const phases = [];
    for (const month of [...groups.keys()].sort().reverse()) {
      const items = groups.get(month).sort((a, b) => b.dataset.date.localeCompare(a.dataset.date));
      const [year, number] = month.split("-");
      const phase = document.createElement("section");
      phase.className = "phase";
      phase.dataset.calendarMonth = month;
      phase.dataset.phaseOrder = year + number;
      phase.id = `log-month-${month}`;
      const heading = document.createElement("div");
      heading.className = "phase-heading";
      const title = document.createElement("h2");
      title.dataset.i18nZh = `${year}年${Number(number)}月`;
      title.dataset.i18nEn = `${monthNames[Number(number) - 1]} ${year}`;
      title.textContent = title.dataset.i18nZh;
      const range = document.createElement("div");
      range.className = "range";
      range.textContent = `${items.at(-1).dataset.date} — ${items[0].dataset.date}`;
      heading.append(title, range);
      const timeline = document.createElement("div");
      timeline.className = "timeline";
      timeline.append(...items);
      phase.append(heading, timeline);
      root.append(phase);
      phases.push(phase);
    }
    historicalPhases.forEach(phase => { phase.hidden = true; });
    return phases;
  }

  window.OERFLogChronology = Object.freeze({ groupEntries });
})();
