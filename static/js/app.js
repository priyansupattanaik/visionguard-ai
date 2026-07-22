/* ============================================================
   VisionGuard — App Controller (Vanilla JS)
   All original API contracts preserved.
   ============================================================ */

(() => {
  "use strict";

  // ---- State ----
  const state = { query: "", matches: [] };

  // ---- DOM cache ----
  const $ = (sel) => document.querySelector(sel);
  const els = {
    systemStatus:       $("#systemStatus"),
    statusText:         $("#systemStatus .status-text"),
    statusDot:          $("#systemStatus .status-dot"),
    sampleSelect:       $("#sampleSelect"),
    videoPreviewWrap:   $("#videoPreviewWrap"),
    videoPreview:       $("#videoPreview"),
    videoPreviewName:   $("#videoPreviewName"),
    videoUpload:        $("#videoUpload"),
    scanForm:           $("#scanForm"),
    scanButton:         $("#scanButton"),
    scanStatus:         $("#scanStatus"),
    scanProgress:       $("#scanProgress"),
    scanProgressFill:   $("#scanProgressFill"),
    scanProgressLabel:  $("#scanProgressLabel"),
    fileUploadZone:     $("#fileUploadZone"),
    zeroQueryPanel:     $("#zeroQueryPanel"),
    zeroQueryButton:    $("#zeroQueryButton"),
    zeroQuerySummary:   $("#zeroQuerySummary"),
    zeroQueryInventory: $("#zeroQueryInventory"),
    zeroQueryTimeline:  $("#zeroQueryTimeline"),
    queryInput:         $("#queryInput"),
    queryButton:        $("#queryButton"),
    queryStatus:        $("#queryStatus"),
    verificationStatus: $("#verificationStatus"),
    resultsList:        $("#resultsList"),
    resultsCount:       $("#resultsCount"),
    exportButton:       $("#exportButton"),
    exportStatus:       $("#exportStatus"),
    downloadList:       $("#downloadList"),
    toastContainer:     $("#toastContainer"),
  };

  // ---- Utilities ----
  function setMessage(el, text, kind = "") {
    if (!el) return;
    el.textContent = text || "";
    el.className = kind ? `status-msg ${kind}` : "status-msg";
  }

  function setVerification(label, warning) {
    const el = els.verificationStatus;
    if (!el) return;
    const span = el.querySelector("span") || el;
    span.textContent = warning ? `${label}. ${warning}` : label;
    el.className = warning ? "verification-chip warn" : "verification-chip";
  }

  async function fetchJson(url, options = {}) {
    const res = await fetch(url, options);
    const data = await res.json();
    if (!res.ok || data.ok === false) {
      throw new Error(data.message || `Request failed with ${res.status}`);
    }
    return data;
  }

  // ---- Toast system ----
  function showToast(message, type = "info", duration = 4000) {
    const container = els.toastContainer;
    if (!container) return;

    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;

    const iconSvgs = {
      success: `<svg class="toast__icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.857-9.809a.75.75 0 00-1.214-.882l-3.483 4.79-1.88-1.88a.75.75 0 10-1.06 1.061l2.5 2.5a.75.75 0 001.137-.089l4-5.5z" clip-rule="evenodd"/></svg>`,
      error: `<svg class="toast__icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z" clip-rule="evenodd"/></svg>`,
      info: `<svg class="toast__icon" viewBox="0 0 20 20" fill="currentColor" width="16" height="16"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a.75.75 0 000 1.5h.253a.25.25 0 01.244.304l-.459 2.066A1.75 1.75 0 0010.747 15H11a.75.75 0 000-1.5h-.253a.25.25 0 01-.244-.304l.459-2.066A1.75 1.75 0 009.253 9H9z" clip-rule="evenodd"/></svg>`,
    };

    toast.innerHTML = `${iconSvgs[type] || iconSvgs.info}<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.classList.add("out");
      toast.addEventListener("animationend", () => toast.remove());
    }, duration);
  }

  // ---- Button loading state ----
  function setButtonLoading(btn, loading) {
    if (!btn) return;
    btn.disabled = loading;
    btn.classList.toggle("loading", loading);
  }

  // ---- Progress bar ----
  function showProgress(pct, label) {
    if (!els.scanProgress) return;
    els.scanProgress.style.display = "";
    if (els.scanProgressFill) els.scanProgressFill.style.width = `${Math.min(100, pct)}%`;
    if (els.scanProgressLabel) els.scanProgressLabel.textContent = label || "";
  }

  function hideProgress() {
    if (els.scanProgress) els.scanProgress.style.display = "none";
  }

  async function readScanStream(options) {
    const response = await fetch("/api/scan", { ...options, headers: { ...(options.headers || {}), Accept: "application/x-ndjson" } });
    if (!response.ok || !response.body) throw new Error(`Scan request failed with ${response.status}`);
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = ""; let result = null;
    while (true) {
      const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
      const lines = buffer.split("\n"); buffer = lines.pop();
      for (const line of lines) { if (!line) continue; const event = JSON.parse(line);
        if (event.kind === "progress") { const match = event.status.match(/(\d+)%/); showProgress(match ? Number(match[1]) : 8, event.status); setMessage(els.scanStatus, event.status, "scanning"); }
        if (event.kind === "error") throw new Error(event.message);
        if (event.kind === "done") result = event;
      }
      if (done) break;
    }
    if (!result) throw new Error("Scan did not complete."); return result;
  }

  // ---- API: Load status ----
  async function loadStatus() {
    try {
      const data = await fetchJson("/api/status");
      if (els.statusText) els.statusText.textContent = data.status;
      setVerification(data.verification_label, data.warning);
    } catch (err) {
      if (els.statusText) els.statusText.textContent = err.message;
    }
  }

  // ---- API: Load assets ----
  async function loadAssets() {
    try {
      const data = await fetchJson("/api/assets");
      if (!els.sampleSelect) return;
      els.sampleSelect.innerHTML = "";
      for (const asset of data.assets) {
        const option = document.createElement("option");
        option.value = asset.name;
        option.textContent = `${asset.name} (${Math.round(asset.size / 1024)} KB)`;
        option.dataset.url = asset.url;
        els.sampleSelect.appendChild(option);
      }
      updateSamplePreview();
    } catch (err) {
      setMessage(els.scanStatus, err.message, "error");
    }
  }

  function showVideoPreview(url, name) {
    if (!els.videoPreview || !els.videoPreviewWrap || !url) return;
    els.videoPreview.src = url;
    els.videoPreviewWrap.style.display = "";
    if (els.videoPreviewName) els.videoPreviewName.textContent = name || "Selected video";
  }

  function updateSamplePreview() {
    const option = els.sampleSelect?.selectedOptions?.[0];
    if (option) showVideoPreview(option.dataset.url, option.value);
  }

  // ---- Render matches ----
  function renderMatches(matches) {
    state.matches = matches;
    if (els.downloadList) els.downloadList.innerHTML = "";

    if (!matches.length) {
      if (els.resultsList) {
        els.resultsList.className = "results-list results-list--empty";
        els.resultsList.innerHTML = `
          <div class="empty-state">
            <svg viewBox="0 0 20 20" fill="currentColor" width="40" height="40" opacity=".25"><path fill-rule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clip-rule="evenodd"/></svg>
            <p>No strong matches found.</p>
          </div>`;
      }
      if (els.resultsCount) els.resultsCount.style.display = "none";
      return;
    }

    if (els.resultsCount) {
      els.resultsCount.textContent = matches.length;
      els.resultsCount.style.display = "";
    }

    if (!els.resultsList) return;
    els.resultsList.className = "results-list";
    els.resultsList.innerHTML = "";

    matches.forEach((match, i) => {
      const card = document.createElement("article");
      card.className = "match-card";
      card.style.animationDelay = `${i * 0.06}s`;

      const tracksLabel = match.tracks && match.tracks.length
        ? `<span>🔗 ${match.tracks.length} track(s)</span>` : "";

      card.innerHTML = `
        <input type="checkbox" value="${match.id}" aria-label="Select ${match.label}">
        <div>
          <div class="match-card__title">${match.label}</div>
          <div class="match-card__meta">
            <span>⏱ ${match.start}s – ${match.end}s</span>
            <span>📍 peak ${match.peak_ts}s</span>
            <span>📊 ${match.score}</span>
            ${tracksLabel}
          </div>
          <div class="match-card__summary">${match.summary}</div>
          <span class="match-card__mode">${match.verification_label}</span>
        </div>
      `;
      els.resultsList.appendChild(card);
    });
  }

  // ---- Scan form ----
  if (els.scanForm) {
    els.scanForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      setButtonLoading(els.scanButton, true);
      setMessage(els.scanStatus, "");
      showProgress(5, "Preparing scan…");
      renderMatches([]);

      try {
        let options;
        if (els.videoUpload && els.videoUpload.files.length) {
          const form = new FormData();
          form.append("video", els.videoUpload.files[0]);
          options = { method: "POST", body: form };
        } else {
          options = {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sample: els.sampleSelect ? els.sampleSelect.value : "" }),
          };
        }

        showProgress(25, "Scanning video…");
        const data = await readScanStream(options);
        showProgress(100, "Complete");

        const counts = Object.entries(data.object_counts || {})
          .map(([name, count]) => `${name}: ${count}`)
          .join(", ");

        setMessage(els.scanStatus, `${data.message} ${data.video}: ${data.indexed_windows} windows. ${counts}`, "success");
        setVerification(data.backend.verification_label, (data.warnings || [])[0] || "");

        showToast(`Scan complete — ${data.indexed_windows} windows indexed`, "success");

        await loadStatus();

        // Show zero-query panel
        if (els.zeroQueryPanel) {
          els.zeroQueryPanel.style.display = "";
          els.zeroQueryPanel.classList.add("anim-entry");
          if (els.zeroQueryInventory) els.zeroQueryInventory.innerHTML = "";
          if (els.zeroQueryTimeline) els.zeroQueryTimeline.innerHTML = "";
          setMessage(els.zeroQuerySummary, "Click 'Generate Analysis' for auto-generated insights.");
        }
      } catch (err) {
        setMessage(els.scanStatus, err.message, "error");
        showToast(err.message, "error");
      } finally {
        setButtonLoading(els.scanButton, false);
        setTimeout(hideProgress, 1500);
      }
    });
  }

  // ---- Query ----
  async function runQuery() {
    const query = els.queryInput ? els.queryInput.value.trim() : "";
    if (!query) return;
    state.query = query;
    setButtonLoading(els.queryButton, true);
    setMessage(els.queryStatus, "");

    try {
      const data = await fetchJson("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });

      setMessage(els.queryStatus, `${data.matches.length} match(es) for "${data.query}".`, data.matches.length ? "success" : "");
      setVerification(data.verification_label, data.warning);
      renderMatches(data.matches);

      if (data.matches.length) {
        showToast(`Found ${data.matches.length} match(es)`, "success");
      }
    } catch (err) {
      setMessage(els.queryStatus, err.message, "error");
      showToast(err.message, "error");
    } finally {
      setButtonLoading(els.queryButton, false);
    }
  }

  // ---- Export ----
  async function runExport() {
    if (!els.resultsList) return;
    const selected = [...els.resultsList.querySelectorAll("input[type='checkbox']:checked")]
      .map((input) => input.value);

    if (!selected.length) {
      showToast("Select at least one result to export", "info");
      return;
    }

    setButtonLoading(els.exportButton, true);
    setMessage(els.exportStatus, "");
    if (els.downloadList) els.downloadList.innerHTML = "";

    try {
      const data = await fetchJson("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ selected, query: state.query, segment_timeout: 8 }),
      });

      const warning = (data.warnings || [])[0] || "";
      setMessage(els.exportStatus, warning ? `${data.message} ${warning}` : data.message, warning ? "warn" : "success");

      if (els.downloadList) {
        for (const [kind, file] of Object.entries(data.files || {})) {
          const link = document.createElement("a");
          link.href = file.url;
          link.textContent = `${kind}: ${file.name}`;
          els.downloadList.appendChild(link);
        }
      }

      showToast("Export complete — files ready", "success");
    } catch (err) {
      setMessage(els.exportStatus, err.message, "error");
      showToast(err.message, "error");
    } finally {
      setButtonLoading(els.exportButton, false);
    }
  }

  // ---- Zero-Query ----
  async function runZeroQuery() {
    setButtonLoading(els.zeroQueryButton, true);
    setMessage(els.zeroQuerySummary, "");
    if (els.zeroQueryInventory) els.zeroQueryInventory.innerHTML = "";
    if (els.zeroQueryTimeline) els.zeroQueryTimeline.innerHTML = "";

    try {
      const data = await fetchJson("/api/zero_query");
      setMessage(els.zeroQuerySummary, data.summary || "Analysis complete.", "success");

      // Object inventory
      const inv = data.object_inventory || {};
      const classes = Object.keys(inv);
      if (classes.length && els.zeroQueryInventory) {
        let html = `<h3>Object Inventory</h3><table class="zq-table"><thead><tr><th>Class</th><th>Count</th><th>Total Dwell</th><th>Avg Dwell</th></tr></thead><tbody>`;
        for (const cls of classes) {
          const info = inv[cls];
          html += `<tr><td>${cls}</td><td>${info.count}</td><td>${info.total_dwell_time}s</td><td>${info.avg_dwell_time}s</td></tr>`;
        }
        html += "</tbody></table>";
        els.zeroQueryInventory.innerHTML = html;
      }

      // Event timeline
      const events = data.event_timeline || [];
      if (events.length && els.zeroQueryTimeline) {
        let html = `<h3>Event Timeline</h3><ul class="zq-events">`;
        for (const ev of events) {
          const ts = ev.start != null
            ? `${ev.start.toFixed(1)}s – ${ev.end.toFixed(1)}s`
            : `${(ev.entry_ts || 0).toFixed(1)}s`;
          const typeClass = `ev-type--${ev.type}`;
          let detail = `@ ${ts}`;
          if (ev.objects) detail += ` · ${ev.objects.join(", ")}`;
          if (ev.class) detail += ` · ${ev.class}`;
          if (ev.dwell_time) detail += ` · dwell: ${ev.dwell_time}s`;
          html += `<li><span class="ev-type ${typeClass}">${ev.type.replace(/_/g, " ")}</span> ${detail}</li>`;
        }
        html += "</ul>";
        els.zeroQueryTimeline.innerHTML = html;
      } else if (els.zeroQueryTimeline) {
        els.zeroQueryTimeline.innerHTML = `<p style="color:var(--text-muted);font-size:13px;margin-top:12px">No notable events detected.</p>`;
      }

      showToast("Analysis generated", "success");
    } catch (err) {
      setMessage(els.zeroQuerySummary, err.message, "error");
      showToast(err.message, "error");
    } finally {
      setButtonLoading(els.zeroQueryButton, false);
    }
  }

  // ---- Event bindings ----
  if (els.queryButton) {
    els.queryButton.addEventListener("click", runQuery);
  }

  if (els.queryInput) {
    els.queryInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter") { e.preventDefault(); runQuery(); }
    });
  }

  if (els.exportButton) {
    els.exportButton.addEventListener("click", runExport);
  }

  if (els.zeroQueryButton) {
    els.zeroQueryButton.addEventListener("click", runZeroQuery);
  }

  // File upload zone interactivity
  if (els.fileUploadZone && els.videoUpload) {
    els.fileUploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      els.fileUploadZone.classList.add("dragover");
    });
    els.fileUploadZone.addEventListener("dragleave", () => {
      els.fileUploadZone.classList.remove("dragover");
    });
    els.fileUploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      els.fileUploadZone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        els.videoUpload.files = e.dataTransfer.files;
        showVideoPreview(URL.createObjectURL(e.dataTransfer.files[0]), e.dataTransfer.files[0].name);
        const label = els.fileUploadZone.querySelector(".file-upload__label span");
        if (label) label.textContent = e.dataTransfer.files[0].name;
      }
    });
    els.videoUpload.addEventListener("change", () => {
      const label = els.fileUploadZone.querySelector(".file-upload__label span");
      if (label && els.videoUpload.files.length) {
        label.textContent = els.videoUpload.files[0].name;
        showVideoPreview(URL.createObjectURL(els.videoUpload.files[0]), els.videoUpload.files[0].name);
      }
    });
  }

  if (els.sampleSelect) els.sampleSelect.addEventListener("change", updateSamplePreview);

  // ---- Global exports for inline onclick compatibility ----
  window.visionGuardFind = runQuery;
  window.visionGuardExport = runExport;
  window.visionGuardZeroQuery = runZeroQuery;

  // ---- Init ----
  loadAssets();
  loadStatus();

})();
