(() => {
  "use strict";

  const ACTIVE_SESSION_KEY = "visionguard-active-video-job";

  const state = {
    videoId: null,
    jobId: null,
    query: "",
    matches: [],
    frames: [],
    lastEventId: 0,
    searchable: false,
    detectorReady: false,
    semanticReady: false,
    previewBlobUrl: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const els = {
    statusText: $("#systemStatus .status-text"),
    sampleSelect: $("#sampleSelect"),
    videoPreviewWrap: $("#videoPreviewWrap"),
    videoPreview: $("#videoPreview"),
    videoPreviewName: $("#videoPreviewName"),
    videoUpload: $("#videoUpload"),
    scanForm: $("#scanForm"),
    scanButton: $("#scanButton"),
    indexButton: $("#indexButton"),
    scanStatus: $("#scanStatus"),
    scanProgress: $("#scanProgress"),
    scanProgressFill: $("#scanProgressFill"),
    scanProgressLabel: $("#scanProgressLabel"),
    fileUploadZone: $("#fileUploadZone"),
    processingTimeline: $("#processingTimeline"),
    backendConsole: $("#backendConsole"),
    evidenceStrip: $("#evidenceStrip"),
    evidenceCount: $("#evidenceCount"),
    frameInspector: $("#frameInspector"),
    inspectorImage: $("#inspectorImage"),
    inspectorTimestamp: $("#inspectorTimestamp"),
    inspectorDetails: $("#inspectorDetails"),
    queryInput: $("#queryInput"),
    responseMode: $("#responseMode"),
    queryButton: $("#queryButton"),
    queryStatus: $("#queryStatus"),
    verificationStatus: $("#verificationStatus"),
    resultsList: $("#resultsList"),
    resultsCount: $("#resultsCount"),
    exportButton: $("#exportButton"),
    exportStatus: $("#exportStatus"),
    downloadList: $("#downloadList"),
    zeroQueryPanel: $("#zeroQueryPanel"),
    zeroQueryButton: $("#zeroQueryButton"),
    zeroQuerySummary: $("#zeroQuerySummary"),
    zeroQueryInventory: $("#zeroQueryInventory"),
    zeroQueryTimeline: $("#zeroQueryTimeline"),
    toastContainer: $("#toastContainer"),
    providerName: $("#providerName"),
    textModelStatus: $("#textModelStatus"),
    visionModelStatus: $("#visionModelStatus"),
    nvidiaStatus: $("#nvidiaStatus"),
    groqStatus: $("#groqStatus"),
    modelNotice: $("#modelNotice"),
  };

  async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) throw new Error(data.message || `Request failed with ${response.status}`);
    return data;
  }

  function persistActiveSession(filename, sourceUrl) {
    try {
      window.sessionStorage.setItem(ACTIVE_SESSION_KEY, JSON.stringify({
        videoId: state.videoId,
        jobId: state.jobId,
        filename,
        sourceUrl,
      }));
    } catch (error) {
      console.warn("Active job state could not be persisted for reload recovery.", error);
    }
  }

  function clearActiveSession() {
    try {
      window.sessionStorage.removeItem(ACTIVE_SESSION_KEY);
    } catch (error) {
      console.warn("Active job state could not be cleared.", error);
    }
  }

  function waitForPoll(milliseconds) {
    return new Promise((resolve) => {
      const timer = window.setInterval(() => {
        window.clearInterval(timer);
        resolve();
      }, milliseconds);
    });
  }

  function setMessage(element, text, kind = "") {
    if (!element) return;
    element.textContent = text || "";
    element.className = kind ? `status-msg ${kind}` : "status-msg";
  }

  function setButtonLoading(button, loading) {
    if (!button) return;
    button.disabled = loading;
    button.classList.toggle("loading", loading);
  }

  function setSearchReady(ready) {
    state.searchable = ready;
    if (els.queryInput) {
      els.queryInput.disabled = !ready;
      els.queryInput.placeholder = ready ? "Describe the visible object or event to find" : "Process a video before searching";
    }
    if (els.queryButton) els.queryButton.disabled = !ready;
    if (els.responseMode) els.responseMode.disabled = !ready;
  }

  function setVerification(label, warning = "") {
    if (!els.verificationStatus) return;
    els.verificationStatus.textContent = warning ? `${label}. ${warning}` : label;
    els.verificationStatus.className = warning ? "verification-chip warn" : "verification-chip";
  }

  function setCapability(element, label, stateName) {
    if (!element) return;
    element.textContent = label;
    element.className = `capability-state ${stateName}`;
  }

  function providerLabel(name) {
    if (name === "llama_cpp") return "llama.cpp";
    if (name === "none") return "None";
    return name ? name.charAt(0).toUpperCase() + name.slice(1) : "Unknown";
  }

  async function loadModelHealth() {
    try {
      const data = await fetchJson("/api/model/health");
      if (els.providerName) els.providerName.textContent = providerLabel(data.selected_provider);
      const textModel = data.text_model || {};
      const visionModel = data.vision_model || {};
      const detector = data.detector || {};
      const semantic = data.semantic || {};
      state.detectorReady = detector.ready !== false;
      state.semanticReady = semantic.ready !== false;
      setCapability(
        els.textModelStatus,
        textModel.reachable ? "Connected" : (textModel.configured ? "Disconnected" : "Disabled"),
        textModel.reachable ? "connected" : (textModel.configured ? "disconnected" : "disabled"),
      );
      setCapability(
        els.visionModelStatus,
        visionModel.reachable ? "Connected" : (visionModel.configured ? "Disconnected" : "Not configured"),
        visionModel.reachable ? "connected" : (visionModel.configured ? "disconnected" : "disabled"),
      );
      if (els.nvidiaStatus) els.nvidiaStatus.textContent = data.external_providers?.nvidia || "disabled";
      if (els.groqStatus) els.groqStatus.textContent = data.external_providers?.groq || "disabled";
      if (els.modelNotice) {
        els.modelNotice.textContent = !state.detectorReady
          ? (detector.message || "The local YOLO detector is unavailable. Run scripts\\bootstrap_models.py before indexing.")
          : !state.semanticReady
            ? (semantic.message || "NVIDIA semantic indexing is unavailable. Configure NVIDIA_API_KEY before indexing.")
          : textModel.reachable
          ? `${providerLabel(data.selected_provider)} text reasoning is connected. Evidence rules remain enforced by the backend.`
          : `${textModel.message || "The selected text model is unavailable"} Video upload, frame extraction, and detector-backed object search remain available.`;
      }
    } catch (error) {
      setCapability(els.textModelStatus, "Health check failed", "disconnected");
      setCapability(els.visionModelStatus, "Unknown", "disabled");
      if (els.modelNotice) els.modelNotice.textContent = `Model health could not be loaded: ${error.message}`;
    }
  }

  function showToast(message, type = "info") {
    if (!els.toastContainer) return;
    const toast = document.createElement("div");
    toast.className = `toast toast--${type}`;
    toast.textContent = message;
    els.toastContainer.appendChild(toast);
    const timer = window.setInterval(() => {
      window.clearInterval(timer);
      toast.remove();
    }, 4000);
  }

  function showProgress(percent, label) {
    if (!els.scanProgress) return;
    els.scanProgress.style.display = "";
    if (els.scanProgressFill) els.scanProgressFill.style.width = `${Math.max(0, Math.min(100, percent))}%`;
    if (els.scanProgressLabel) els.scanProgressLabel.textContent = label;
  }

  function stageProgress(stages) {
    if (!stages.length) return 0;
    const value = stages.reduce((sum, stage) => {
      if (["completed", "skipped"].includes(stage.status)) return sum + 1;
      if (stage.status === "failed") return sum + 1;
      if (stage.status === "running" && stage.total > 0) return sum + Math.min(0.95, stage.processed / stage.total);
      return sum;
    }, 0);
    return Math.round((value / stages.length) * 100);
  }

  function renderStages(stages) {
    if (!els.processingTimeline) return;
    els.processingTimeline.innerHTML = "";
    for (const stage of stages) {
      const row = document.createElement("li");
      row.className = "stage-row";
      const marker = document.createElement("span");
      marker.className = `stage-state ${stage.status}`;
      const content = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = stage.label;
      const detail = document.createElement("small");
      const counts = stage.total > 0 ? ` ${stage.processed}/${stage.total}` : "";
      detail.textContent = `${stage.status}${counts}${stage.message ? ` — ${stage.message}` : ""}`;
      content.append(title, detail);
      row.append(marker, content);
      els.processingTimeline.appendChild(row);
    }
  }

  async function loadJobEvents() {
    if (!state.jobId) return;
    const data = await fetchJson(`/api/jobs/${state.jobId}/events?after=${state.lastEventId}`);
    if (!data.events.length) return;
    if (state.lastEventId === 0 && els.backendConsole) els.backendConsole.innerHTML = "";
    for (const event of data.events) {
      state.lastEventId = Math.max(state.lastEventId, event.event_id);
      const row = document.createElement("li");
      row.textContent = `${event.timestamp} · ${event.stage} · ${event.status}${event.message ? ` · ${event.message}` : ""}`;
      els.backendConsole?.appendChild(row);
    }
    if (els.backendConsole) els.backendConsole.scrollTop = els.backendConsole.scrollHeight;
  }

  function seekVideo(frame) {
    if (!els.videoPreview || !frame) return;
    els.videoPreview.currentTime = frame.timestamp_ms / 1000;
    els.videoPreview.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function inspectFrame(frame, seek = false) {
    if (!frame || !els.frameInspector) return;
    els.frameInspector.style.display = "grid";
    els.inspectorImage.src = frame.image_url;
    els.inspectorTimestamp.textContent = `${(frame.timestamp_ms / 1000).toFixed(3)}s · frame ${frame.frame_number}`;
    const objects = frame.objects?.length ? frame.objects.join(", ") : "no detector objects";
    els.inspectorDetails.textContent = `${objects} · ${frame.detections?.length || 0} detections · ${frame.selection_reason || "selected frame"}`;
    if (seek) seekVideo(frame);
  }

  function renderFrames(frames) {
    if (!els.evidenceStrip) return;
    const existingIds = new Set(state.frames.map((frame) => frame.frame_id));
    const appendOnly = state.frames.length > 0 && frames.length >= state.frames.length && state.frames.every((frame, index) => frames[index]?.frame_id === frame.frame_id);
    state.frames = frames;
    if (appendOnly) {
      for (const frame of frames) {
        if (!existingIds.has(frame.frame_id)) appendEvidenceThumbnail(frame);
      }
      if (els.evidenceCount) {
        els.evidenceCount.textContent = String(frames.length);
        els.evidenceCount.style.display = "";
      }
      return;
    }
    els.evidenceStrip.innerHTML = "";
    if (!frames.length) {
      const empty = document.createElement("p");
      empty.className = "evidence-empty";
      empty.textContent = "No evidence frame image has been committed yet.";
      els.evidenceStrip.appendChild(empty);
      if (els.evidenceCount) els.evidenceCount.style.display = "none";
      return;
    }
    for (const frame of frames) appendEvidenceThumbnail(frame);
    if (els.evidenceCount) {
      els.evidenceCount.textContent = String(frames.length);
      els.evidenceCount.style.display = "";
    }
  }

  function appendEvidenceThumbnail(frame) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "evidence-thumb";
    button.title = `Seek to ${(frame.timestamp_ms / 1000).toFixed(3)} seconds`;
    const image = document.createElement("img");
    image.src = frame.image_url;
    image.alt = `Evidence frame ${frame.frame_number}`;
    image.loading = "lazy";
    const label = document.createElement("span");
    label.textContent = `${(frame.timestamp_ms / 1000).toFixed(2)}s`;
    button.append(image, label);
    button.addEventListener("click", () => inspectFrame(frame, true));
    els.evidenceStrip.appendChild(button);
  }

  async function refreshFrames() {
    if (!state.videoId) return;
    const data = await fetchJson(`/api/videos/${state.videoId}/frames`);
    if (data.frames.length !== state.frames.length) renderFrames(data.frames);
  }

  function showVideoPreview(url, name) {
    if (state.previewBlobUrl && state.previewBlobUrl !== url) {
      URL.revokeObjectURL(state.previewBlobUrl);
      state.previewBlobUrl = null;
    }
    if (!url || !els.videoPreview) return;
    els.videoPreview.src = url;
    els.videoPreviewWrap.style.display = "";
    els.videoPreviewName.textContent = name;
  }

  function updateSamplePreview() {
    if (els.videoUpload?.files.length) return;
    const option = els.sampleSelect?.selectedOptions?.[0];
    if (option) showVideoPreview(option.dataset.url, option.value);
  }

  async function loadAssets() {
    const data = await fetchJson("/api/assets");
    els.sampleSelect.innerHTML = "";
    for (const asset of data.assets) {
      const option = document.createElement("option");
      option.value = asset.name;
      option.dataset.url = asset.url;
      option.textContent = `${asset.name} (${Math.round(asset.size / 1024)} KB)`;
      els.sampleSelect.appendChild(option);
    }
    updateSamplePreview();
  }

  async function loadStatus() {
    try {
      const data = await fetchJson("/api/status");
      els.statusText.textContent = data.status;
      $("#systemStatus")?.classList.remove("error");
      setVerification(data.verification_label, data.warning);
    } catch (error) {
      els.statusText.textContent = error.message;
      $("#systemStatus")?.classList.add("error");
    }
  }

  async function pollProcessing() {
    let consecutiveStatusFailures = 0;
    while (true) {
      let status;
      try {
        status = await fetchJson(`/api/videos/${state.videoId}/status`);
        consecutiveStatusFailures = 0;
      } catch (error) {
        consecutiveStatusFailures += 1;
        setMessage(els.scanStatus, `Processing status temporarily unavailable (${consecutiveStatusFailures}/5): ${error.message}`, "scanning");
        if (consecutiveStatusFailures >= 5) throw error;
        await waitForPoll(1000);
        continue;
      }
      renderStages(status.stages);
      try {
        await loadJobEvents();
      } catch (error) {
        console.warn("Backend event refresh failed; authoritative status polling will continue.", error);
      }
      try {
        await refreshFrames();
      } catch (error) {
        console.warn("Evidence thumbnail refresh failed; authoritative status polling will continue.", error);
      }
      const percent = stageProgress(status.stages);
      const running = status.stages.find((stage) => stage.status === "running");
      showProgress(percent, running ? `${running.label}: ${running.processed}/${running.total || "?"}` : status.status);
      if (status.status === "completed") {
        setSearchReady(true);
        setMessage(els.scanStatus, "Processing completed. Evidence index is query-ready.", "success");
        if (els.zeroQueryPanel) els.zeroQueryPanel.style.display = "";
        showToast("Video is searchable", "success");
        return;
      }
      if (status.status === "failed") throw new Error(status.error || "Video processing failed.");
      await waitForPoll(500);
    }
  }

  async function submitVideo(event) {
    event.preventDefault();
    clearActiveSession();
    setSearchReady(false);
    setButtonLoading(els.scanButton, true);
    setMessage(els.scanStatus, "Uploading video to the backend.", "scanning");
    showProgress(0, "Waiting for upload response");
    renderFrames([]);
    renderMatches([]);
    state.lastEventId = 0;
    state.videoId = null;
    state.jobId = null;
    if (els.backendConsole) els.backendConsole.innerHTML = "<li>Waiting for backend acknowledgement.</li>";
    try {
      let options;
      if (els.videoUpload?.files.length) {
        const form = new FormData();
        form.append("video", els.videoUpload.files[0]);
        options = { method: "POST", body: form };
      } else {
        options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sample: els.sampleSelect?.value || "" }) };
      }
      const upload = await fetchJson("/api/videos/upload", options);
      state.videoId = upload.video_id;
      state.jobId = upload.job_id;
      showVideoPreview(upload.source_url, upload.filename);
      persistActiveSession(upload.filename, upload.source_url);
      if (els.indexButton) els.indexButton.disabled = !state.detectorReady || !state.semanticReady;
      setMessage(
        els.scanStatus,
        state.detectorReady && state.semanticReady
          ? "Upload complete. Review the video, then start evidence indexing."
          : "Upload complete, but indexing is unavailable until required detector and NVIDIA semantic services are ready.",
        state.detectorReady && state.semanticReady ? "success" : "error",
      );
      showProgress(0, "Upload complete — ready to index");
    } catch (error) {
      setMessage(els.scanStatus, error.message, "error");
      showToast(error.message, "error");
    } finally {
      setButtonLoading(els.scanButton, false);
    }
  }

  async function restoreActiveSession() {
    let saved;
    try {
      saved = JSON.parse(window.sessionStorage.getItem(ACTIVE_SESSION_KEY) || "null");
    } catch (error) {
      clearActiveSession();
      return;
    }
    if (!saved?.videoId || !saved?.jobId) return;

    state.videoId = saved.videoId;
    state.jobId = saved.jobId;
    state.lastEventId = 0;
    try {
      const video = await fetchJson(`/api/videos/${state.videoId}`);
      showVideoPreview(video.source_url || saved.sourceUrl, video.filename || saved.filename);
      const status = await fetchJson(`/api/videos/${state.videoId}/status`);
      renderStages(status.stages);
      await loadJobEvents().catch((error) => console.warn("Backend event recovery failed.", error));
      await refreshFrames().catch((error) => console.warn("Evidence thumbnail recovery failed.", error));
      if (status.status === "completed") {
        setSearchReady(true);
        setMessage(els.scanStatus, "Processing completed. Evidence index is query-ready.", "success");
        if (els.zeroQueryPanel) els.zeroQueryPanel.style.display = "";
        return;
      }
      if (status.status === "failed") {
        setSearchReady(false);
        setMessage(els.scanStatus, status.error || "Video processing failed.", "error");
        return;
      }
      setSearchReady(false);
      setMessage(els.scanStatus, "Restored the active indexing job after page reload.", "scanning");
      await pollProcessing();
    } catch (error) {
      clearActiveSession();
      state.videoId = null;
      state.jobId = null;
      setSearchReady(false);
      setMessage(els.scanStatus, `The previous job could not be restored: ${error.message}`, "error");
    }
  }

  async function startIndexing() {
    if (!state.videoId) return;
    setSearchReady(false);
    setButtonLoading(els.indexButton, true);
    setMessage(els.scanStatus, "Starting evidence indexing.", "scanning");
    try {
      await fetchJson(`/api/videos/${state.videoId}/index`, { method: "POST" });
      await pollProcessing();
      await loadStatus();
    } catch (error) {
      setMessage(els.scanStatus, error.message, "error");
      showToast(error.message, "error");
    } finally {
      setButtonLoading(els.indexButton, false);
    }
  }

  function renderMatches(matches) {
    state.matches = matches;
    els.resultsList.innerHTML = "";
    if (!matches.length) {
      els.resultsList.className = "results-list results-list--empty";
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No evidence-backed result is available.";
      els.resultsList.appendChild(empty);
      els.resultsCount.style.display = "none";
      return;
    }
    els.resultsList.className = "results-list";
    els.resultsCount.textContent = String(matches.length);
    els.resultsCount.style.display = "";
    for (const match of matches) {
      const card = document.createElement("article");
      card.className = "match-card match-card--evidence";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = match.id;
      checkbox.setAttribute("aria-label", `Select ${match.label}`);
      const image = document.createElement("img");
      image.className = "match-card__image";
      image.src = match.frame.image_url;
      image.alt = `Evidence at ${(match.frame.timestamp_ms / 1000).toFixed(2)} seconds`;
      const content = document.createElement("button");
      content.type = "button";
      content.className = "match-card__content";
      const title = document.createElement("strong");
      title.textContent = `${(match.frame.timestamp_ms / 1000).toFixed(3)}s · frame ${match.frame.frame_number}`;
      const summary = document.createElement("span");
      summary.textContent = match.summary;
      const mode = document.createElement("small");
      const provenance = match.claim_provenance || "unknown provenance";
      const evidenceState = (match.evidence_state || "unknown").replaceAll("_", " ");
      mode.textContent = `${evidenceState} · ${provenance} · ranking score ${match.score}`;
      content.append(title, summary, mode);
      content.addEventListener("click", () => inspectFrame(match.frame, true));
      card.append(checkbox, image, content);
      els.resultsList.appendChild(card);
    }
  }

  async function runQuery() {
    const query = els.queryInput?.value.trim() || "";
    if (!query || !state.videoId || !state.searchable) return;
    state.query = query;
    setButtonLoading(els.queryButton, true);
    try {
      const response_mode = els.responseMode?.value || "both";
      const data = await fetchJson(`/api/videos/${state.videoId}/query`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, response_mode }) });
      renderMatches(data.matches);
      const displayText = data.response_mode === "frames" && !data.insufficient_evidence ? data.message : data.answer;
      setMessage(els.queryStatus, displayText, data.insufficient_evidence ? "warn" : "success");
      setVerification(data.verification_label, data.warning);
      if (!data.insufficient_evidence && data.frames.length) inspectFrame(data.frames[0], false);
    } catch (error) {
      setMessage(els.queryStatus, error.message, "error");
      showToast(error.message, "error");
    } finally {
      setButtonLoading(els.queryButton, false);
      if (state.searchable) els.queryButton.disabled = false;
    }
  }

  async function runZeroQuery() {
    setButtonLoading(els.zeroQueryButton, true);
    try {
      const data = await fetchJson("/api/zero_query");
      setMessage(els.zeroQuerySummary, data.summary || "Stored detector analysis loaded.", "success");
      els.zeroQueryInventory.innerHTML = "";
      const inventory = document.createElement("ul");
      for (const [name, info] of Object.entries(data.object_inventory || {})) {
        const row = document.createElement("li");
        row.textContent = `${name}: ${info.count} track(s), ${info.total_dwell_time}s total dwell`;
        inventory.appendChild(row);
      }
      els.zeroQueryInventory.appendChild(inventory);
      els.zeroQueryTimeline.innerHTML = "";
      const timeline = document.createElement("ul");
      for (const item of data.event_timeline || []) {
        const row = document.createElement("li");
        row.textContent = `${item.type.replaceAll("_", " ")} at ${Number(item.start ?? item.entry_ts ?? 0).toFixed(2)}s`;
        timeline.appendChild(row);
      }
      els.zeroQueryTimeline.appendChild(timeline);
    } catch (error) {
      setMessage(els.zeroQuerySummary, error.message, "error");
    } finally {
      setButtonLoading(els.zeroQueryButton, false);
    }
  }

  async function runExport() {
    const selected = [...els.resultsList.querySelectorAll("input[type='checkbox']:checked")].map((input) => input.value);
    if (!selected.length) return showToast("Select at least one evidence result", "info");
    setButtonLoading(els.exportButton, true);
    try {
      const data = await fetchJson("/api/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ selected, query: state.query, segment_timeout: 8 }) });
      setMessage(els.exportStatus, data.message, data.warnings?.length ? "warn" : "success");
      els.downloadList.innerHTML = "";
      for (const [kind, file] of Object.entries(data.files || {})) {
        const link = document.createElement("a");
        link.href = file.url;
        link.textContent = `${kind}: ${file.name}`;
        els.downloadList.appendChild(link);
      }
    } catch (error) {
      setMessage(els.exportStatus, error.message, "error");
    } finally {
      setButtonLoading(els.exportButton, false);
    }
  }

  els.scanForm?.addEventListener("submit", submitVideo);
  els.indexButton?.addEventListener("click", startIndexing);
  els.queryButton?.addEventListener("click", runQuery);
  els.queryInput?.addEventListener("keydown", (event) => { if (event.key === "Enter") { event.preventDefault(); runQuery(); } });
  els.exportButton?.addEventListener("click", runExport);
  els.zeroQueryButton?.addEventListener("click", runZeroQuery);
  els.sampleSelect?.addEventListener("change", updateSamplePreview);
  els.videoUpload?.addEventListener("change", () => {
    if (!els.videoUpload.files.length) return;
    const file = els.videoUpload.files[0];
    els.fileUploadZone.querySelector("span").textContent = file.name;
    state.previewBlobUrl = URL.createObjectURL(file);
    showVideoPreview(state.previewBlobUrl, file.name);
  });
  els.fileUploadZone?.addEventListener("dragover", (event) => { event.preventDefault(); els.fileUploadZone.classList.add("dragover"); });
  els.fileUploadZone?.addEventListener("dragleave", () => els.fileUploadZone.classList.remove("dragover"));
  els.fileUploadZone?.addEventListener("drop", (event) => {
    event.preventDefault();
    els.fileUploadZone.classList.remove("dragover");
    if (!event.dataTransfer.files.length) return;
    els.videoUpload.files = event.dataTransfer.files;
    els.videoUpload.dispatchEvent(new Event("change"));
  });

  async function initialize() {
    setSearchReady(false);
    await Promise.allSettled([loadAssets(), loadStatus(), loadModelHealth()]);
    await restoreActiveSession();
  }

  initialize();
})();
