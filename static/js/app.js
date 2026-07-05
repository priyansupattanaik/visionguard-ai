const state = {
  query: "",
  matches: [],
};

const els = {
  systemStatus: document.querySelector("#systemStatus"),
  sampleSelect: document.querySelector("#sampleSelect"),
  videoUpload: document.querySelector("#videoUpload"),
  scanForm: document.querySelector("#scanForm"),
  scanButton: document.querySelector("#scanButton"),
  scanStatus: document.querySelector("#scanStatus"),
  queryInput: document.querySelector("#queryInput"),
  queryButton: document.querySelector("#queryButton"),
  queryStatus: document.querySelector("#queryStatus"),
  verificationStatus: document.querySelector("#verificationStatus"),
  resultsList: document.querySelector("#resultsList"),
  exportButton: document.querySelector("#exportButton"),
  exportStatus: document.querySelector("#exportStatus"),
  downloadList: document.querySelector("#downloadList"),
};

function setMessage(el, text, kind = "") {
  el.textContent = text || "";
  el.className = `message ${kind}`.trim();
}

function setVerification(label, warning) {
  els.verificationStatus.textContent = warning ? `${label}. ${warning}` : label;
  els.verificationStatus.className = warning ? "verification warn" : "verification";
}

async function fetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(data.message || `Request failed with ${res.status}`);
  }
  return data;
}

async function loadStatus() {
  try {
    const data = await fetchJson("/api/status");
    els.systemStatus.textContent = data.status;
    setVerification(data.verification_label, data.warning);
  } catch (err) {
    els.systemStatus.textContent = err.message;
  }
}

async function loadAssets() {
  const data = await fetchJson("/api/assets");
  els.sampleSelect.innerHTML = "";
  for (const asset of data.assets) {
    const option = document.createElement("option");
    option.value = asset.name;
    option.textContent = `${asset.name} (${Math.round(asset.size / 1024)} KB)`;
    els.sampleSelect.appendChild(option);
  }
}

function renderMatches(matches) {
  state.matches = matches;
  els.downloadList.innerHTML = "";
  if (!matches.length) {
    els.resultsList.className = "results-list empty";
    els.resultsList.textContent = "No strong matches found.";
    return;
  }
  els.resultsList.className = "results-list";
  els.resultsList.innerHTML = "";
  for (const match of matches) {
    const card = document.createElement("article");
    card.className = "match-card";
    card.innerHTML = `
      <input type="checkbox" value="${match.id}">
      <div>
        <strong>${match.label}</strong>
        <div class="match-meta">${match.start}s - ${match.end}s | best frame ${match.peak_ts}s | score ${match.score}</div>
        <div>${match.summary}</div>
        <span class="mode">${match.verification_label}</span>
      </div>
    `;
    els.resultsList.appendChild(card);
  }
}

els.scanForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  els.scanButton.disabled = true;
  setMessage(els.scanStatus, "Scan started...");
  renderMatches([]);
  try {
    let options;
    if (els.videoUpload.files.length) {
      const form = new FormData();
      form.append("video", els.videoUpload.files[0]);
      options = { method: "POST", body: form };
    } else {
      options = {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample: els.sampleSelect.value }),
      };
    }
    const data = await fetchJson("/api/scan", options);
    const counts = Object.entries(data.object_counts || {})
      .map(([name, count]) => `${name}: ${count}`)
      .join(", ");
    setMessage(els.scanStatus, `${data.message} ${data.video}: ${data.indexed_windows} windows. ${counts}`);
    setVerification(data.backend.verification_label, (data.warnings || [])[0] || "");
    await loadStatus();
  } catch (err) {
    setMessage(els.scanStatus, err.message, "error");
  } finally {
    els.scanButton.disabled = false;
  }
});

els.queryButton.addEventListener("click", async () => {
  const query = els.queryInput.value.trim();
  state.query = query;
  setMessage(els.queryStatus, "Searching...");
  try {
    const data = await fetchJson("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    setMessage(els.queryStatus, `${data.matches.length} match(es) for "${data.query}".`);
    setVerification(data.verification_label, data.warning);
    renderMatches(data.matches);
  } catch (err) {
    setMessage(els.queryStatus, err.message, "error");
  }
});

els.exportButton.addEventListener("click", async () => {
  const selected = [...els.resultsList.querySelectorAll("input[type='checkbox']:checked")]
    .map((input) => input.value);
  setMessage(els.exportStatus, "Export started...");
  els.downloadList.innerHTML = "";
  try {
    const data = await fetchJson("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected, query: state.query, segment_timeout: 8 }),
    });
    const warning = (data.warnings || [])[0] || "";
    setMessage(els.exportStatus, warning ? `${data.message} ${warning}` : data.message, warning ? "warn" : "");
    for (const [kind, file] of Object.entries(data.files || {})) {
      const link = document.createElement("a");
      link.href = file.url;
      link.textContent = `${kind}: ${file.name}`;
      els.downloadList.appendChild(link);
    }
  } catch (err) {
    setMessage(els.exportStatus, err.message, "error");
  }
});

loadAssets().catch((err) => setMessage(els.scanStatus, err.message, "error"));
loadStatus();
