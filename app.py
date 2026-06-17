import base64
import os
import threading
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

setup_cache()

APP_TITLE = "Vision Guard"
ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "output" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pipe = VisionGuardPipeline()
threading.Thread(target=pipe.warmup_models, daemon=True).start()

app = FastAPI(title=APP_TITLE)
state_lock = threading.Lock()
app_state = {
    "scan_running": False,
    "scan_status": "ready",
    "preview_b64": None,
    "meta": None,
    "video": None,
    "hits": [],
    "query": "",
    "artifacts": {},
}


def _set_state(**kwargs):
    with state_lock:
        app_state.update(kwargs)


def _get_state():
    with state_lock:
        return dict(app_state)


def _sample_videos():
    assets = ROOT / "assets"
    if not assets.exists():
        return []
    return sorted(p.name for p in assets.glob("*.mp4"))


def _in_colab():
    return bool(
        os.getenv("COLAB_RELEASE_TAG")
        or os.getenv("COLAB_BACKEND_VERSION")
        or os.getenv("COLAB_GPU")
        or os.getenv("JPY_PARENT_PID") and str(ROOT).startswith("/content/")
    )


def _runtime_host():
    override = os.getenv("VISION_GUARD_HOST", "").strip()
    if override:
        return override
    if _in_colab() or os.getenv("KAGGLE_KERNEL_RUN_TYPE"):
        return "0.0.0.0"
    return "127.0.0.1"


def _encode_preview(image):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _meta_text(meta):
    if not meta:
        return {}
    return {
        "video": os.path.basename(meta["video"]),
        "duration": round(float(meta["duration"]), 2),
        "fps": round(float(meta["fps"]), 2),
        "sample_sec": round(float(meta["sample_sec"]), 2),
        "segments": int(meta["segments"]),
        "retriever": meta.get("retriever", "numpy"),
        "verifier": meta.get("verifier", "none"),
    }


def _safe_path(path_str):
    if not path_str:
        raise HTTPException(status_code=404, detail="missing path")
    path = Path(path_str).resolve()
    allowed = [ROOT / "output", ROOT / "assets"]
    for base in allowed:
        try:
            path.relative_to(base.resolve())
            if path.exists():
                return path
        except Exception:
            continue
    raise HTTPException(status_code=404, detail="file not found")


def _image_url(path_str):
    return f"/artifact?path={quote(path_str)}"


def _rows_for_ui(rows):
    out = []
    for row in rows:
        frame_path = row.get("gallery_frame") or row.get("representative_frame_path") or row.get("frame_path")
        out.append({
            "label": row["label"],
            "summary": row["summary"],
            "objects": row.get("objects", []),
            "start": round(float(row["start"]), 2),
            "end": round(float(row["end"]), 2),
            "peak_ts": round(float(row.get("peak_ts", row["start"])), 2),
            "score": round(float(row.get("score", 0.0)), 4),
            "low_confidence": bool(row.get("low_confidence")),
            "image_url": _image_url(frame_path) if frame_path else None,
        })
    return out


def _scan_worker(video_path):
    _set_state(scan_running=True, scan_status="starting scan", preview_b64=None, meta=None, video=video_path, hits=[], query="", artifacts={})
    meta = None
    try:
        for ev in pipe.index_video_iter(video_path):
            if ev["kind"] == "preview":
                _set_state(scan_status=ev["status"], preview_b64=_encode_preview(ev["image"]))
            elif ev["kind"] == "done":
                meta = ev["meta"]
        _set_state(scan_running=False, scan_status="scan complete", meta=_meta_text(meta))
    except Exception as exc:
        _set_state(scan_running=False, scan_status=f"scan failed: {exc}")


HTML = """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vision Guard</title>
  <style>
    :root{--bg:#09111e;--panel:#111b2b;--panel2:#162234;--text:#f4f7fb;--muted:#9fb0c6;--line:#24354c;--accent:#1da2c5;--accent2:#8dd7cb}
    *{box-sizing:border-box} body{margin:0;background:radial-gradient(circle at top,#163a52 0,#09111e 48%);color:var(--text);font-family:Segoe UI,system-ui,sans-serif}
    .wrap{max-width:1280px;margin:0 auto;padding:24px}
    .hero{padding:24px;border-radius:24px;background:linear-gradient(135deg,#13344a 0,#1b7384 58%,#a5d7c8 100%);box-shadow:0 24px 60px rgba(0,0,0,.28)}
    .hero h1{margin:0 0 8px;font-size:36px}.hero p{margin:0;color:#eef9ff}
    .grid{display:grid;grid-template-columns:380px 1fr;gap:18px;margin-top:18px}
    .panel{background:rgba(17,27,43,.92);border:1px solid var(--line);border-radius:22px;padding:16px}
    .label{display:inline-block;background:#1699bb;color:#fff;border-radius:10px;padding:6px 10px;font-weight:700;margin-bottom:10px}
    .status{margin:10px 0;color:var(--muted)} .meta{color:var(--muted);font-size:14px;line-height:1.7}
    .preview,.gallery img{width:100%;border-radius:16px;border:1px solid var(--line);background:#0d1624}
    .preview{min-height:220px;object-fit:cover}
    .field,.btn,select{width:100%;border-radius:14px;border:1px solid #314764;background:var(--panel2);color:var(--text);padding:14px}
    .btn{background:linear-gradient(135deg,#0f88ab,#21b4d3);font-weight:700;cursor:pointer}
    .btn.secondary{background:#203149}.btn:disabled{opacity:.55;cursor:not-allowed}
    .row{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .results{display:grid;gap:14px}.card{border:1px solid var(--line);border-radius:18px;padding:14px;background:#0f1928}
    .card h3{margin:0 0 6px}.muted{color:var(--muted)} .gallery{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
    .downloads a{display:inline-block;margin-right:12px;color:#9fe8ff}.hidden{display:none}
    @media (max-width: 980px){.grid{grid-template-columns:1fr}}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Vision Guard</h1>
      <p>Scan a video first. Then search with natural language, review matched frames, and export only what you want.</p>
    </div>
    <div class="grid">
      <section class="panel">
        <div class="label">scan video</div>
        <div class="row">
          <select id="sample"></select>
          <button class="btn secondary" onclick="startSampleScan()">scan sample</button>
        </div>
        <div style="height:10px"></div>
        <input id="upload" class="field" type="file" accept="video/*">
        <div style="height:10px"></div>
        <button id="uploadBtn" class="btn" onclick="startUploadScan()">upload and scan</button>
        <div id="status" class="status">ready</div>
        <img id="preview" class="preview hidden" alt="preview">
        <div id="meta" class="meta"></div>
        <div style="height:18px"></div>
        <div class="label">query</div>
        <input id="query" class="field" placeholder="person sitting near gate, white car entering, fight near road">
        <div style="height:10px"></div>
        <button id="findBtn" class="btn secondary" onclick="findMatches()" disabled>find matches</button>
      </section>
      <section class="panel">
        <div class="label">matched frames</div>
        <div id="answer" class="muted">No scan results yet.</div>
        <div id="downloads" class="downloads" style="margin-top:10px"></div>
        <div id="results" class="results" style="margin-top:14px"></div>
      </section>
    </div>
  </div>
  <script>
    let scanTimer = null;
    let currentRows = [];

    async function loadSamples() {
      const res = await fetch('/api/samples');
      const data = await res.json();
      const select = document.getElementById('sample');
      select.innerHTML = '';
      for (const name of data.samples) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
      }
    }

    function renderMeta(meta) {
      if (!meta) {
        document.getElementById('meta').innerHTML = '';
        return;
      }
      document.getElementById('meta').innerHTML =
        `video: <b>${meta.video}</b><br>` +
        `duration: ${meta.duration}s<br>` +
        `fps: ${meta.fps}<br>` +
        `sampled every: ${meta.sample_sec}s<br>` +
        `indexed windows: ${meta.segments}<br>` +
        `retriever: ${meta.retriever}<br>` +
        `verifier: ${meta.verifier}`;
    }

    function renderRows(rows) {
      currentRows = rows || [];
      const wrap = document.getElementById('results');
      const answer = document.getElementById('answer');
      const downloads = document.getElementById('downloads');
      downloads.innerHTML = '';
      if (!rows || !rows.length) {
        wrap.innerHTML = '';
        answer.textContent = 'No strong matches found.';
        return;
      }
      answer.textContent = `Found ${rows.length} matched frames.`;
      wrap.innerHTML = rows.map((row, idx) => `
        <div class="card">
          <div style="display:flex;justify-content:space-between;gap:12px;align-items:start">
            <div>
              <h3>${idx + 1}. ${row.peak_ts}s</h3>
              <div class="muted">${row.start}s - ${row.end}s</div>
            </div>
            <label><input type="checkbox" value="${row.label}" checked> export</label>
          </div>
          <div style="height:10px"></div>
          ${row.image_url ? `<img src="${row.image_url}" class="preview" alt="match">` : ''}
          <div style="height:10px"></div>
          <div>${row.low_confidence ? '<span class="muted">low confidence</span><br>' : ''}${row.summary}</div>
          <div class="muted" style="margin-top:8px">objects: ${row.objects.join(', ') || 'none'}</div>
        </div>
      `).join('');
      downloads.innerHTML = `<button class="btn" onclick="exportSelected()">export selected</button>`;
    }

    async function pollScan() {
      const res = await fetch('/api/scan/status');
      const data = await res.json();
      document.getElementById('status').textContent = data.status;
      renderMeta(data.meta);
      const preview = document.getElementById('preview');
      if (data.preview_b64) {
        preview.src = `data:image/jpeg;base64,${data.preview_b64}`;
        preview.classList.remove('hidden');
      }
      document.getElementById('findBtn').disabled = data.scan_running || !data.meta;
      document.getElementById('uploadBtn').disabled = data.scan_running;
      if (!data.scan_running && scanTimer) {
        clearInterval(scanTimer);
        scanTimer = null;
      }
    }

    async function startSampleScan() {
      const sample = document.getElementById('sample').value;
      await fetch('/api/scan/sample', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({sample})
      });
      document.getElementById('answer').textContent = 'Scanning sample video...';
      renderRows([]);
      if (!scanTimer) scanTimer = setInterval(pollScan, 1000);
      pollScan();
    }

    async function startUploadScan() {
      const file = document.getElementById('upload').files[0];
      if (!file) return;
      const form = new FormData();
      form.append('video', file);
      await fetch('/api/scan/upload', {method: 'POST', body: form});
      document.getElementById('answer').textContent = 'Uploading and scanning video...';
      renderRows([]);
      if (!scanTimer) scanTimer = setInterval(pollScan, 1000);
      pollScan();
    }

    async function findMatches() {
      const query = document.getElementById('query').value.trim();
      if (!query) return;
      document.getElementById('answer').textContent = 'Searching...';
      const res = await fetch('/api/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query})
      });
      const data = await res.json();
      renderRows(data.rows || []);
    }

    async function exportSelected() {
      const chosen = Array.from(document.querySelectorAll('#results input[type=checkbox]:checked')).map(x => x.value);
      const query = document.getElementById('query').value.trim();
      const res = await fetch('/api/export', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({picks: chosen, query})
      });
      const data = await res.json();
      const downloads = document.getElementById('downloads');
      downloads.innerHTML = `
        <a href="${data.zip_url}" target="_blank">zip</a>
        <a href="${data.html_url}" target="_blank">html report</a>
        <a href="${data.csv_url}" target="_blank">csv report</a>
      `;
    }

    loadSamples();
    pollScan();
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.get("/api/samples")
def api_samples():
    return {"samples": _sample_videos()}


@app.get("/api/scan/status")
def api_scan_status():
    return _get_state()


@app.post("/api/scan/sample")
def api_scan_sample(payload: dict):
    sample = payload.get("sample", "")
    path = ROOT / "assets" / sample
    if not path.exists():
        raise HTTPException(status_code=404, detail="sample not found")
    if _get_state()["scan_running"]:
        raise HTTPException(status_code=409, detail="scan already running")
    threading.Thread(target=_scan_worker, args=(str(path),), daemon=True).start()
    return {"ok": True}


@app.post("/api/scan/upload")
def api_scan_upload(video: UploadFile = File(...)):
    if _get_state()["scan_running"]:
        raise HTTPException(status_code=409, detail="scan already running")
    suffix = Path(video.filename or "upload.mp4").suffix or ".mp4"
    out_path = UPLOAD_DIR / f"upload_{threading.get_ident()}{suffix}"
    with open(out_path, "wb") as f:
        f.write(video.file.read())
    threading.Thread(target=_scan_worker, args=(str(out_path),), daemon=True).start()
    return {"ok": True}


@app.post("/api/search")
def api_search(payload: dict):
    query = (payload.get("query") or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query required")
    if _get_state()["scan_running"]:
        raise HTTPException(status_code=409, detail="scan in progress")
    if not pipe.idx:
        raise HTTPException(status_code=400, detail="scan a video first")
    rows = None
    for hits in pipe.search_stream(query, top_k=4):
        rows = pipe.prepare_hits(hits, query)
        if rows:
            break
    if rows is None:
        rows = pipe.prepare_hits(pipe.search(query, top_k=4), query)
    ui_rows = _rows_for_ui(rows)
    _set_state(hits=rows, query=query)
    return {"rows": ui_rows}


@app.post("/api/export")
def api_export(payload: dict):
    picks = payload.get("picks") or []
    query = (payload.get("query") or "").strip()
    if not picks or not query:
        raise HTTPException(status_code=400, detail="picks and query required")
    if not pipe.last_hits:
        raise HTTPException(status_code=400, detail="no matches ready")
    zipf, html, csv = pipe.export_selected(picks, query)
    if not zipf:
        raise HTTPException(status_code=400, detail="nothing selected")
    return {
        "zip_url": _image_url(zipf),
        "html_url": _image_url(html),
        "csv_url": _image_url(csv),
    }


@app.get("/artifact")
def artifact(path: str):
    safe = _safe_path(path)
    media_type = None
    if safe.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    elif safe.suffix.lower() == ".png":
        media_type = "image/png"
    elif safe.suffix.lower() == ".mp4":
        media_type = "video/mp4"
    elif safe.suffix.lower() == ".csv":
        media_type = "text/csv"
    elif safe.suffix.lower() == ".html":
        media_type = "text/html"
    elif safe.suffix.lower() == ".zip":
        media_type = "application/zip"
    return FileResponse(str(safe), media_type=media_type, filename=safe.name)


if __name__ == "__main__":
    host = _runtime_host()
    if host == "127.0.0.1":
        print("Open Vision Guard at http://127.0.0.1:7860")
    else:
        print("Open Vision Guard through your notebook or remote host on port 7860")
    uvicorn.run(app, host=host, port=7860, log_level="info")
