import os

os.environ["VISION_GUARD_SKIP_WARMUP"] = "1"

from flask_app import create_app


class DummyPipeline:
    def __init__(self):
        self.idx = None
        self.last_hits = []

    def warmup_status(self):
        return "Models loading..."

    def verification_mode(self):
        return "nvidia_api_unconfigured"

    def export_selected_detailed(self, picks, query, segment_timeout=20):
        return {
            "ok": False,
            "message": "No selected matches to export.",
            "files": {},
            "rows": [],
            "export_mode": "none",
            "warnings": [],
        }


def make_client():
    app = create_app(testing=True, start_warmup=False, pipeline=DummyPipeline())
    return app.test_client()


def test_flask_app_imports_without_launching_server():
    app = create_app(testing=True, start_warmup=False, pipeline=DummyPipeline())
    assert app.name == "app.web.server"


def test_index_is_custom_html_without_gradio_markers():
    client = make_client()
    res = client.get("/")
    body = res.get_data(as_text=True).lower()
    assert res.status_code == 200
    assert "vision guard" in body
    assert "gradio-container" not in body
    assert "gradio" not in body


def test_status_returns_json_and_honest_api_configuration_label():
    client = make_client()
    res = client.get("/api/status")
    data = res.get_json()
    assert res.status_code == 200
    assert data["ok"] is True
    assert data["verification_mode"] == "nvidia_api_unconfigured"
    assert data["verification_label"] == "NVIDIA API key is not configured"


def test_assets_include_asset3_when_present():
    client = make_client()
    res = client.get("/api/assets")
    data = res.get_json()
    names = [asset["name"] for asset in data["assets"]]
    assert res.status_code == 200
    assert "asset3.mp4" in names


def test_query_before_scan_returns_controlled_error():
    client = make_client()
    res = client.post("/api/query", json={"query": "person"})
    data = res.get_json()
    assert res.status_code == 400
    assert data["ok"] is False
    assert data["message"] == "Scan a video before searching."


def test_export_empty_selection_returns_bounded_error():
    client = make_client()
    pipe = client.application.config["PIPELINE"]
    pipe.idx = {"video": "assets/asset3.mp4"}
    res = client.post("/api/export", json={"selected": [], "query": "person"})
    data = res.get_json()
    assert res.status_code == 400
    assert data["ok"] is False
    assert "Select at least one" in data["message"]


def test_static_files_load():
    client = make_client()
    css = client.get("/static/css/app.css")
    js = client.get("/static/js/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert b"gradio" not in css.data.lower()


def test_readme_documents_flask_entry_point():
    with open("README.md", "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "python flask_app.py" in body
