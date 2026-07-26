import os

os.environ["VISION_GUARD_SKIP_WARMUP"] = "1"

from visionguard.web_app.server import create_app


class DummyPipeline:
    def __init__(self):
        self.idx = None
        self.last_hits = []

    def warmup_status(self):
        return "Models loading..."

    def verification_mode(self):
        return "nvidia_api_unconfigured"

    def model_health(self, refresh=False):
        return {
            "selected_provider": "llama_cpp",
            "text_model": {"configured": True, "reachable": False, "url": "http://127.0.0.1:8080", "message": "not reachable"},
            "vision_model": {"configured": True, "reachable": False, "url": "http://127.0.0.1:8081", "message": "not reachable"},
            "detector": {"ready": True, "model": "yolo11m.pt", "message": "available"},
            "external_providers": {"nvidia": "disabled", "groq": "disabled"},
        }

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


def test_web_app_factory_imports_without_launching_server():
    app = create_app(testing=True, start_warmup=False, pipeline=DummyPipeline())
    assert app.name == "visionguard.web_app.server"


def test_index_is_custom_html_without_gradio_markers():
    client = make_client()
    res = client.get("/")
    body = res.get_data(as_text=True).lower()
    assert res.status_code == 200
    assert "visionguard" in body
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


def test_model_health_reports_selected_local_provider_without_blocking_video_backend():
    client = make_client()
    res = client.get("/api/model/health")
    data = res.get_json()

    assert res.status_code == 200
    assert data["selected_provider"] == "llama_cpp"
    assert data["text_model"]["reachable"] is False
    assert data["external_providers"] == {"nvidia": "disabled", "groq": "disabled"}


def test_indexing_is_rejected_before_a_missing_detector_can_create_a_failed_job():
    class MissingDetectorPipeline(DummyPipeline):
        def model_health(self, refresh=False):
            health = super().model_health(refresh=refresh)
            health["detector"] = {
                "ready": False,
                "model": "yolo11m.pt",
                "message": "YOLO model 'yolo11m.pt' is missing from '.models'. Run scripts/bootstrap_models.py before indexing a video.",
            }
            return health

    app = create_app(testing=True, start_warmup=False, pipeline=MissingDetectorPipeline())
    client = app.test_client()
    uploaded = client.post("/api/videos/upload", json={"sample": "asset3.mp4"}).get_json()

    response = client.post(f"/api/videos/{uploaded['video_id']}/index")

    assert response.status_code == 409
    assert "bootstrap_models.py" in response.get_json()["message"]
    assert client.get(f"/api/videos/{uploaded['video_id']}/status").get_json()["status"] == "waiting"


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


def test_readme_documents_primary_entry_point():
    with open("README.md", "r", encoding="utf-8") as fh:
        body = fh.read()
    assert "run.py" in body
