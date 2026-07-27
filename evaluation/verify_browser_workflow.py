"""Drive the real VisionGuard browser UI through Edge's DevTools protocol."""

from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import websocket

ROOT = Path(__file__).resolve().parents[1]
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


class DevTools:
    def __init__(self, url: str):
        self.socket = websocket.create_connection(url, timeout=10)
        self.sequence = 0
        self.console_errors: list[str] = []
        self.network_errors: list[str] = []
        self.cancelled_requests: list[str] = []
        self.api_responses: list[dict] = []

    def close(self) -> None:
        self.socket.close()

    def _event(self, message: dict) -> None:
        method = message.get("method")
        params = message.get("params", {})
        if method == "Runtime.exceptionThrown":
            self.console_errors.append(params.get("exceptionDetails", {}).get("text", "Runtime exception"))
        elif method == "Runtime.consoleAPICalled" and params.get("type") in {"error", "assert"}:
            values = [item.get("value", item.get("description", "")) for item in params.get("args", [])]
            self.console_errors.append(" ".join(str(value) for value in values))
        elif method == "Log.entryAdded" and params.get("entry", {}).get("level") == "error":
            self.console_errors.append(params["entry"].get("text", "Browser log error"))
        elif method == "Network.loadingFailed":
            error = params.get("errorText", "Network request failed")
            if error == "net::ERR_ABORTED":
                self.cancelled_requests.append(error)
            else:
                self.network_errors.append(error)
        elif method == "Network.responseReceived":
            response = params.get("response", {})
            if "/api/" in response.get("url", ""):
                self.api_responses.append({"url": response["url"], "status": int(response.get("status", 0))})

    def call(self, method: str, params: dict | None = None) -> dict:
        self.sequence += 1
        request_id = self.sequence
        self.socket.send(json.dumps({"id": request_id, "method": method, "params": params or {}}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            self._event(message)

    def evaluate(self, expression: str):
        result = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True, "awaitPromise": True})
        if result.get("exceptionDetails"):
            details = result["exceptionDetails"]
            description = details.get("exception", {}).get("description")
            raise RuntimeError(description or details.get("text", "Evaluation failed"))
        return result.get("result", {}).get("value")


def wait_http(url: str, timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become ready: {url}")


def wait_until(devtools: DevTools, expression: str, timeout: float, description: str) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            if devtools.evaluate(expression):
                return
            last_error = None
        except RuntimeError as exc:
            # Page transitions can briefly invalidate the execution context.
            # Retry until the deadline, but retain the concrete browser error.
            last_error = exc
        time.sleep(0.25)
    suffix = f" Last browser error: {last_error}" if last_error else ""
    raise RuntimeError(f"Timed out waiting for {description}.{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=ROOT / "sample_videos" / "asset3.mp4")
    parser.add_argument("--query", default="find the person")
    parser.add_argument("--insufficient-query", default="person fell from cycle")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--debug-port", type=int, default=9223)
    args = parser.parse_args()
    video = args.video.resolve()
    if not EDGE.is_file() or not video.is_file():
        raise SystemExit("Microsoft Edge or the requested video is unavailable.")

    environment = os.environ.copy()
    environment.update({
        "VISION_GUARD_PORT": str(args.server_port),
        "VISION_GUARD_SKIP_WARMUP": "1",
        "VERIFIER_READY_TIMEOUT": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    server = subprocess.Popen(
        [str(ROOT / ".venv" / "Scripts" / "python.exe"), "run.py"],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    edge = None
    devtools = None
    try:
        base_url = f"http://127.0.0.1:{args.server_port}"
        wait_http(f"{base_url}/api/status")
        profile_context = tempfile.TemporaryDirectory(prefix="visionguard-edge-", ignore_cleanup_errors=True)
        profile = profile_context.name
        try:
            edge = subprocess.Popen([
                str(EDGE),
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--remote-allow-origins=*",
                f"--remote-debugging-port={args.debug_port}",
                f"--user-data-dir={profile}",
                "about:blank",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            wait_http(f"http://127.0.0.1:{args.debug_port}/json/version")
            pages = json.loads(urlopen(f"http://127.0.0.1:{args.debug_port}/json").read())
            page = next(item for item in pages if item.get("type") == "page")
            devtools = DevTools(page["webSocketDebuggerUrl"])
            for domain in ("Page", "Runtime", "Network", "Log", "DOM"):
                devtools.call(f"{domain}.enable")
            devtools.call("Page.navigate", {"url": base_url})
            wait_until(devtools, "document.readyState === 'complete'", 20, "page load")

            document = devtools.call("DOM.getDocument", {"depth": 1})["root"]["nodeId"]
            input_node = devtools.call("DOM.querySelector", {"nodeId": document, "selector": "#videoUpload"})["nodeId"]
            devtools.call("DOM.setFileInputFiles", {"nodeId": input_node, "files": [str(video)]})
            devtools.evaluate("document.querySelector('#videoUpload').dispatchEvent(new Event('change', {bubbles:true})); document.querySelector('#scanForm').requestSubmit(); true")
            try:
                wait_until(devtools, "document.querySelector('#indexButton')?.disabled === false", 120, "index button readiness")
            except RuntimeError as exc:
                diagnostics = devtools.evaluate("({scanStatus:document.querySelector('#scanStatus')?.textContent, modelNotice:document.querySelector('#modelNotice')?.textContent, nvidia:document.querySelector('#nvidiaStatus')?.textContent, provider:document.querySelector('#providerName')?.textContent, bodyReady:document.readyState})")
                raise RuntimeError(f"{exc} UI diagnostics: {json.dumps(diagnostics)} API responses: {json.dumps(devtools.api_responses[-12:])}") from exc
            devtools.evaluate("document.querySelector('#indexButton').click(); true")
            try:
                wait_until(devtools, "document.querySelector('#queryInput')?.disabled === false", 1200, "query-ready state")
            except RuntimeError as exc:
                diagnostics = devtools.evaluate("({scanStatus:document.querySelector('#scanStatus')?.textContent, progress:document.querySelector('#scanProgressLabel')?.textContent, stages:[...document.querySelectorAll('#processingTimeline .stage-row')].map(x=>x.textContent), lastEvent:document.querySelector('#backendConsole li:last-child')?.textContent, bodyReady:document.readyState})")
                raise RuntimeError(f"{exc} UI diagnostics: {json.dumps(diagnostics)} API responses: {json.dumps(devtools.api_responses[-12:])}") from exc
            wait_until(devtools, "document.querySelectorAll('#evidenceStrip .evidence-thumb').length > 0", 20, "real evidence thumbnails")
            devtools.evaluate(f"document.querySelector('#queryInput').value = {json.dumps(args.query)}; document.querySelector('#queryButton').click(); true")
            wait_until(devtools, "document.querySelectorAll('#resultsList .match-card').length > 0", 60, "evidence-backed query results")
            wait_until(devtools, "document.querySelector('#resultsList .match-card__image')?.complete === true", 20, "result image load")
            devtools.evaluate("document.querySelectorAll('#resultsList .match-card__content')[0].click(); true")
            time.sleep(0.5)

            desktop = devtools.evaluate("({queryDisabled:document.querySelector('#queryInput').disabled, stages:[...document.querySelectorAll('#processingTimeline .stage-row small')].map(x=>x.textContent), evidenceCount:document.querySelectorAll('#evidenceStrip .evidence-thumb').length, eventCount:document.querySelectorAll('#backendConsole li').length, resultCount:document.querySelectorAll('#resultsList .match-card').length, resultTitle:document.querySelectorAll('#resultsList .match-card__content strong')[0].textContent, inspectorVisible:getComputedStyle(document.querySelector('#frameInspector')).display !== 'none', videoTime:document.querySelector('#videoPreview').currentTime, imageWidth:document.querySelector('#resultsList .match-card__image').naturalWidth, provider:document.querySelector('#providerName').textContent, textModel:document.querySelector('#textModelStatus').textContent, visionModel:document.querySelector('#visionModelStatus').textContent, bodyText:document.body.innerText})")
            expected_seconds = float(desktop["resultTitle"].split("s", 1)[0])
            if args.screenshot:
                screenshot_path = args.screenshot.resolve()
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                screenshot = devtools.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": True})
                screenshot_path.write_bytes(base64.b64decode(screenshot["data"]))

            devtools.evaluate(f"document.querySelector('#queryInput').value = {json.dumps(args.insufficient_query)}; document.querySelector('#queryButton').click(); true")
            wait_until(devtools, "document.querySelector('#queryStatus').textContent.toLowerCase().includes('insufficient evidence')", 60, "honest insufficient-evidence response")
            insufficient = devtools.evaluate("({status:document.querySelector('#queryStatus').textContent, resultCount:document.querySelectorAll('#resultsList .match-card').length})")

            devtools.call("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
            time.sleep(0.3)
            mobile = devtools.evaluate("({innerWidth:window.innerWidth, scrollWidth:document.documentElement.scrollWidth, bodyScrollWidth:document.body.scrollWidth})")

            result = {
                "page_loaded": True,
                "query_disabled_after_ready": desktop["queryDisabled"],
                "terminal_stage_rows": desktop["stages"],
                "evidence_thumbnail_count": desktop["evidenceCount"],
                "backend_event_count": desktop["eventCount"],
                "result_count": desktop["resultCount"],
                "result_image_width": desktop["imageWidth"],
                "frame_inspector_visible": desktop["inspectorVisible"],
                "result_timestamp_seconds": expected_seconds,
                "video_current_time": desktop["videoTime"],
                "seek_matches_result": abs(desktop["videoTime"] - expected_seconds) < 0.15,
                "provider": desktop["provider"],
                "text_model_status": desktop["textModel"],
                "vision_model_status": desktop["visionModel"],
                "insufficient_query": args.insufficient_query,
                "insufficient_query_status": insufficient["status"],
                "insufficient_query_result_count": insufficient["resultCount"],
                "insufficient_query_is_grounded": insufficient["resultCount"] == 0,
                "ocr_marked_skipped": "ocr completed" in desktop["bodyText"].lower() and "ocr is not implemented" in desktop["bodyText"].lower(),
                "captioning_marked_skipped": "captions generated" in desktop["bodyText"].lower() and "caption generation is not implemented" in desktop["bodyText"].lower(),
                "mobile_metrics": mobile,
                "mobile_has_no_horizontal_overflow": mobile["scrollWidth"] <= mobile["innerWidth"] and mobile["bodyScrollWidth"] <= mobile["innerWidth"],
                "api_responses": devtools.api_responses,
                "api_http_errors": [row for row in devtools.api_responses if row["status"] >= 400],
                "console_errors": devtools.console_errors,
                "network_errors": devtools.network_errors,
                "cancelled_media_or_image_requests": devtools.cancelled_requests,
            }
            required = (
                not result["query_disabled_after_ready"],
                result["evidence_thumbnail_count"] > 0,
                result["backend_event_count"] > 0,
                result["result_count"] > 0,
                result["result_image_width"] > 0,
                result["frame_inspector_visible"],
                result["seek_matches_result"],
                result["insufficient_query_is_grounded"],
                result["ocr_marked_skipped"],
                result["captioning_marked_skipped"],
                result["mobile_has_no_horizontal_overflow"],
                not result["api_http_errors"],
                not result["console_errors"],
                not result["network_errors"],
            )
            print(json.dumps(result, indent=2))
            if not all(required):
                raise RuntimeError("Browser verification failed one or more required checks.")
        finally:
            if edge is not None:
                edge.terminate()
                try:
                    edge.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    edge.kill()
                edge = None
            profile_context.cleanup()
    finally:
        if devtools is not None:
            devtools.close()
        if edge is not None:
            edge.terminate()
            try:
                edge.wait(timeout=5)
            except subprocess.TimeoutExpired:
                edge.kill()
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
