import os
import sys
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

os.environ.setdefault("VISION_GUARD_SKIP_WARMUP", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flask_app import create_app

BASE_URL = "http://127.0.0.1:7860"
OUT_DIR = Path("qa_screenshots")


class ServerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.app = create_app(start_warmup=False)
        self.server = make_server("127.0.0.1", 7860, self.app, threaded=True)
        self.ctx = self.app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()

    def stop(self):
        self.server.shutdown()
        self.ctx.pop()


def assert_no_overflow(page):
    overflow = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth")
    if overflow:
        raise AssertionError("horizontal overflow detected")


def capture(page, name, width, height):
    page.set_viewport_size({"width": width, "height": height})
    page.goto(BASE_URL, wait_until="networkidle")
    assert_no_overflow(page)
    path = OUT_DIR / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return path


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rows = []
    server = ServerThread()
    server.start()
    time.sleep(1)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        console_errors = []
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)

        for name, width in [
            ("desktop_1440", 1440),
            ("laptop_1024", 1024),
            ("tablet_768", 768),
            ("mobile_390", 390),
        ]:
            path = capture(page, name, width, 900)
            rows.append((name, str(path), width, "PASS", "layout captured"))

        page.set_viewport_size({"width": 1440, "height": 900})
        page.goto(BASE_URL, wait_until="networkidle")
        if "Gradio" in page.content() or "gradio-container" in page.content():
            raise AssertionError("Gradio marker found in Flask UI")
        page.select_option("#sampleSelect", "asset3.mp4")
        page.get_by_role("button", name="Scan video").click()
        page.wait_for_function("() => document.querySelector('#scanStatus').textContent.includes('Scan complete')", timeout=180000)
        path = OUT_DIR / "scan_result_asset3.png"
        page.screenshot(path=str(path), full_page=True)
        rows.append(("scan_result_asset3", str(path), 1440, "PASS", "asset3 scan completed"))

        page.locator("#queryInput").fill("person")
        with page.expect_response(lambda res: res.url.endswith("/api/query"), timeout=180000):
            page.evaluate("() => document.querySelector('#queryButton').click()")
        page.wait_for_function("() => document.querySelector('#queryStatus').textContent.includes('match')", timeout=10000)
        path = OUT_DIR / "query_result_person.png"
        page.screenshot(path=str(path), full_page=True)
        rows.append(("query_result_person", str(path), 1440, "PASS", "person query completed"))

        page.locator(".match-card input[type='checkbox']").first.click()
        with page.expect_response(lambda res: res.url.endswith("/api/export"), timeout=120000):
            page.evaluate("() => document.querySelector('#exportButton').click()")
        page.wait_for_function("() => document.querySelector('#exportStatus').textContent.length > 0 && !document.querySelector('#exportStatus').textContent.includes('started')", timeout=10000)
        path = OUT_DIR / "export_status_empty.png"
        page.screenshot(path=str(path), full_page=True)
        rows.append(("export_status", str(path), 1440, "PASS", "export status captured"))

        if console_errors:
            raise AssertionError("fatal console errors: " + " | ".join(console_errors[:5]))
        browser.close()
    server.stop()

    print("| Screenshot | Path | Viewport | Passed? | Notes |")
    print("| ---------- | ---- | -------: | ------: | ----- |")
    for name, path, width, result, notes in rows:
        print(f"| {name} | {path} | {width}px | {result} | {notes} |")


if __name__ == "__main__":
    main()
