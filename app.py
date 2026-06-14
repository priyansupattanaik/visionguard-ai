import os

import gradio as gr

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

setup_cache()
pipe = VisionGuardPipeline()
css = """
.gradio-container{max-width:1180px!important}
.hero{padding:22px 24px;border-radius:22px;background:linear-gradient(135deg,#16364a 0%,#1f6d78 60%,#b7d9c8 100%);color:#fff;margin-bottom:16px}
.hero h1{margin:0 0 6px 0;font-size:34px}
.hero p{margin:0;font-size:15px;opacity:.96}
.card{padding:12px 14px;border:1px solid #d7e4ea;border-radius:14px;background:#f7fbfd}
"""


def _meta(meta):
    return (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- indexed windows: `{meta['segments']}`"
    )


def scan(video, q):
    if not video:
        yield "upload a video first", None, "", [], None, None, gr.update(choices=[], value=[]), None, None, []
        return
    if not q or not q.strip():
        yield "enter a natural-language query", None, "", [], None, None, gr.update(choices=[], value=[]), None, None, []
        return
    yield "starting scan", None, "", [], None, None, gr.update(choices=[], value=[]), None, None, []
    meta = None
    for ev in pipe.index_video_iter(video):
        if ev["kind"] == "preview":
            yield ev["status"], ev["image"], "", [], None, None, gr.update(choices=[], value=[]), None, None, []
        else:
            meta = ev["meta"]
    hits = pipe.search(q.strip(), top_k=4)
    seg = pipe.segment_hits(hits, q.strip())
    rows = [[i, round(x["score"], 4), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])] for i, x in enumerate(seg, 1)]
    gal = []
    for x in seg:
        for fp in x["frames"][:3]:
            gal.append((fp, x["label"]))
    choices = [x["label"] for x in seg]
    first = seg[0]["clip"] if seg else None
    all_clips = [x["clip"] for x in seg] + [x["raw_clip"] for x in seg]
    yield "scan complete", None, _meta(meta), rows, first, all_clips, gr.update(choices=choices, value=choices[:1]), gal, q.strip(), seg


def export_selected(picks, q, hits):
    if not hits or not picks:
        return None, None, None
    pipe.last_hits = hits
    return pipe.export_selected(picks)


with gr.Blocks(title="VisionGuard AI") as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>VisionGuard AI</h1>
  <p>Upload a video, type a query, watch indexing live, review each matched segmented clip, and export only the clips you want.</p>
</div>
"""
    )
    q_state = gr.State("")
    hits_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="cctv video")
            query = gr.Textbox(label="query", placeholder="person sitting near gate, white car entering, group near entrance")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            scan_btn = gr.Button("scan video", variant="primary")
            status = gr.Markdown("ready")
            live = gr.Image(label="live indexing preview", interactive=False)
            info = gr.Markdown()
            gr.Markdown("<div class='card'>In Colab, mount Drive before running if you want model downloads to stay cached across sessions.</div>")

        with gr.Column(scale=2):
            table = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False)
            clip = gr.Video(label="segmented top clip")
            clips = gr.Files(label="all matched clips")
            pick = gr.CheckboxGroup(label="choose clips to export")
            export_btn = gr.Button("export selected")
            zipf = gr.File(label="zip")
            html = gr.File(label="html report")
            csv = gr.File(label="csv report")
            gallery = gr.Gallery(label="segmented preview frames", columns=3, height="auto")

    scan_btn.click(scan, [video, query], [status, live, info, table, clip, clips, pick, gallery, q_state, hits_state])
    export_btn.click(export_selected, [pick, q_state, hits_state], [zipf, html, csv])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.launch(server_name="0.0.0.0", share=share, show_error=True, css=css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"))
