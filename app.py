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
    blank_pick = gr.update(choices=[], value=[])
    blank_one = gr.update(choices=[], value=None)
    if not video:
        yield "upload a video first", None, "", [], None, None, blank_pick, blank_one, None, "", None, []
        return
    if not q or not q.strip():
        yield "enter a natural-language query", None, "", [], None, None, blank_pick, blank_one, None, "", None, []
        return
    yield "starting scan", None, "", [], None, None, blank_pick, blank_one, None, "", None, []
    meta = None
    for ev in pipe.index_video_iter(video):
        if ev["kind"] == "preview":
            yield ev["status"], ev["image"], "", [], None, None, blank_pick, blank_one, None, "", None, []
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
    note = f"### {seg[0]['label']}\n\n{seg[0]['summary']}" if seg else ""
    yield "scan complete", None, _meta(meta), rows, first, all_clips, gr.update(choices=choices, value=choices[:1]), gr.update(choices=choices, value=choices[0] if choices else None), gal, note, q.strip(), seg


def show_match(label, hits):
    if not hits or not label:
        return None, [], ""
    pipe.last_hits = hits
    return pipe.pick_match(label)


def export_selected(picks, q, hits):
    if not hits or not picks:
        return None, None, None
    pipe.last_hits = hits
    return pipe.export_selected(picks)


with gr.Blocks(title="VisionGuard AI", css=css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")) as demo:
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
            query = gr.Textbox(label="query", placeholder="person sitting near gate, white car entering, fight near road, car accident")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            scan_btn = gr.Button("scan video", variant="primary")
            status = gr.Markdown("ready")
            live = gr.Image(label="live indexing preview", interactive=False)
            info = gr.Markdown()
            gr.Markdown("<div class='card'>In Colab, mount Drive once. After that, use git pull in the same folder instead of deleting and cloning the repo again.</div>")

        with gr.Column(scale=2):
            table = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False)
            pick_one = gr.Dropdown(label="view one match", choices=[], value=None)
            clip = gr.Video(label="selected match clip")
            clips = gr.Files(label="all matched clips")
            pick = gr.CheckboxGroup(label="choose clips to export")
            export_btn = gr.Button("export selected")
            zipf = gr.File(label="zip")
            html = gr.File(label="html report")
            csv = gr.File(label="csv report")
            gallery = gr.Gallery(label="segmented preview frames", columns=3, height="auto")
            match_md = gr.Markdown()

    scan_btn.click(scan, [video, query], [status, live, info, table, clip, clips, pick, pick_one, gallery, match_md, q_state, hits_state])
    pick_one.change(show_match, [pick_one, hits_state], [clip, gallery, match_md])
    export_btn.click(export_selected, [pick, q_state, hits_state], [zipf, html, csv])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.launch(server_name="0.0.0.0", share=share, show_error=True)
