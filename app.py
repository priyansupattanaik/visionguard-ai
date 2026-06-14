import os
import warnings

import gradio as gr

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

setup_cache()
warnings.filterwarnings("ignore", message="The parameters have been moved from the Blocks constructor to the launch\\(\\) method in Gradio 6\\.0: theme, css.*")
pipe = VisionGuardPipeline()
css = """
.gradio-container{max-width:1240px!important}
.gradio-container,.gradio-container *{box-sizing:border-box}
.hero{padding:22px 24px;border-radius:22px;background:linear-gradient(135deg,#16364a 0%,#1f6d78 60%,#b7d9c8 100%);color:#fff;margin-bottom:16px}
.hero h1{margin:0 0 6px 0;font-size:34px}
.hero p{margin:0;font-size:15px;opacity:.96}
.app-shell{gap:18px}
.panel{border:1px solid #253043;border-radius:18px;background:#111827;padding:14px}
.note-card{padding:12px 14px;border:1px solid #314056;border-radius:16px;background:#182334;color:#d6e4f0;margin-top:12px;line-height:1.5}
.tight-md{margin-top:8px}
.tight-md p{margin:0}
.result-stack{gap:14px}
.export-files{gap:14px}
.hidden-empty{min-height:0!important}
"""


def _meta(meta):
    return (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- indexed windows: `{meta['segments']}`"
    )


def _ans(q, rows):
    out = [f"## answer for `{q}`", ""]
    if not rows:
        out.append("no strong matches found")
        return "\n".join(out)
    for i, x in enumerate(rows, 1):
        out.append(f"{i}. `{x['start']:.2f}s - {x['end']:.2f}s`")
        out.append(f"   {x['summary']}")
    return "\n".join(out)


def scan_only(video):
    if not video:
        yield "upload a video first", None, "", gr.update(interactive=False), "", []
        return
    yield "starting scan", None, "", gr.update(interactive=False), "", []
    meta = None
    for ev in pipe.index_video_iter(video):
        if ev["kind"] == "preview":
            yield ev["status"], ev["image"], "", gr.update(interactive=False), "", []
        else:
            meta = ev["meta"]
    yield "scan complete", None, _meta(meta), gr.update(interactive=True), "", []


def find_query(q):
    blank_pick = gr.update(choices=[], value=[])
    blank_one = gr.update(choices=[], value=None)
    if not pipe.idx:
        return "scan a video first", "", [], None, [], blank_pick, blank_one, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    if not q or not q.strip():
        return "enter a natural-language query", "", [], None, [], blank_pick, blank_one, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    hits = pipe.search(q.strip(), top_k=4)
    seg = pipe.prepare_hits(hits, q.strip())
    rows = [[i, round(x["score"], 4), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])] for i, x in enumerate(seg, 1)]
    ans = _ans(q.strip(), seg)
    choices = [x["label"] for x in seg]
    first = seg[0]["raw_clip"] if seg else None
    ready = [x["raw_clip"] for x in seg if x["raw_clip"]]
    note = f"### {seg[0]['label']}\n\n{seg[0]['summary']}\n\nfirst clip is ready now. other clips are trimming in the background." if seg else ""
    return "matches ready", ans, rows, first, ready, gr.update(choices=choices, value=choices[:1]), gr.update(choices=choices, value=choices[0] if choices else None), [], note, q.strip(), seg, gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def show_match(label, q, hits):
    if not hits or not label:
        return None, [], ""
    pipe.last_hits = hits
    return pipe.pick_match(label, q)


def export_selected(picks, q, hits):
    if not hits or not picks:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    pipe.last_hits = hits
    zipf, html, csv = pipe.export_selected(picks, q)
    return gr.update(visible=True, value=zipf), gr.update(visible=True, value=html), gr.update(visible=True, value=csv)


with gr.Blocks(title="VisionGuard AI", css=css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")) as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>VisionGuard AI</h1>
  <p>Step 1: scan the video. Step 2: write a query and find matches. Then review each match and export only what you want.</p>
</div>
"""
    )
    q_state = gr.State("")
    hits_state = gr.State([])

    with gr.Row(elem_classes="app-shell"):
        with gr.Column(scale=1, elem_classes="panel"):
            video = gr.Video(label="cctv video", elem_classes="hidden-empty")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            scan_btn = gr.Button("step 1: scan video", variant="primary")
            status = gr.Markdown("ready")
            live = gr.Image(label="live indexing preview", interactive=False, elem_classes="hidden-empty")
            info = gr.Markdown(elem_classes="tight-md")
            gr.HTML("<div class='note-card'>After scanning finishes, the query box becomes active. In Colab, mount Drive once and use git pull for updates.</div>")

            query = gr.Textbox(label="query", placeholder="person sitting near gate, white car entering, fight near road, car accident", interactive=False)
            find_btn = gr.Button("step 2: find matches", interactive=False)

        with gr.Column(scale=2, elem_classes="panel result-stack"):
            answer = gr.Markdown(elem_classes="tight-md")
            table = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False, wrap=True)
            pick_one = gr.Dropdown(label="view one match", choices=[], value=None)
            clip = gr.Video(label="selected match clip", elem_classes="hidden-empty")
            clips = gr.Files(label="all matched clips")
            pick = gr.CheckboxGroup(label="choose clips to export")
            export_btn = gr.Button("export selected")
            with gr.Row(elem_classes="export-files"):
                zipf = gr.File(label="zip", visible=False)
                html = gr.File(label="html report", visible=False)
                csv = gr.File(label="csv report", visible=False)
            gallery = gr.Gallery(label="segmented preview frames", columns=3, height="auto")
            match_md = gr.Markdown(elem_classes="tight-md")

    scan_btn.click(scan_only, [video], [status, live, info, query, q_state, hits_state])
    scan_btn.click(lambda: gr.update(interactive=True), None, find_btn)
    find_btn.click(find_query, [query], [status, answer, table, clip, clips, pick, pick_one, gallery, match_md, q_state, hits_state, zipf, html, csv])
    pick_one.change(show_match, [pick_one, q_state, hits_state], [clip, gallery, match_md])
    export_btn.click(export_selected, [pick, q_state, hits_state], [zipf, html, csv])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.launch(server_name="0.0.0.0", share=share, show_error=True)
