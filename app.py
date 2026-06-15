import os
import warnings

import gradio as gr

from cache_utils import setup_cache
from pipeline import VisionGuardPipeline

setup_cache()
warnings.filterwarnings("ignore", category=DeprecationWarning, module="gradio.*")
warnings.filterwarnings("ignore", category=UserWarning, message="The parameters have been moved from the Blocks constructor to the launch\\(\\) method in Gradio 6\\.0: theme, css.*")
warnings.filterwarnings("ignore", message="The 'theme' parameter in the Blocks constructor will be removed in Gradio 6\\.0.*")
warnings.filterwarnings("ignore", message="The 'css' parameter in the Blocks constructor will be removed in Gradio 6\\.0.*")
pipe = VisionGuardPipeline()
theme = gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")
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
        f"- indexed windows: `{meta['segments']}`\n"
        f"- retriever: `{meta.get('retriever', 'numpy')}`\n"
        f"- verifier: `{meta.get('verifier', 'none')}`"
    )


def _ans(q, rows):
    out = [f"## answer for `{q}`", ""]
    if not rows:
        out.append("no strong matches found")
        return "\n".join(out)
    for i, x in enumerate(rows, 1):
        out.append(f"{i}. `best frame {x.get('peak_ts', x['start']):.2f}s | clip {x['start']:.2f}s - {x['end']:.2f}s`")
        prefix = "low confidence: " if x.get("low_confidence") else ""
        out.append(f"   {prefix}{x['summary']}")
    return "\n".join(out)


def _gallery(rows):
    out = []
    for x in rows:
        frame_path = x.get("display_frame_path") or x.get("frame_path")
        if not frame_path:
            continue
        prefix = "low confidence | " if x.get("low_confidence") else ""
        out.append((frame_path, f"{x['label']} | {prefix}{x['summary']}"))
    return out


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
    if not pipe.idx:
        return "scan a video first", "", [], blank_pick, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    if not q or not q.strip():
        return "enter a natural-language query", "", [], blank_pick, [], "", "", [], gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    hits = pipe.search(q.strip(), top_k=4)
    seg = pipe.prepare_hits(hits, q.strip())
    rows = [[i, round(x["score"], 4), round(x.get("peak_ts", x["start"]), 2), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])] for i, x in enumerate(seg, 1)]
    ans = _ans(q.strip(), seg)
    choices = [x["label"] for x in seg]
    gal = _gallery(seg)
    if not seg:
        note = "### matched frames\n\nNo strong frame matches were found for this query."
    elif any(x.get("low_confidence") for x in seg):
        note = "### matched frames\n\nThe gallery below shows the nearest available sampled frames. These results are low confidence, so review them carefully before export."
    else:
        note = "### matched frames\n\nThe gallery below shows the top sampled frames for your query. Select any rows you want to export as clips and reports."
    return "matches ready", ans, rows, gr.update(choices=choices, value=choices[:1]), gal, note, q.strip(), seg, gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)


def export_selected(picks, q, hits):
    if not hits or not picks:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    pipe.last_hits = hits
    zipf, html, csv = pipe.export_selected(picks, q)
    return gr.update(visible=True, value=zipf), gr.update(visible=True, value=html), gr.update(visible=True, value=csv)


with gr.Blocks(title="VisionGuard AI", css=css, theme=theme) as demo:
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
            query = gr.Textbox(label="query", placeholder="person sitting near gate, white car entering, fight near road, car accident", interactive=False)
            find_btn = gr.Button("step 2: find matches", interactive=False)

        with gr.Column(scale=2, elem_classes="panel result-stack"):
            answer = gr.Markdown(elem_classes="tight-md")
            table = gr.Dataframe(headers=["rank", "score", "moment", "start", "end", "summary", "objects"], interactive=False, wrap=True)
            pick = gr.CheckboxGroup(label="choose clips to export")
            export_btn = gr.Button("export selected")
            with gr.Row(elem_classes="export-files"):
                zipf = gr.File(label="zip", visible=False)
                html = gr.File(label="html report", visible=False)
                csv = gr.File(label="csv report", visible=False)
            gallery = gr.Gallery(label="matched frames", columns=2, height="auto")
            match_md = gr.Markdown(elem_classes="tight-md")

    scan_btn.click(scan_only, [video], [status, live, info, query, q_state, hits_state])
    scan_btn.click(lambda: gr.update(interactive=True), None, find_btn)
    find_btn.click(find_query, [query], [status, answer, table, pick, gallery, match_md, q_state, hits_state, zipf, html, csv])
    export_btn.click(export_selected, [pick, q_state, hits_state], [zipf, html, csv])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.launch(server_name="0.0.0.0", share=share, show_error=True)
