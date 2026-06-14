import os

import gradio as gr

from pipeline import VisionGuardPipeline


pipe = VisionGuardPipeline()
css = """
.gradio-container{max-width:1280px!important}
.hero{padding:20px 24px;border-radius:20px;background:linear-gradient(135deg,#12344d 0%,#1e6f86 55%,#7cc4b8 100%);color:#fff;margin-bottom:18px}
.hero h1{margin:0 0 8px 0;font-size:34px}
.hero p{margin:0;font-size:15px;opacity:.95}
.cardnote{padding:10px 14px;border:1px solid #d9e5ec;border-radius:14px;background:#f7fbfd}
"""


def _cls(mode):
    mp = {
        "all": None,
        "person": [0],
        "vehicle": [1, 2, 3, 5, 7],
    }
    return mp.get(mode, None)


def _fmt_meta(meta):
    return (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- search windows: `{meta['segments']}`"
    )


def build_idx(video, mode, sample_sec, win_sec, progress=gr.Progress()):
    if not video:
        return "upload a video first", None, None
    res = pipe.index_video(video, cls=_cls(mode), sample_sec=sample_sec, win_sec=win_sec, progress=progress)
    return _fmt_meta(res["meta"]), res["index_json"], gr.update(interactive=True)


def run_search(q, top_k, clip_pad):
    if not pipe.idx:
        return "index a video first", [], None, None, None, None, None
    if not q or not q.strip():
        return "enter a natural-language query", [], None, None, None, None, None
    hits = pipe.search(q.strip(), top_k=int(top_k))
    exp = pipe.export_hits(q.strip(), hits, clip_pad=clip_pad)
    rows = []
    md = [f"## matches for `{q}`", ""]
    gal = []
    if not exp["hits"]:
        md.append("no strong matches found")
    for i, x in enumerate(exp["hits"], 1):
        rows.append([i, round(x["score"], 4), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])])
        md.append(f"{i}. `{x['start']:.2f}s - {x['end']:.2f}s` score `{x['score']:.4f}`  ")
        md.append(f"   {x['summary']}")
        gal.append((x["frame_path"], f"{i}. {x['start']:.2f}s - {x['end']:.2f}s"))
    first = exp["hits"][0]["clip"] if exp["hits"] else None
    clips = [x["clip"] for x in exp["hits"]]
    files = [exp["html"], exp["csv"], exp["json"], exp["zip"]]
    best = exp["hits"][0] if exp["hits"] else None
    jump = None if best is None else f"top match: `{best['start']:.2f}s - {best['end']:.2f}s`"
    return "\n".join(md), rows, first, clips, files, exp["json"], gal if gal else jump


with gr.Blocks(title="VisionGuard AI", theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=css) as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>VisionGuard AI</h1>
  <p>Index CCTV footage once, search it with natural language, jump to the matched part, and export clips with timestamp records.</p>
</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="cctv video")
            mode = gr.Radio(["all", "person", "vehicle"], value="all", label="focus")
            sample_sec = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="sample every (sec)")
            win_sec = gr.Slider(2.0, 15.0, value=6.0, step=1.0, label="match window (sec)")
            idx_btn = gr.Button("index video", variant="primary")
            idx_md = gr.Markdown()
            idx_file = gr.File(label="index json")
            gr.Markdown("<div class='cardnote'>Use <b>all</b> if you want broad natural search. Use person or vehicle only when you want faster indexing on long videos.</div>")

            gr.Markdown("### query")
            q = gr.Textbox(placeholder="person sitting near gate, white car entering, group at entrance")
            top_k = gr.Slider(1, 10, value=5, step=1, label="top clips")
            clip_pad = gr.Slider(0.0, 6.0, value=2.0, step=0.5, label="clip padding (sec)")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            search_btn = gr.Button("search and export", interactive=False)

        with gr.Column(scale=2):
            out_md = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False)
            out_vid = gr.Video(label="top clip")
            out_clips = gr.Files(label="all clips")
            out_files = gr.Files(label="reports")
            raw_json = gr.File(label="raw search json")
            out_gallery = gr.Gallery(label="matched keyframes", columns=3, height="auto")

    idx_btn.click(build_idx, [video, mode, sample_sec, win_sec], [idx_md, idx_file, search_btn])
    search_btn.click(run_search, [q, top_k, clip_pad], [out_md, out_tbl, out_vid, out_clips, out_files, raw_json, out_gallery])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.queue().launch(server_name="0.0.0.0", share=share, show_error=True)
