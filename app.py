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
.status{padding:12px 14px;border-radius:14px;background:#eef6fa;border:1px solid #d4e5ef}
"""


def _fmt_meta(meta):
    return (
        f"video: `{os.path.basename(meta['video'])}`\n\n"
        f"- duration: `{meta['duration']:.2f}s`\n"
        f"- fps: `{meta['fps']:.2f}`\n"
        f"- sampled every: `{meta['sample_sec']:.2f}s`\n"
        f"- search windows: `{meta['segments']}`"
    )


def run_all(video, q, progress=gr.Progress()):
    if not video:
        return "upload a video first", "", [], None, None, None, None
    if not q or not q.strip():
        return "enter a natural-language query", "", [], None, None, None, None
    progress(0.01, desc="indexing video")
    res = pipe.index_video(video, cls=None, sample_sec=1.5, win_sec=6.0, progress=progress)
    progress(0.99, desc="ranking matches")
    hits = pipe.search(q.strip(), top_k=5)
    hits = pipe.verify(q.strip(), hits[:3], clip_pad=2.0, max_sec=8.0, progress=progress)
    hits = hits[:5]
    exp = pipe.export_hits(q.strip(), hits)
    rows = []
    md = [f"## matches for `{q}`", ""]
    gal = []
    if not exp["hits"]:
        md.append("no strong matches found")
    for i, x in enumerate(exp["hits"], 1):
        rows.append([i, round(x["qwen_score"], 4), round(x["start"], 2), round(x["end"], 2), x["summary"], ", ".join(x["objects"])])
        md.append(f"{i}. `{x['start']:.2f}s - {x['end']:.2f}s` score `{x['qwen_score']:.4f}`  ")
        md.append(f"   {x['summary']}")
        gal.append((x["frame_path"], f"{i}. {x['start']:.2f}s - {x['end']:.2f}s"))
    first = exp["hits"][0]["clip"] if exp["hits"] else None
    clips = [x["clip"] for x in exp["hits"]]
    files = [exp["html"], exp["csv"], exp["json"], exp["zip"]]
    status = "<div class='status'>scan complete</div>"
    return status, _fmt_meta(res["meta"]), rows, first, clips, files, gal if gal else None


with gr.Blocks(title="VisionGuard AI", theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=css) as demo:
    gr.HTML(
        """
<div class="hero">
  <h1>VisionGuard AI</h1>
  <p>Upload a video, write a query, scan the footage, then review separate matched clips and export the ones you need.</p>
</div>
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            video = gr.Video(label="cctv video")
            q = gr.Textbox(placeholder="person sitting near gate, white car entering, group at entrance")
            src = ["assets/asset1.mp4", "assets/asset2.mp4", "assets/asset3.mp4"]
            good = [x for x in src if os.path.exists(x)]
            if good:
                gr.Examples(good, inputs=video, label="sample videos")
            scan_btn = gr.Button("scan video", variant="primary")
            out_status = gr.HTML("<div class='status'>ready</div>")
            meta_md = gr.Markdown()
            gr.Markdown("<div class='cardnote'>The app uses broad retrieval first and verifies only the top few candidate clips to keep Colab usable.</div>")

        with gr.Column(scale=2):
            out_md = gr.Markdown()
            out_tbl = gr.Dataframe(headers=["rank", "score", "start", "end", "summary", "objects"], interactive=False, visible=False)
            out_vid = gr.Video(label="top clip")
            out_clips = gr.Files(label="all clips")
            out_files = gr.Files(label="reports")
            out_gallery = gr.Gallery(label="matched keyframes", columns=3, height="auto")

    scan_btn.click(run_all, [video, q], [out_status, meta_md, out_tbl, out_vid, out_clips, out_files, out_gallery])


if __name__ == "__main__":
    share = bool(os.getenv("COLAB_RELEASE_TAG") or os.getenv("KAGGLE_KERNEL_RUN_TYPE"))
    demo.queue().launch(server_name="0.0.0.0", share=share, show_error=True)
